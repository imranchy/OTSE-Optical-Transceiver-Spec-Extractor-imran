"""Step 2: robust atomic parameter extraction via focused LLM passes.

This version keeps the public extract_atoms() API intact while avoiding a common
failure mode: a Pydantic-valid object where essentially every useful field is
null/empty even though explicit evidence is present in the datasheet.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from transceiver_models import TransceiverSpecs
from pipeline.pdf_parser import ParsedDatasheet
from pipeline.extraction_quality import coverage_issues, deterministic_hints, merge_missing

logger = logging.getLogger(__name__)
BASE_URL = "http://127.0.0.1:11434"

_LIST_FIELDS: set[str] = {
    f for f, info in TransceiverSpecs.model_fields.items()
    if "list" in str(info.annotation).lower()
}

# Small, semantically coherent passes work better on 4-bit local models than one
# very large all-fields request, while preserving the same final schema.
FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity_and_physical", (
        "vendor", "model", "form_factor", "standards_claimed", "connector_type",
        "fiber_type", "temp_min_c", "temp_max_c", "power_consumption_w",
        "provenance_datasheet_sections", "notes",
    )),
    ("interfaces_and_spectrum", (
        "wavelength_min_nm", "wavelength_max_nm", "wavelength_center_nm",
        "wavelength_band", "frequency_min_thz", "frequency_max_thz",
        "channel_spacing_ghz", "channel_total", "host_interface_name",
        "host_interface_id_hex", "host_interface_id", "media_interface_name",
        "media_interface_id_hex", "media_interface_id", "frame", "standards_code",
        "provenance_datasheet_sections", "notes",
    )),
    ("optical_performance_and_fec", (
        "baud_rate_gbaud", "bit_rate_gbps", "roll_off_percent", "modulation_format",
        "reach_km", "tx_power_min_dbm", "tx_power_max_dbm", "tx_in_band_osnr_db",
        "tx_out_of_band_osnr_db", "rx_sensitivity_dbm", "rx_osnr_db",
        "rx_osnr_tolerance_db_max", "rx_overload_dbm", "cd_tolerance_ps_nm",
        "pdl_db", "pmd_tolerance_ps", "fec_type", "fec_overhead_percent",
        "pre_fec_ber_threshold", "post_fec_ber_target", "extinction_ratio_db",
        "los_assert_dbm", "los_deassert_dbm", "provenance_datasheet_sections", "notes",
    )),
)

SYSTEM_PROMPT = """You are an optical-transceiver datasheet extraction engine.
Extract only facts explicitly supported by the supplied datasheet text.
Rules:
- Copy numeric values exactly as represented semantically; do not invent, convert units, or calculate.
- If a requested field is absent, return null (or [] for list fields when appropriate).
- Collect all distinct explicitly stated values for list fields.
- Keep module-level values separate from application/mode-specific values; do not guess correlations here.
- A model/part number is not a form factor. A connector is not a host/media interface code.
- GHz grid/channel spacing is not Gbps bit rate.
- Transmitter output power, receiver sensitivity, receiver overload, and OSNR are distinct fields.
- Standards, FEC, modulation, reach, interfaces, and application codes must be copied from source wording.
- Do not infer a standard merely because another compatible standard is present.
- Return ONLY one valid JSON object with the requested keys. No prose or markdown fences.
"""

USER_PROMPT_TEMPLATE = """## Requested fields
{schema}

## Section headings
{headings}

## Datasheet content
{content}

Return one JSON object containing only the requested fields.
"""

RETRY_PROMPT_TEMPLATE = """The previous extraction missed fields for which explicit source evidence exists.
Extract ONLY these fields and return one JSON object:
{fields}

High-confidence source cues already detected (use these only if the text itself supports them):
{hints}

Datasheet content:
{content}
"""


def _coerce_types(raw: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(raw.items()):
        if value is None:
            continue
        if key in _LIST_FIELDS:
            if not isinstance(value, list):
                raw[key] = [value]
        elif isinstance(value, list) and len(value) == 1:
            raw[key] = value[0]
    return raw


def _parse_json_from_response(raw: str) -> Any:
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        value = json.loads(s)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(s[start:end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}

def _schema_for_llm(model_type: type) -> dict[str, Any]:
    """Return a compact JSON schema for an LLM-facing structured extraction task."""
    schema = model_type.model_json_schema()

    # Remove metadata that is useful for documentation but not necessary
    # for the model prompt.
    schema.pop("title", None)

    return schema


def extract_schema(
    *,
    model_type: type,
    schema_name: str,
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    base_url: str = BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> tuple[Any | None, dict[str, Any]]:
    """Run a generic schema-constrained extraction and validate with Pydantic."""

    client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="no-key",
        timeout=600.0,
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    text = choice.message.content or ""

    if not text.strip():
        text = getattr(choice.message, "reasoning_content", None) or ""

    raw: dict[str, Any] = {
        "schema_name": schema_name,
        "model_id": model_id,
        "finish_reason": choice.finish_reason,
        "response_text": text,
    }

    if not text.strip():
        logger.warning(
            "Empty LLM response for schema=%s (finish_reason=%s)",
            schema_name,
            choice.finish_reason,
        )
        return None, raw

    parsed = _parse_json_from_response(text)
    raw["parsed"] = parsed

    if not isinstance(parsed, dict) or not parsed:
        logger.warning("Could not parse valid JSON for schema=%s", schema_name)
        return None, raw

    try:
        result = model_type.model_validate(parsed)
        return result, raw
    except ValidationError as exc:
        logger.warning(
            "Schema validation failed for %s: %d errors",
            schema_name,
            exc.error_count(),
        )

        raw["validation_errors"] = exc.errors()

        return None, raw


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n\n[...truncated...]"


def _extract_headings(md_text: str) -> str:
    headings = [line.strip() for line in md_text.splitlines() if re.match(r"^#{1,6}\s+", line)]
    return "\n".join(headings) if headings else "(none found)"


def _schema_for_fields(fields: tuple[str, ...] | list[str]) -> str:
    schema = TransceiverSpecs.model_json_schema().get("properties", {})
    compact: dict[str, Any] = {}
    for name in fields:
        info = schema.get(name, {})
        compact[name] = {
            "type": info.get("type", info.get("anyOf", "")),
            "description": info.get("description", ""),
        }
    return json.dumps(compact, indent=2)


def _completion(client: OpenAI, model_id: str, prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    if not text.strip():
        text = getattr(choice.message, "reasoning_content", None) or ""
    if not text.strip():
        logger.warning("Empty LLM response (finish_reason=%s)", choice.finish_reason)
        return {}
    parsed = _parse_json_from_response(text)
    return parsed if isinstance(parsed, dict) else {}


def _merge_pass(target: dict[str, Any], partial: dict[str, Any], allowed: set[str]) -> None:
    for key, value in partial.items():
        if key in allowed and value is not None:
            # Merge lists without duplicates; otherwise take latest explicit value.
            if isinstance(value, list) and isinstance(target.get(key), list):
                target[key] = list(dict.fromkeys([*target[key], *value]))
            elif key not in target or target.get(key) in (None, [], ""):
                target[key] = value


def extract_atoms(
    parsed: ParsedDatasheet,
    *,
    model_id: str | None = None,
    max_chars: int = 120_000,
    base_url: str = BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> tuple[TransceiverSpecs | None, dict[str, Any]]:
    """Extract atoms with focused passes, deterministic recovery, and evidence coverage validation.

    The return signature is unchanged: (validated_specs, raw_dict).
    """
    client = OpenAI(base_url=f"{base_url}/v1", api_key="no-key", timeout=600.0)
    source = parsed.markdown or parsed.full_text
    content = _truncate(source, max_chars)
    headings = _extract_headings(source)

    if not model_id:
        models_resp = client.models.list()
        model_id = models_resp.data[0].id if models_resp.data else "default"

    merged: dict[str, Any] = {}
    for group_name, fields in FIELD_GROUPS:
        prompt = USER_PROMPT_TEMPLATE.format(
            schema=_schema_for_fields(fields),
            headings=headings,
            content=content,
        )
        logger.info("Atomic extraction pass=%s fields=%d model=%s", group_name, len(fields), model_id)
        partial = _completion(client, model_id, prompt, temperature, max_tokens)
        _merge_pass(merged, partial, set(fields))

    merged = _coerce_types(merged)

    # Deterministic recovery is deliberately conservative and only fills blanks.
    hints = deterministic_hints(source)
    merged = merge_missing(merged, hints)

    # If explicit evidence still has no corresponding output, run one small targeted retry.
    issues = coverage_issues(source, merged)
    if issues:
        missing_fields = [issue.split(":", 1)[1] for issue in issues]
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            fields=_schema_for_fields(missing_fields),
            hints=json.dumps({k: v for k, v in hints.items() if k in missing_fields}, indent=2),
            content=content,
        )
        logger.warning("Coverage retry for fields: %s", ", ".join(missing_fields))
        retry = _completion(client, model_id, retry_prompt, 0.0, max_tokens)
        _merge_pass(merged, retry, set(missing_fields))
        merged = _coerce_types(merge_missing(merged, hints))

    final_issues = coverage_issues(source, merged)
    if final_issues:
        logger.error("Extraction rejected by evidence coverage gate: %s", final_issues)
        merged["suggestions"] = "Extraction incomplete: " + "; ".join(final_issues)
        return None, merged

    try:
        specs = TransceiverSpecs.model_validate(merged)
        logger.info("Extraction validated successfully with evidence coverage gate.")
        return specs, merged
    except ValidationError as exc:
        logger.warning("Validation failed: %s", exc.error_count())
        for err in exc.errors():
            logger.debug("  %s: %s", err["loc"], err["msg"])
        return None, merged
