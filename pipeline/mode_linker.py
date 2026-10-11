"""Direct, schema-constrained extraction of application-mode tables."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pipeline.atom_extractor import BASE_URL, _schema_for_llm, extract_schema
from pipeline.pdf_parser import ParsedDatasheet
from transceiver_models import ApplicationModeSet, MODE_FIELDS

logger = logging.getLogger(__name__)

MODE_SYSTEM_PROMPT = """You extract optical-transceiver application modes.

Each output item represents exactly one selectable operating-mode table row.
Copy only values that co-occur in that row. Never mix values from separate
rows, never infer a missing value, and never output glossary definitions.
Use null for a missing scalar. Return all source rows from OIF, OpenZR+,
OpenROADM, or equivalent application-mode tables.

Mapping rules:
- Copy the value from an "Application mode", "Application code", or "CMIS
  Application Code" column into application_code, even if it is only a number.
- label is a readable name such as "OpenZR+ app 4"; it must not replace
  application_code.
- Map a "Line rate" or "Payload" value such as 400G to payload_gbps as 400.
"""

MODE_PROMPT_TEMPLATE = """## Output JSON schema
```json
{schema}
```

## Module-level context
```json
{general}
```

## Datasheet content
{content}

Extract application-mode rows only. Return one JSON object conforming exactly to the schema.
"""


def separate_atoms(specs_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retained for UI/report compatibility; rows now come directly from source."""
    multi: dict[str, Any] = {}
    general: dict[str, Any] = {}
    for key, value in specs_dict.items():
        if key in MODE_FIELDS and isinstance(value, list) and len(value) > 1:
            multi[key] = value
        elif value is not None:
            general[key] = value
    return multi, general


def _filter_sections_by_provenance(
    markdown: str, provenance: str | list[str]
) -> str:
    """Return source sections named by provenance for the UI preview.

    The extraction pipeline no longer depends on this filter, but the Streamlit
    UI uses it to show the operator which sections supported an extraction.
    """
    targets = provenance if isinstance(provenance, list) else re.split(r"[,;\n]", provenance)
    normalized_targets = {
        re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()
        for value in targets
        if value.strip()
    }
    if not normalized_targets:
        return markdown

    sections: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []
    for line in markdown.splitlines(keepends=True):
        if re.match(r"^#{1,6}\s+", line):
            if heading or body:
                sections.append((heading, body))
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            body = [line]
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, body))

    selected = []
    for section_heading, section_body in sections:
        normalized_heading = re.sub(r"[^a-z0-9 ]", "", section_heading.lower()).strip()
        if any(target in normalized_heading or normalized_heading in target for target in normalized_targets):
            selected.append("".join(section_body))
    return "\n".join(selected) if selected else markdown


def _mode_content(parsed: ParsedDatasheet) -> str:
    content = parsed.markdown or parsed.full_text
    return content if len(content) <= 80_000 else content[:80_000] + "\n\n[...truncated...]"


def synthesize_modes(
    specs_dict: dict[str, Any],
    parsed: ParsedDatasheet,
    *,
    run_dir: Path | None = None,
    model_id: str | None = None,
    base_url: str = BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Extract mode rows directly, rather than reconstructing detached atoms."""
    multi, general = separate_atoms(specs_dict)
    prompt = MODE_PROMPT_TEMPLATE.format(
        schema=json.dumps(_schema_for_llm(ApplicationModeSet), indent=2),
        general=json.dumps(general, indent=2),
        content=_mode_content(parsed),
    )
    result, raw = extract_schema(
        model_type=ApplicationModeSet,
        schema_name="application_modes",
        system_prompt=MODE_SYSTEM_PROMPT,
        user_prompt=prompt,
        model_id=model_id or "otse-qwen4b-final",
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if run_dir is not None:
        (run_dir / "raw_llm_modes_response.json").write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )
    if result is None:
        return None, multi, general
    data = result.model_dump(mode="json")
    logger.info("Extracted %d application modes directly from source rows", len(data["modes"]))
    return data, multi, general


def store_modes(run_dir: Path, modes: dict[str, Any], multi: dict[str, Any], general: dict[str, Any]) -> None:
    """Save direct mode extraction together with legacy UI context."""
    (run_dir / "raw_llm_modes.json").write_text(json.dumps(modes, indent=2), encoding="utf-8")
    (run_dir / "modes.json").write_text(json.dumps(modes, indent=2), encoding="utf-8")
    (run_dir / "mode_atoms.json").write_text(
        json.dumps({"multi_valued": multi, "general": general}, indent=2), encoding="utf-8"
    )
