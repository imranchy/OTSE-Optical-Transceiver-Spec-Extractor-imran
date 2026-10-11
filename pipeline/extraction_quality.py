from __future__ import annotations

import re
from typing import Any

# Only values that can be recovered from unambiguous, explicit wording.
# These helpers never infer one field from another.

FORM_FACTORS = ("QSFP-DD", "QSFP28", "QSFP+", "OSFP", "CFP2-DCO", "CFP2", "SFP28", "SFP+")


def _uniq(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def deterministic_hints(text: str) -> dict[str, Any]:
    """Recover high-confidence atoms explicitly present in parsed datasheet text."""
    hints: dict[str, Any] = {}
    low = text.lower()

    # Form factor: exact token match only.
    found_ff = [ff for ff in FORM_FACTORS if re.search(rf"(?<![A-Za-z0-9]){re.escape(ff)}(?![A-Za-z0-9])", text, re.I)]
    if found_ff:
        hints["form_factor"] = _uniq(found_ff)

    # Vendor: conservative branding patterns, not a vendor dictionary.
    # Capture explicit manufacturer/vendor labels when present.
    m = re.search(r"(?im)^\s*(?:manufacturer|vendor|brand)\s*[:|]\s*([^|\n]{2,80})", text)
    if m:
        hints["vendor"] = m.group(1).strip()

    # Model/part number labels.
    models: list[str] = []
    for pat in (
        r"(?im)^\s*(?:part\s*(?:number|no\.?|#)|pn|p/n|model)\s*[:|#-]*\s*([A-Z0-9][A-Z0-9._/+\-]{3,})\s*$",
        r"(?im)^\s*([A-Z0-9][A-Z0-9._/+\-]{5,})\s*$",
    ):
        for mm in re.finditer(pat, text):
            value = mm.group(1).strip().rstrip(".,;:")
            if any(ch.isdigit() for ch in value) and value.upper() not in FORM_FACTORS:
                models.append(value)
        if models:
            break
    if models:
        hints["model"] = _uniq(models[:12])

    # Frequency ranges such as 191.3-196.1 THz or 191.28 – 196.12THz.
    freq_ranges = re.findall(r"(?<!\d)(1\d{2}\.\d+)\s*(?:-|–|—|to)\s*(1\d{2}\.\d+)\s*THz\b", text, re.I)
    if freq_ranges:
        mins = [float(a) for a, _ in freq_ranges]
        maxs = [float(b) for _, b in freq_ranges]
        hints["frequency_min_thz"] = _uniq(mins)
        hints["frequency_max_thz"] = _uniq(maxs)

    # Channel spacing/grid. Keep only explicit GHz values near spacing/grid/steps.
    spacing: list[float] = []
    for mm in re.finditer(r"(?i)(\d+(?:\.\d+)?)\s*GHz\s*(?:grid|spacing|steps?)", text):
        spacing.append(float(mm.group(1)))
    for mm in re.finditer(r"(?i)(?:grid|spacing|steps?)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*GHz", text):
        spacing.append(float(mm.group(1)))
    if spacing:
        hints["channel_spacing_ghz"] = _uniq(spacing)

    # Temperature ranges.
    temp_ranges = re.findall(r"(?<!\d)([-+]?\d+(?:\.\d+)?)\s*°?\s*C\s*(?:to|-|–|—)\s*([-+]?\d+(?:\.\d+)?)\s*°?\s*C", text, re.I)
    if temp_ranges:
        # Prefer the range nearest an operating-temperature label.
        labelled = re.search(r"(?is)(?:operating|case)\s+temperature.{0,80}?([-+]?\d+(?:\.\d+)?)\s*°?\s*C\s*(?:to|-|–|—)\s*([-+]?\d+(?:\.\d+)?)\s*°?\s*C", text)
        a, b = labelled.groups() if labelled else temp_ranges[0]
        hints["temp_min_c"] = float(a)
        hints["temp_max_c"] = float(b)

    # Power consumption: explicit phrase only.
    powers = [float(x) for x in re.findall(r"(?i)power\s+consumption[^\n|]{0,40}?(\d+(?:\.\d+)?)\s*W\b", text)]
    if powers:
        hints["power_consumption_w"] = _uniq(powers)

    # Bit rate and reach are broad but only captured when immediately expressed with units.
    bit_rates = [float(x) for x in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*G(?:b/s|bps|bE)?\b", text, re.I)]
    if bit_rates:
        hints["bit_rate_gbps"] = _uniq(bit_rates)

    reaches = [float(x) for x in re.findall(r"(?i)(?:reach|distance)[^\n|]{0,30}?(\d+(?:\.\d+)?)\s*km\b", text)]
    if reaches:
        hints["reach_km"] = _uniq(reaches)

    # Connector / fiber types from explicit tokens.
    connectors: list[str] = []
    for token in ("LC", "MPO", "SC", "FC", "ST"):
        if re.search(rf"(?<![A-Za-z]){token}(?![A-Za-z])", text, re.I):
            connectors.append(token)
    if connectors:
        hints["connector_type"] = _uniq(connectors)

    fibers: list[str] = []
    if re.search(r"\b(?:single[- ]?mode|SMF|SM)\b", text, re.I):
        fibers.append("SMF")
    if re.search(r"\b(?:multi[- ]?mode|MMF)\b", text, re.I):
        fibers.append("MMF")
    if fibers:
        hints["fiber_type"] = _uniq(fibers)

    # Common standards are literal strings, not inferred capabilities.
    standards = []
    for label, pat in (
        ("OIF 400ZR", r"\bOIF[- ]?400ZR\b|\bOIF\s+400ZR\b"),
        ("OpenZR+", r"\bOpenZR\+\b"),
        ("CMIS 4.0", r"\bCMIS\s*4(?:\.0)?\b"),
        ("CMIS 5.0", r"\bCMIS\s*5(?:\.0)?\b"),
        ("QSFP-DD MSA", r"\bQSFP-DD\s+MSA\b"),
    ):
        if re.search(pat, text, re.I):
            standards.append(label)
    if standards:
        hints["standards_claimed"] = standards

    # Explicit modulation tokens.
    mods = []
    for mod in ("NRZ", "PAM4", "QPSK", "8QAM", "16QAM", "64QAM"):
        if re.search(rf"\b{re.escape(mod)}\b", text, re.I):
            mods.append(mod)
    if mods:
        hints["modulation_format"] = mods

    # C/L/O band text only.
    bands = []
    for band, pat in (("C-band", r"\bC[- ]?Band\b"), ("L-band", r"\bL[- ]?Band\b"), ("O-band", r"\bO[- ]?Band\b")):
        if re.search(pat, text, re.I):
            bands.append(band)
    if bands:
        hints["wavelength_band"] = bands

    return hints


def meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def filled_fields(data: dict[str, Any]) -> set[str]:
    return {k for k, v in data.items() if meaningful_value(v)}


def evidence_expected_fields(text: str) -> set[str]:
    return set(deterministic_hints(text))


def coverage_issues(text: str, data: dict[str, Any]) -> list[str]:
    """Return evidence-backed missing-field issues; no minimum field-count heuristic."""
    expected = evidence_expected_fields(text)
    present = filled_fields(data)
    missing = sorted(expected - present)
    return [f"explicit_source_evidence_missing:{name}" for name in missing]


def merge_missing(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Fill only absent/empty values. Never overwrite an LLM-extracted non-empty value."""
    merged = dict(primary)
    for key, value in fallback.items():
        if key not in merged or not meaningful_value(merged.get(key)):
            merged[key] = value
    return merged
