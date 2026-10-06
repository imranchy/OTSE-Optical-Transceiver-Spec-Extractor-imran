from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

EMPTY_VALUES = (None, "", [], {})
METADATA_FIELDS = {"notes", "provenance_datasheet_sections", "suggestions"}

UNIT_PATTERNS = {
    "reach_km": [r"km"],
    "wavelength_min_nm": [r"nm"], "wavelength_max_nm": [r"nm"], "wavelength_center_nm": [r"nm"],
    "frequency_min_thz": [r"thz"], "frequency_max_thz": [r"thz"],
    "channel_spacing_ghz": [r"ghz"],
    "baud_rate_gbaud": [r"gbaud", r"gbd", r"gbd/s"],
    "bit_rate_gbps": [r"gb/s", r"gbps", r"gbit/s", r"gb/s"],
    "power_consumption_w": [r"w", r"watt"],
    "temp_min_c": [r"°?c", r"deg(?:ree)?s?\s*c"], "temp_max_c": [r"°?c", r"deg(?:ree)?s?\s*c"],
    "tx_power_min_dbm": [r"dbm"], "tx_power_max_dbm": [r"dbm"],
    "rx_sensitivity_dbm": [r"dbm"], "rx_overload_dbm": [r"dbm"],
    "los_assert_dbm": [r"dbm"], "los_deassert_dbm": [r"dbm"],
    "tx_in_band_osnr_db": [r"db"], "tx_out_of_band_osnr_db": [r"db"],
    "rx_osnr_db": [r"db"], "rx_osnr_tolerance_db_max": [r"db"],
    "cd_tolerance_ps_nm": [r"ps/nm", r"ps\s*/\s*nm"], "pdl_db": [r"db"],
    "pmd_tolerance_ps": [r"ps"], "extinction_ratio_db": [r"db"],
    "fec_overhead_percent": [r"%"],
}

CONTEXT_TERMS = {
    "reach_km": ["reach", "distance", "link length", "transmission"],
    "wavelength_min_nm": ["wavelength"], "wavelength_max_nm": ["wavelength"], "wavelength_center_nm": ["wavelength", "lambda"],
    "frequency_min_thz": ["frequency"], "frequency_max_thz": ["frequency"],
    "channel_spacing_ghz": ["spacing", "grid", "channel"],
    "baud_rate_gbaud": ["baud", "symbol rate"],
    "bit_rate_gbps": ["bit rate", "data rate", "signaling rate", "aggregate"],
    "power_consumption_w": ["power", "consumption"],
    "temp_min_c": ["temperature", "operating"], "temp_max_c": ["temperature", "operating"],
    "tx_power_min_dbm": ["launch power", "transmit", "transmitter", "tx power", "average launch"],
    "tx_power_max_dbm": ["launch power", "transmit", "transmitter", "tx power", "average launch"],
    "rx_sensitivity_dbm": ["sensitivity", "receiver sensitivity"],
    "rx_overload_dbm": ["overload", "maximum receive", "receiver overload"],
    "los_assert_dbm": ["los assert", "los_assert", "losa"], "los_deassert_dbm": ["los de-assert", "los deassert", "losd"],
    "tx_in_band_osnr_db": ["osnr"], "tx_out_of_band_osnr_db": ["osnr"], "rx_osnr_db": ["osnr"], "rx_osnr_tolerance_db_max": ["osnr"],
    "cd_tolerance_ps_nm": ["chromatic dispersion", "cd"], "pdl_db": ["pdl", "polarization dependent loss"],
    "pmd_tolerance_ps": ["pmd", "dgd", "polarization mode dispersion"], "extinction_ratio_db": ["extinction ratio"],
    "fec_overhead_percent": ["fec", "overhead"],
}

ALIASES = {
    "fiber_type": {
        "smf": ["smf", "single mode", "single-mode", "singlemode"],
        "sm": ["smf", "single mode", "single-mode", "singlemode"],
        "mmf": ["mmf", "multi mode", "multi-mode", "multimode"],
        "mm": ["mmf", "multi mode", "multi-mode", "multimode"],
    },
    "modulation_format": {
        "dp-16qam": ["dp-16qam", "dp 16qam", "dual polarization 16qam", "dual-polarization 16-qam", "16qam"],
        "pam4": ["pam4", "pam-4", "4-level pulse amplitude"],
        "nrz": ["nrz", "non-return-to-zero"],
    },
    "fec_type": {
        "cfec": ["cfec", "c-fec"], "ofec": ["ofec", "o-fec"],
        "rs-fec": ["rs-fec", "rs fec", "reed-solomon", "rs(528,514)", "rs(544,514)"],
    },
}

def load_json(path: Path, default=None):
    if not path.exists(): return default
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default

def is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}

def as_list(v: Any) -> list[Any]:
    if is_empty(v): return []
    return v if isinstance(v, list) else [v]

def norm_text(s: Any) -> str:
    s = str(s).lower().replace("×", "x").replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9.+%-]+", " ", s)
    return " ".join(s.split())

def norm_item(v: Any) -> Any:
    if isinstance(v, bool): return v
    if isinstance(v, (int,float)):
        return round(float(v), 8)
    return norm_text(v)

def norm_value(v: Any) -> tuple:
    vals=[norm_item(x) for x in as_list(v)]
    return tuple(sorted(vals, key=lambda x: str(x)))

def compare_values(a: Any, b: Any) -> str:
    if is_empty(a) and is_empty(b): return "BOTH_EMPTY"
    if norm_value(a) == norm_value(b): return "AGREE"
    if is_empty(a): return "14B_ONLY"
    if is_empty(b): return "4B_ONLY"
    return "DISAGREE"

def _num_variants(v: float) -> list[str]:
    out={str(v), f"{v:g}"}
    if float(v).is_integer(): out.add(str(int(v)))
    return sorted(out, key=len, reverse=True)

def find_numeric_evidence(text: str, field: str, value: float):
    low=text.lower().replace("−", "-").replace("–", "-")
    units=UNIT_PATTERNS.get(field, [])
    contexts=CONTEXT_TERMS.get(field, [])
    best=None
    for nv in _num_variants(float(value)):
        pat=rf"(?<![\d.])-?{re.escape(nv.lstrip('-'))}(?![\d.])"
        if float(value)<0: pat=rf"(?<![\d.])-{re.escape(nv.lstrip('-'))}(?![\d.])"
        for m in re.finditer(pat, low):
            lo=max(0,m.start()-180); hi=min(len(low),m.end()+180); win=low[lo:hi]
            unit_ok = (not units) or any(re.search(u, win, re.I) for u in units)
            context_ok = (not contexts) or any(c in win for c in contexts)
            score = 2 if unit_ok and context_ok else 1 if unit_ok else 0
            candidate=(score, text[lo:hi].replace("\n"," ").strip())
            if best is None or candidate[0]>best[0]: best=candidate
    return best or (0, "")

def find_string_evidence(text: str, field: str, value: str):
    low=norm_text(text); nv=norm_text(value)
    variants=[nv]
    amap=ALIASES.get(field,{})
    if nv in amap: variants += [norm_text(x) for x in amap[nv]]
    # Handle common composite tokens conservatively.
    variants += [nv.replace(" ", "-")]
    for v in dict.fromkeys(variants):
        if v and v in low:
            return 2, v
    # weak token overlap for long standards/codes
    toks=[t for t in re.findall(r"[a-z0-9]+", nv) if len(t)>=3]
    if len(toks)>=2 and sum(t in low for t in toks) >= max(2, int(len(toks)*0.75)):
        return 1, "token-overlap"
    return 0, ""

def evidence_for_value(text: str, field: str, value: Any):
    if is_empty(value): return {"score":0,"supported":False,"strong":False,"excerpts":[]}
    scores=[]; excerpts=[]
    for item in as_list(value):
        if isinstance(item,(int,float)) and not isinstance(item,bool):
            sc,ex=find_numeric_evidence(text,field,float(item))
        else:
            sc,ex=find_string_evidence(text,field,str(item))
        scores.append(sc)
        if ex: excerpts.append(ex[:500])
    score=min(scores) if scores else 0
    return {"score":score,"supported":score>=1,"strong":score>=2,"excerpts":excerpts[:4]}

def iter_documents(runs_root: Path):
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        src=d/"source"/"content.md"
        if not src.exists(): continue
        text=src.read_text(encoding="utf-8",errors="replace")
        # Ignore obsolete/failed smoke runs with effectively no parsed source.
        if len(text.strip()) < 500: continue
        yield d,text

def metrics_for(doc: Path, tag: str) -> dict:
    return load_json(doc/tag/"metrics.json", {}) or {}
