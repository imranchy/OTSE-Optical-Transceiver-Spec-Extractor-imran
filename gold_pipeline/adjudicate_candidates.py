from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_REPORTS = Path('gold_dataset/reports')
DEFAULT_RUNS = Path('gold_dataset/benchmark_runs')
DEFAULT_OUT = Path('gold_dataset/gold_candidates')

METADATA_FIELDS = {'notes', 'suggestions', 'provenance_datasheet_sections'}

# Strong source evidence is enough to promote these 14B-only scalar/identity facts.
# Relationship-sensitive optical limits, reach, temperatures, and interfaces are
# intentionally excluded.
SAFE_14B_ONLY_FIELDS = {
    'vendor', 'model', 'form_factor', 'frame', 'connector_type', 'fiber_type',
    'bit_rate_gbps', 'wavelength_band', 'wavelength_min_nm',
    'wavelength_max_nm', 'wavelength_center_nm', 'frequency_min_thz',
    'frequency_max_thz', 'channel_spacing_ghz', 'channel_total',
    'extinction_ratio_db', 'los_assert_dbm', 'los_deassert_dbm',
    'fec_type', 'modulation_format', 'media_interface_id_hex',
}

# 4B-only promotion is deliberately stricter because the baseline showed several
# semantic association errors even when a number existed in the source.
SAFE_4B_ONLY_FIELDS = {
    'vendor', 'model', 'form_factor', 'frame', 'connector_type', 'fiber_type',
    'wavelength_band', 'wavelength_min_nm', 'wavelength_max_nm',
    'wavelength_center_nm', 'frequency_min_thz', 'frequency_max_thz',
    'channel_spacing_ghz', 'channel_total',
}

FORM_FACTORS = ['QSFP-DD', 'QSFP28', 'QSFP+', 'SFP28', 'SFP+', 'OSFP', 'CFP2', 'CFP']


def parse_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s.lower() == 'nan':
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def dedupe(values: list[Any]) -> list[Any]:
    out = []
    seen = set()
    for v in values:
        key = stable_json(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def normalize_connector(value: Any) -> list[str]:
    result: list[str] = []
    for item in as_list(value):
        s = str(item).upper().strip()
        # Connector count and simplex/duplex are topology, not connector family.
        if 'MPO' in s or 'MTP' in s:
            result.append('MPO')
        elif re.search(r'\bLC\b', s):
            result.append('LC')
        elif re.search(r'\bSC\b', s):
            result.append('SC')
        else:
            s = re.sub(r'^\s*(?:\d+\s*[Xx]|[Xx]\s*\d+)\s*', '', s)
            s = re.sub(r'\b(?:DUPLEX|SIMPLEX)\b', '', s).strip(' /-')
            if s:
                result.append(s)
    return sorted(set(result))


def normalize_fiber(value: Any) -> list[str]:
    result: list[str] = []
    for item in as_list(value):
        s = str(item).upper().replace('-', ' ').strip()
        if s in {'SM', 'SMF', 'SINGLE MODE', 'SINGLEMODE', 'SINGLE MODE FIBER', 'SINGLE MODE FIBRE'}:
            result.append('SMF')
        elif s in {'MM', 'MMF', 'MULTI MODE', 'MULTIMODE', 'MULTI MODE FIBER', 'MULTI MODE FIBRE'}:
            result.append('MMF')
        else:
            result.append(str(item).strip())
    return sorted(set(result))


def source_dp_forms(source: str) -> set[str]:
    u = source.upper().replace(' ', '')
    found = set()
    aliases = {
        'QPSK': ['DP-QPSK', 'DPQPSK'],
        '8QAM': ['DP-8QAM', 'DP8QAM'],
        '16QAM': ['DP-16QAM', 'DP16QAM'],
        '64QAM': ['DP-64QAM', 'DP64QAM'],
    }
    for base, forms in aliases.items():
        if any(f.replace(' ', '') in u for f in forms):
            found.add('DP-' + base)
    return found


def normalize_modulation(value: Any, source: str) -> list[str]:
    dp_in_source = source_dp_forms(source)
    result = []
    for item in as_list(value):
        s = str(item).upper().replace(' ', '')
        s = s.replace('DPQPSK', 'DP-QPSK').replace('DP8QAM', 'DP-8QAM').replace('DP16QAM', 'DP-16QAM').replace('DP64QAM', 'DP-64QAM')
        if s in {'QPSK', '8QAM', '16QAM', '64QAM'} and ('DP-' + s) in dp_in_source:
            s = 'DP-' + s
        result.append(s)
    return sorted(set(result))


def normalized_for_field(field: str, value: Any, source: str) -> Any:
    if field == 'connector_type':
        return normalize_connector(value)
    if field == 'fiber_type':
        return normalize_fiber(value)
    if field == 'modulation_format':
        return normalize_modulation(value, source)
    return value


def first_source_text(runs_dir: Path, doc_id: str) -> str:
    path = runs_dir / doc_id / 'source' / 'content.md'
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def title_form_factor(source: str) -> str | None:
    # Prefer the title/front matter so mentions of compatible legacy cages deeper
    # in the document do not become the module's own form factor.
    head = source[:1800].upper()
    hits = []
    for ff in FORM_FACTORS:
        pos = head.find(ff.upper())
        if pos >= 0:
            hits.append((pos, ff))
    return min(hits)[1] if hits else None


def resolve_form_factor(v4: Any, v14: Any, source: str) -> tuple[Any | None, str | None]:
    a = [str(x) for x in as_list(v4)]
    b = [str(x) for x in as_list(v14)]
    title_ff = title_form_factor(source)
    if title_ff and (title_ff in a or title_ff in b):
        # Only resolve when one side offers the title-backed form factor as a
        # single unambiguous answer and the other is a noisy superset.
        if b == [title_ff] and title_ff in a:
            return [title_ff], 'TITLE_BACKED_14B_FORM_FACTOR'
        if a == [title_ff] and title_ff in b:
            return [title_ff], 'TITLE_BACKED_4B_FORM_FACTOR'
    return None, None


def number_near_operating_temperature(source: str, number: float) -> bool:
    pats = [r'operating\s+temperature', r'case\s+temperature', r'temperature\s+range']
    for m in re.finditer('|'.join(pats), source, flags=re.I):
        window = source[max(0, m.start()-120):m.end()+220]
        if re.search(rf'(?<![\d.]){re.escape(str(number))}(?![\d.])', window):
            return True
    return False


def resolve_temperature(field: str, v4: Any, v14: Any, source: str) -> tuple[Any | None, str | None]:
    if field not in {'temp_min_c', 'temp_max_c'}:
        return None, None
    vals = []
    for label, val in [('4B', v4), ('14B', v14)]:
        if isinstance(val, (int, float)):
            vals.append((label, float(val), number_near_operating_temperature(source, float(val))))
    direct = [x for x in vals if x[2]]
    if len(direct) == 1:
        return direct[0][1], f'OPERATING_TEMPERATURE_CONTEXT_{direct[0][0]}'
    return None, None


def contextual_tx_score(source: str, number: float, field: str) -> int:
    score = 0
    s = source
    num = re.escape(str(number))
    for m in re.finditer(num, s, flags=re.I):
        w = s[max(0, m.start()-180):m.end()+180].lower()
        if any(k in w for k in ['average launch power', 'transmit power', 'transmitter power', 'tx power', 'optical output power']):
            score += 3
        if 'off transmitter' in w or 'off-transmitter' in w or 'los assert' in w or 'los deassert' in w:
            score -= 4
        if field.endswith('_min_dbm') and 'min' in w:
            score += 1
        if field.endswith('_max_dbm') and 'max' in w:
            score += 1
    return score


def resolve_tx_power(field: str, v4: Any, v14: Any, source: str) -> tuple[Any | None, str | None]:
    if field not in {'tx_power_min_dbm', 'tx_power_max_dbm'}:
        return None, None
    if not isinstance(v4, list) or not isinstance(v14, list) or len(v4) != 1 or len(v14) != 1:
        return None, None
    try:
        n4, n14 = float(v4[0]), float(v14[0])
    except Exception:
        return None, None
    s4 = contextual_tx_score(source, n4, field)
    s14 = contextual_tx_score(source, n14, field)
    if s14 >= 3 and s14 >= s4 + 3:
        return [n14], 'TX_POWER_CONTEXT_14B'
    if s4 >= 3 and s4 >= s14 + 3:
        return [n4], 'TX_POWER_CONTEXT_4B'
    return None, None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description='Conservatively adjudicate OTSE gold-label candidates.')
    ap.add_argument('--reports-dir', type=Path, default=DEFAULT_REPORTS)
    ap.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    src = args.reports_dir / 'verification_candidates.csv'
    if not src.exists():
        raise SystemExit(f'Missing {src}. Run evidence_validator.py first.')

    with src.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    accepted: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    decisions = Counter()
    source_cache: dict[str, str] = {}

    for row in rows:
        doc = row['document_id']
        field = row['field']
        status = row['status']
        comparison = row['comparison']
        v4 = parse_cell(row.get('value_4b'))
        v14 = parse_cell(row.get('value_14b'))
        e4 = int(float(row.get('evidence_score_4b') or 0))
        e14 = int(float(row.get('evidence_score_14b') or 0))
        valid4 = str(row.get('atomic_valid_4b', '')).lower() == 'true'
        valid14 = str(row.get('atomic_valid_14b', '')).lower() == 'true'
        source = source_cache.setdefault(doc, first_source_text(args.runs_dir, doc))

        base = dict(row)

        if status == 'BOTH_EMPTY':
            decisions['BOTH_EMPTY'] += 1
            continue

        if field in METADATA_FIELDS or status == 'REVIEW_METADATA':
            base.update({'decision': 'IGNORE_METADATA', 'gold_value': '', 'rule': 'NOT_TRAINING_TARGET'})
            ignored.append(base)
            decisions['IGNORE_METADATA'] += 1
            continue

        # Existing high-confidence consensus remains accepted.
        if status == 'AUTO_ACCEPT' and valid4 and valid14:
            gold = normalized_for_field(field, v14, source)
            base.update({'decision': 'ACCEPT', 'gold_value': stable_json(gold), 'rule': 'MODEL_AGREEMENT_STRONG_EVIDENCE'})
            accepted.append(base)
            decisions['ACCEPT_MODEL_AGREEMENT'] += 1
            continue

        # If normalization turns a conflict into true agreement, accept it.
        if comparison == 'DISAGREE' and valid4 and valid14:
            n4 = normalized_for_field(field, v4, source)
            n14 = normalized_for_field(field, v14, source)
            if n4 == n14 and n4 not in (None, [], '') and min(e4, e14) >= 1:
                base.update({'decision': 'ACCEPT', 'gold_value': stable_json(n14), 'rule': f'NORMALIZED_{field.upper()}_AGREEMENT'})
                accepted.append(base)
                decisions['ACCEPT_NORMALIZED_AGREEMENT'] += 1
                continue

            gold, rule = resolve_form_factor(v4, v14, source) if field == 'form_factor' else (None, None)
            if gold is None:
                gold, rule = resolve_temperature(field, v4, v14, source)
            if gold is None:
                gold, rule = resolve_tx_power(field, v4, v14, source)
            if gold is not None:
                base.update({'decision': 'ACCEPT', 'gold_value': stable_json(gold), 'rule': rule})
                accepted.append(base)
                decisions['ACCEPT_CONTEXT_PRECEDENCE'] += 1
                continue

        # Strong 14B-only scalar evidence can be promoted for low-ambiguity fields.
        if comparison == '14B_ONLY' and valid14 and e14 >= 2 and field in SAFE_14B_ONLY_FIELDS:
            gold = normalized_for_field(field, v14, source)
            base.update({'decision': 'ACCEPT', 'gold_value': stable_json(gold), 'rule': '14B_ONLY_STRONG_SOURCE_EVIDENCE_SAFE_FIELD'})
            accepted.append(base)
            decisions['ACCEPT_14B_ONLY_SAFE'] += 1
            continue

        # Very conservative 4B-only promotion.
        if comparison == '4B_ONLY' and valid4 and e4 >= 2 and field in SAFE_4B_ONLY_FIELDS:
            gold = normalized_for_field(field, v4, source)
            base.update({'decision': 'ACCEPT', 'gold_value': stable_json(gold), 'rule': '4B_ONLY_STRONG_SOURCE_EVIDENCE_SAFE_FIELD'})
            accepted.append(base)
            decisions['ACCEPT_4B_ONLY_SAFE'] += 1
            continue

        # Evidence advantage can settle only low-ambiguity identity/scalar fields.
        if comparison == 'DISAGREE':
            if valid14 and e14 >= 2 and e4 == 0 and field in SAFE_14B_ONLY_FIELDS:
                gold = normalized_for_field(field, v14, source)
                base.update({'decision': 'ACCEPT', 'gold_value': stable_json(gold), 'rule': '14B_EVIDENCE_ADVANTAGE_SAFE_FIELD'})
                accepted.append(base)
                decisions['ACCEPT_14B_EVIDENCE_ADVANTAGE'] += 1
                continue
            if valid4 and e4 >= 2 and e14 == 0 and field in SAFE_4B_ONLY_FIELDS:
                gold = normalized_for_field(field, v4, source)
                base.update({'decision': 'ACCEPT', 'gold_value': stable_json(gold), 'rule': '4B_EVIDENCE_ADVANTAGE_SAFE_FIELD'})
                accepted.append(base)
                decisions['ACCEPT_4B_EVIDENCE_ADVANTAGE'] += 1
                continue

        # Everything else remains unresolved for semantic review.
        base.update({'decision': 'REVIEW', 'gold_value': '', 'rule': 'UNRESOLVED_CONSERVATIVE'})
        unresolved.append(base)
        decisions['REVIEW'] += 1

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ['decision', 'gold_value', 'rule']
    write_csv(out / 'accepted_fields.csv', accepted, fields)
    write_csv(out / 'unresolved.csv', unresolved, fields)
    write_csv(out / 'ignored_metadata.csv', ignored, fields)

    # Build partial gold specs. These are candidates, not final gold until unresolved
    # cases are adjudicated.
    by_doc: dict[str, dict[str, Any]] = defaultdict(dict)
    provenance: dict[str, dict[str, str]] = defaultdict(dict)
    for row in accepted:
        doc = row['document_id']
        field = row['field']
        by_doc[doc][field] = parse_cell(row['gold_value'])
        provenance[doc][field] = row['rule']

    cand_dir = out / 'partial_gold_specs'
    cand_dir.mkdir(exist_ok=True)
    for doc, specs in sorted(by_doc.items()):
        payload = {
            'document_id': doc,
            'status': 'PARTIAL_GOLD_CANDIDATE',
            'accepted_fields': specs,
            'adjudication_rules': provenance[doc],
            'warning': 'Do not use as final training gold until unresolved.csv is adjudicated.',
        }
        (cand_dir / f'{doc}.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    summary = {
        'input_rows': len(rows),
        'accepted_fields': len(accepted),
        'unresolved_fields': len(unresolved),
        'ignored_metadata': len(ignored),
        'both_empty_skipped': decisions.get('BOTH_EMPTY', 0),
        'decision_counts': dict(sorted(decisions.items())),
        'policy': {
            'modes_auto_adjudicated': False,
            'relationship_sensitive_fields': 'left unresolved unless exact model agreement already passed the prior validator',
            '14b_only_promotion': sorted(SAFE_14B_ONLY_FIELDS),
            '4b_only_promotion': sorted(SAFE_4B_ONLY_FIELDS),
        },
    }
    (out / 'adjudication_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Accepted:   {out / "accepted_fields.csv"}')
    print(f'Unresolved: {out / "unresolved.csv"}')


if __name__ == '__main__':
    main()
