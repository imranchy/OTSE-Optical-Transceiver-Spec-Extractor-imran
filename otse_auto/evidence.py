"""All-page, label-blind retrieval with stable blocks and actual tokenizer budgets."""
from __future__ import annotations
import re
from .core import GROUPS, messages_for, digest_obj, source_path

KEYWORDS = {
 'identity_and_physical': ['part number','ordering','temperature','operating','power consumption','power dissipation','connector','duplex','fiber','fibre','sfp','qsfp','osfp','cfp','msa','standard','compliant'],
 'interfaces_and_spectrum': ['wavelength','nm','frequency','thz','ghz','spacing','grid','channel','electrical interface','host interface','media interface','interface id','cmis','application','identifier','lan-wdm','cwdm','c-band','o-band'],
 'optical_performance_and_fec': ['receiver','transmitter','sensitivity','overload','optical power','dbm','osnr','fec','ber','baud','gbd','gbps','gb/s','reach','transmission distance','km','extinction','dispersion','pmd','pdl','los','assert','deassert','modulation','pam4','qpsk','nrz'],
}


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def quote_present(quote, text):
    return isinstance(quote, str) and len(normalized(quote)) >= 4 and normalized(quote) in normalized(text)


def read_pages(root, row):
    import fitz
    with fitz.open(source_path(root, row)) as doc:
        pages = [p.get_text('text', sort=True) for p in doc]
    if sum(len(p.strip()) for p in pages) < 80:
        raise ValueError('Insufficient native PDF text; excluded, not treated as absent: ' + row['document_id'])
    return pages


def make_blocks(pages):
    blocks = []
    for page, text in enumerate(pages, 1):
        lines = text.splitlines()
        # Retain original horizontal spacing, with moderate line overlap. Explicit block
        # IDs prevent repeated-page chunks being accidentally overwritten in a dict.
        for start in range(0, len(lines), 18):
            selected = lines[max(0, start - 3):start + 21]
            body = '\n'.join(selected).strip()
            if len(body) < 30:
                continue
            # Large standalone native-text lines get their own block; never trim values.
            blocks.append({'id': f'P{page:03d}B{start:05d}', 'page': page, 'start_line': max(0, start - 3), 'text': body})
    return blocks


def serialize_blocks(blocks):
    return '\n\n'.join(f"### PAGE {b['page']} BLOCK {b['id']}\n{b['text']}" for b in blocks)


def rank_blocks(blocks, group):
    words = KEYWORDS[group]
    def score(b):
        low = b['text'].casefold()
        weight = sum(min(3, low.count(k)) * (4 if len(k) > 5 else 1) for k in words)
        if group == 'identity_and_physical' and 'ordering' in low:
            weight += 30
        return weight
    return sorted(blocks, key=lambda b: (-score(b), b['page'], b['start_line']))


def prompt_tokens(tokenizer, messages):
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def choose_blocks(pages, group, max_chars=7000):
    blocks = make_blocks(pages)
    if not blocks:
        return []
    # Include the document title/identity context even for optical/interface groups.
    identity = blocks[0]
    ordered = [identity] + [b for b in rank_blocks(blocks, group) if b['id'] != identity['id']]
    selected = []
    for block in ordered:
        # Avoid near-duplicate spans selected by overlapping native-text windows.
        if any(b['page'] == block['page'] and abs(b['start_line'] - block['start_line']) < 10 for b in selected):
            continue
        proposed = selected + [block]
        if len(serialize_blocks(proposed)) <= max_chars:
            selected = proposed
    return sorted(selected, key=lambda b: (b['page'], b['start_line']))


def fit_blocks(pages, group, tokenizer, max_chars, max_length, output_reserve=1024):
    # Choose entirely from source text and field group, never from target values.
    # Shrinking happens BEFORE annotation, so references match the deployed input.
    cap = max_chars
    while cap >= 1400:
        selected = choose_blocks(pages, group, cap)
        source = serialize_blocks(selected)
        tokens = prompt_tokens(tokenizer, messages_for(group, source)) if selected else 10**9
        if selected and tokens + output_reserve <= max_length:
            return {'blocks': selected, 'source': source, 'source_view_sha256': digest_obj(source),
                    'prompt_tokens': tokens, 'evidence_characters': len(source),
                    'requested_character_cap': max_chars, 'effective_character_cap': cap,
                    'pages_selected': sorted({b['page'] for b in selected}), 'total_pdf_pages': len(pages)}
        cap -= 400
    raise ValueError('No evidence view fits the token budget for ' + group)


def alternate_order(blocks):
    # All evidence is retained; never change units, shuffle table rows, or delete support.
    if len(blocks) < 3:
        return list(blocks)
    return [blocks[0]] + list(reversed(blocks[1:]))
