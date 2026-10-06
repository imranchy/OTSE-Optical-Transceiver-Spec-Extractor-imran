
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(CFG["model_id"], trust_remote_code=True)

def load_jsonl(path):
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]

def rendered_len(rec):
    text = tok.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])

for split_key in ("train_file", "validation_file"):
    path = ROOT / CFG[split_key]
    rows = load_jsonl(path)
    lengths = [rendered_len(r) for r in rows]
    s = sorted(lengths)
    p95 = s[min(len(s)-1, int(len(s)*0.95))]
    print(f"{split_key}: samples={len(rows)} min={min(lengths)} median={s[len(s)//2]} p95={p95} "
          f"max={max(lengths)} over_{CFG['max_length']}={sum(x > CFG['max_length'] for x in lengths)}")
