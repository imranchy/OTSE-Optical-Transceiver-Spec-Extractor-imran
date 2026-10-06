
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CFG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())

AutoTokenizer.from_pretrained(CFG["model_id"], trust_remote_code=True)
q = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
AutoModelForCausalLM.from_pretrained(
    CFG["model_id"],
    quantization_config=q,
    dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)
print("4-bit base model load: OK")
print("Allocated GB:", round(torch.cuda.memory_allocated()/1024**3, 3))
print("Reserved GB:", round(torch.cuda.memory_reserved()/1024**3, 3))
