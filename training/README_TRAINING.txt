
OTSE QLoRA training harness

Place this folder at repo_root\training\

Run in order:
1) python training\inspect_sequence_lengths.py
2) python training\smoke_load_4bit.py
3) python training\train_qlora.py

Configuration:
- Qwen/Qwen3-4B-Thinking-2507
- 4-bit NF4 + double quantization
- BF16 compute
- LoRA r=16, alpha=32, dropout=0.05
- target: q/k/v/o + gate/up/down projections
- batch size 1
- gradient accumulation 8
- max_length 4096
- assistant_only_loss=True
- 3 epochs
- validation/save every epoch
- load best checkpoint by eval_loss

Note: the base checkpoint is thinking-only. This training does not convert it into a non-thinking model.
