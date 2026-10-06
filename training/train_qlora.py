
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--resume_from_checkpoint", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if cfg["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("Base model:", cfg["model_id"])

    dataset = load_dataset("json", data_files={
        "train": str((ROOT / cfg["train_file"]).resolve()),
        "validation": str((ROOT / cfg["validation_file"]).resolve()),
    })

    def to_prompt_completion(example):
        messages = example["messages"]

        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Each training example must end with an assistant message.")

        return {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }


    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset["train"].column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=qconf,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["gradient_checkpointing"],
    )

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )

    args_sft = SFTConfig(
        output_dir=str((ROOT / cfg["output_dir"]).resolve()),
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        logging_first_step=True,
        save_strategy=cfg["save_strategy"],
        eval_strategy=cfg["eval_strategy"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=cfg["bf16"],
        fp16=False,
        tf32=True,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        max_length=cfg["max_length"],
        packing=cfg["packing"],
        assistant_only_loss=False,
        completion_only_loss=cfg.get("completion_only_loss", True),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=args_sft,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = Path(args_sft.output_dir) / "adapter_final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    (Path(args_sft.output_dir) / "final_eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    print("Saved adapter:", final_dir)
    print("Final eval metrics:", metrics)

if __name__ == "__main__":
    main()
