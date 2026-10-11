from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OTSE QLoRA adapter."
    )
    parser.add_argument(
        "--config",
        default=str(HERE / "config.json"),
        help="Path to training config JSON.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Optional checkpoint directory to resume training from.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Environment checks
    # ------------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if cfg.get("bf16", False) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported by this GPU.")

    print("=" * 78)
    print("OTSE QLoRA training")
    print("=" * 78)
    print("Config:", config_path)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("Base model:", cfg["model_id"])
    print("Max length:", cfg["max_length"])
    print("Maximum epochs:", cfg["num_train_epochs"])
    print("Learning rate:", cfg["learning_rate"])
    print()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    train_path = (ROOT / cfg["train_file"]).resolve()
    validation_path = (ROOT / cfg["validation_file"]).resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {validation_path}"
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )

    print("Raw dataset:")
    print("  train:", len(dataset["train"]))
    print("  validation:", len(dataset["validation"]))

    # ------------------------------------------------------------------
    # Convert messages format -> prompt/completion format.
    #
    # We deliberately retain completion-only training because this was the
    # working OTSE training path and avoids relying on assistant-only masks.
    # ------------------------------------------------------------------

    def to_prompt_completion(example: dict) -> dict:
        messages = example["messages"]

        if not messages:
            raise ValueError("Training example contains no messages.")

        if messages[-1].get("role") != "assistant":
            raise ValueError(
                "Each training example must end with an assistant message."
            )

        return {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }

    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset["train"].column_names,
        desc="Converting to prompt/completion format",
    )

    print()
    print("Prompt/completion dataset:")
    print("  train:", len(dataset["train"]))
    print("  validation:", len(dataset["validation"]))
    print()

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Tokenizer loaded.")
    print("Pad token:", repr(tokenizer.pad_token))
    print("EOS token:", repr(tokenizer.eos_token))
    print()

    # ------------------------------------------------------------------
    # 4-bit QLoRA base model
    # ------------------------------------------------------------------

    print("Loading 4-bit base model...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg[
            "bnb_4bit_use_double_quant"
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Training and gradient checkpointing require cache to be disabled.
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg[
            "gradient_checkpointing"
        ],
    )

    print("Base model loaded.")
    print(
        "GPU allocated GB:",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
    )
    print(
        "GPU reserved GB:",
        round(torch.cuda.memory_reserved() / 1024**3, 3),
    )
    print()

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )

    output_dir = (ROOT / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", output_dir)
    print()
    print("LoRA configuration:")
    print("  r:", cfg["lora_r"])
    print("  alpha:", cfg["lora_alpha"])
    print("  dropout:", cfg["lora_dropout"])
    print("  target modules:")
    for module_name in cfg["target_modules"]:
        print("   -", module_name)
    print()

    # ------------------------------------------------------------------
    # TRL / SFT training arguments
    # ------------------------------------------------------------------

    training_args = SFTConfig(
        output_dir=str(output_dir),

        # Training duration
        num_train_epochs=cfg["num_train_epochs"],

        # Batch configuration
        per_device_train_batch_size=cfg[
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=cfg[
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=cfg[
            "gradient_accumulation_steps"
        ],

        # Optimizer
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_steps=cfg.get("warmup_steps", 0),
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        # Precision
        bf16=cfg.get("bf16", True),
        fp16=False,

        # Memory
        gradient_checkpointing=cfg.get(
            "gradient_checkpointing",
            True,
        ),

        # Sequence handling
        max_length=cfg["max_length"],
        packing=cfg.get("packing", False),

        # We train only on the assistant completion.
        completion_only_loss=cfg.get(
            "completion_only_loss",
            True,
        ),

        # Logging
        logging_strategy="steps",
        logging_steps=cfg.get("logging_steps", 1),

        # Validation
        eval_strategy=cfg.get(
            "eval_strategy",
            "epoch",
        ),

        # Checkpoints
        save_strategy=cfg.get(
            "save_strategy",
            "epoch",
        ),
        save_total_limit=cfg.get(
            "save_total_limit",
            3,
        ),

        # Restore the checkpoint with the best validation loss.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Reproducibility
        seed=cfg.get("seed", 42),
        data_seed=cfg.get("seed", 42),

        # Misc
        report_to="none",
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------

    print("Creating SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.get(
                    "early_stopping_patience",
                    2,
                ),
                early_stopping_threshold=cfg.get(
                    "early_stopping_threshold",
                    0.001,
                ),
            )
        ],
    )

    # ------------------------------------------------------------------
    # Trainable parameter report
    # ------------------------------------------------------------------

    trainable_params = 0
    total_params = 0

    for parameter in trainer.model.parameters():
        count = parameter.numel()
        total_params += count

        if parameter.requires_grad:
            trainable_params += count

    trainable_percent = (
        100.0 * trainable_params / total_params
        if total_params
        else 0.0
    )

    print()
    print("=" * 78)
    print("Model parameter summary")
    print("=" * 78)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable percentage: {trainable_percent:.4f}%")
    print()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print("=" * 78)
    print("Starting training")
    print("=" * 78)

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    print()
    print("=" * 78)
    print("Training finished")
    print("=" * 78)

    train_metrics = dict(train_result.metrics)

    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("Running final validation")
    print("=" * 78)

    eval_metrics = trainer.evaluate()

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ------------------------------------------------------------------
    # Save final/best adapter
    # ------------------------------------------------------------------

    adapter_dir = output_dir / "adapter_final"

    print()
    print("Saving final adapter to:")
    print(adapter_dir)

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # ------------------------------------------------------------------
    # Save consolidated metrics
    # ------------------------------------------------------------------

    final_metrics = {
        "base_model": cfg["model_id"],
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "max_length": cfg["max_length"],
        "maximum_epochs": cfg["num_train_epochs"],
        "learning_rate": cfg["learning_rate"],
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": trainable_percent,
        "lora": {
            "r": cfg["lora_r"],
            "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"],
            "target_modules": cfg["target_modules"],
        },
        "quantization": {
            "load_in_4bit": True,
            "quant_type": cfg["bnb_4bit_quant_type"],
            "double_quant": cfg[
                "bnb_4bit_use_double_quant"
            ],
            "compute_dtype": "bfloat16",
        },
        "early_stopping": {
            "patience": cfg.get(
                "early_stopping_patience",
                2,
            ),
            "threshold": cfg.get(
                "early_stopping_threshold",
                0.001,
            ),
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
    }

    metrics_path = output_dir / "final_eval_metrics.json"

    metrics_path.write_text(
        json.dumps(
            final_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("QLoRA training complete")
    print("=" * 78)
    print("Adapter:", adapter_dir)
    print("Metrics:", metrics_path)
    print(
        "Best checkpoint:",
        trainer.state.best_model_checkpoint,
    )
    print(
        "Best validation metric:",
        trainer.state.best_metric,
    )
    print()


if __name__ == "__main__":
    main()