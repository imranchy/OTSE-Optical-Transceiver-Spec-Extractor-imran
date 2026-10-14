"""OTSE: sequential DAPT -> SFT on a common Qwen3-4B-Instruct base.
Run from repository root, in the original training virtual environment.
The historical SFT-only adapter is left untouched.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

BASE = 'Qwen/Qwen3-4B-Instruct-2507'
ROOT = Path(__file__).resolve().parents[3] if Path(__file__).resolve().parent.name == 'scripts' else Path.cwd()
DATA = ROOT / 'experiments/dapt_vs_sft/data'
OUT = ROOT / 'experiments/dapt_vs_sft/outputs'
SFT_DATA = ROOT / 'data/corpus_v2/pseudo_qlora'
LORA_MODULES = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def records(path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={'': 0}, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tok


def config(path, epochs, lr, max_length, is_dapt):
    return SFTConfig(
        output_dir=str(path), num_train_epochs=epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=lr,
        lr_scheduler_type='cosine', warmup_steps=2,
        optim='paged_adamw_8bit', bf16=True, fp16=False,
        gradient_checkpointing=True, max_length=max_length,
        packing=is_dapt, completion_only_loss=False if is_dapt else True,
        dataset_text_field='text',
        eval_strategy='epoch', save_strategy='no',
        logging_steps=10, logging_strategy='steps',
        report_to='none', seed=42, data_seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )


def run_phase(name, model, tok, train_ds, val_ds, args, peft_config=None):
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    cfg = config(path, args.dapt_epochs if name == 'dapt_only' else args.sft_epochs,
                 args.dapt_lr if name == 'dapt_only' else args.sft_lr,
                 args.dapt_length if name == 'dapt_only' else args.sft_length,
                 name == 'dapt_only')
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tok,
                         **({'peft_config': peft_config} if peft_config else {}))
    print(f'\n=== TRAINING {name} ===', flush=True)
    start = time.time()
    result = trainer.train()
    eval_metrics = trainer.evaluate()
    adapter = path / 'adapter_final'
    trainer.model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    summary = {
        'phase': name, 'base_model': BASE,
        'training_records': len(train_ds), 'validation_records': len(val_ds),
        'train_metrics': result.metrics, 'eval_metrics': eval_metrics,
        'wall_seconds': round(time.time() - start, 2),
        'adapter': str(adapter),
        'hyperparameters': {'epochs': cfg.num_train_epochs,
            'learning_rate': cfg.learning_rate, 'max_length': cfg.max_length,
            'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
            'lora_rank': 32, 'lora_alpha': 64, 'nf4': True},
        'note': 'Validation loss is not optical extraction F1.'
    }
    (path / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    del trainer
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dapt-epochs', type=float, default=1.0)
    parser.add_argument('--sft-epochs', type=float, default=2.0)
    parser.add_argument('--dapt-lr', type=float, default=1e-5)
    parser.add_argument('--sft-lr', type=float, default=2e-5)
    parser.add_argument('--dapt-length', type=int, default=1024)
    parser.add_argument('--sft-length', type=int, default=8448)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA GPU with BF16 required')
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    raw_train = records(DATA / 'dapt_train.jsonl')
    raw_val = records(DATA / 'dapt_validation.jsonl')
    sft_train = records(SFT_DATA / 'train.jsonl')
    sft_val = records(SFT_DATA / 'validation.jsonl')
    ids_dapt = {x['document_id'] for x in raw_train}
    ids_dapt_val = {x['document_id'] for x in raw_val}
    ids_sft_val = {x['document_id'] for x in sft_val}
    ids_sft_train = {x['document_id'] for x in sft_train}
    assert not (ids_dapt & ids_dapt_val), 'DAPT train/validation overlap'
    assert not (ids_dapt & ids_sft_val), 'DAPT training overlaps SFT validation'
    assert not (ids_sft_train & ids_sft_val), 'SFT train/validation overlap'
    # DAPT validation is an LM validation partition; SFT training may share documents.
    # Do not interpret DAPT validation loss as independent extraction evaluation.
    dtrain = Dataset.from_list([{'text': x['text']} for x in raw_train])
    dval = Dataset.from_list([{'text': x['text']} for x in raw_val])
    def convert(x):
        msgs = x['messages']
        if not msgs or msgs[-1]['role'] != 'assistant':
            raise ValueError('SFT record must end with assistant response')
        return {'prompt': msgs[:-1], 'completion': [msgs[-1]]}
    strain = Dataset.from_list([convert(x) for x in sft_train])
    sval = Dataset.from_list([convert(x) for x in sft_val])
    manifest = {'base_model': BASE, 'dapt_train_pages': len(dtrain),
                'dapt_validation_pages': len(dval), 'sft_train_examples': len(strain),
                'sft_validation_examples': len(sval),
                'historical_sft_adapter': str(ROOT / 'training_outputs/qwen3-4b-otse-expanded-745/adapter_final'),
                'evaluation_status': 'Extraction micro-F1 must be evaluated separately on matched gold references.'}
    (OUT / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)
    model, tok = model_and_tokenizer()
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      bias='none', task_type='CAUSAL_LM', target_modules=LORA_MODULES)
    dapt_result = run_phase('dapt_only', model, tok, dtrain, dval, args, lora)
    del model
    torch.cuda.empty_cache()
    # Reload from the same base; continue training the DAPT adapter on extraction supervision.
    model, tok = model_and_tokenizer()
    model = PeftModel.from_pretrained(model, dapt_result['adapter'], is_trainable=True)
    model.config.use_cache = False
    combined_result = run_phase('dapt_then_sft', model, tok, strain, sval, args)
    summary = {'dapt_only': dapt_result, 'dapt_then_sft': combined_result,
               'historical_sft_only': manifest['historical_sft_adapter'],
               'warning': 'Training/validation metrics are not a matched extraction benchmark.'}
    (OUT / 'comparison_training_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print('\nTRAINING COMPLETE. Summary:', OUT / 'comparison_training_summary.json', flush=True)

if __name__ == '__main__':
    main()
