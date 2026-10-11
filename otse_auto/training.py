"""Non-destructive NF4 QLoRA continuation with field-level masked supervision."""
from __future__ import annotations
import inspect
from pathlib import Path
from .core import BASE_ADAPTER, BASE_MODEL, LABEL_PROVENANCE, read_json, read_jsonl, write_json, verify_baseline
from .dataset import token_record


def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('Fast tokenizer required to prove label-mask boundaries')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def load_adapter(root, adapter, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this 4B NF4 workflow')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=config,
        torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
    if trainable:
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    model.config.use_cache = not trainable
    return model, dtype


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        import torch
        # Batch size one by default, but general padding correctly masks extra tokens.
        length = max(len(e['input_ids']) for e in examples)
        result = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for e in examples:
            count = length - len(e['input_ids'])
            result['input_ids'].append(e['input_ids'] + [self.tokenizer.pad_token_id] * count)
            result['attention_mask'].append(e['attention_mask'] + [0] * count)
            result['labels'].append(e['labels'] + [-100] * count)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def train(root, out, config):
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from torch.utils.data import Dataset
    report = read_json(out / 'dataset/dataset_report.json')
    if not report.get('ready_for_training'):
        raise RuntimeError('Automated coverage is insufficient; use the RUN command, which expands source labeling automatically. No manual approval is required.')
    done = out / 'training_complete.json'
    if done.exists():
        saved = read_json(done)
        if saved['dataset_signature'] != report['dataset_signature']:
            raise RuntimeError('Dataset changed after training; use a new --output directory')
        print('Training already complete; preserving existing candidate.', flush=True)
        return
    meta = out / 'training_started.json'
    start_record = {'dataset_signature': report['dataset_signature'], 'epochs': config['epochs'], 'lr': config['lr'],
                    'max_length': config['max_length'], 'gradient_accumulation': config['grad_accum']}
    if meta.exists() and read_json(meta) != start_record:
        raise RuntimeError('Cannot resume with different data or training settings; use a new --output directory')
    write_json(meta, start_record)
    set_seed(42)
    tokenizer = get_tokenizer()
    train_rows = read_jsonl(out / 'dataset/train.jsonl')
    dev_rows = read_jsonl(out / 'dataset/dev.jsonl')
    for row in train_rows + dev_rows:
        if row.get('label_provenance') != LABEL_PROVENANCE:
            raise ValueError('Wrong label provenance')
    class EncodedDataset(Dataset):
        def __init__(self, rows):
            self.data = [token_record(row, tokenizer, config['max_length']) for row in rows]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index]
    train_data, dev_data = EncodedDataset(train_rows), EncodedDataset(dev_rows)
    model, dtype = load_adapter(root, root / BASE_ADAPTER, trainable=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError('No trainable adapter parameters')
    dest = out / 'training'
    params = dict(output_dir=str(dest), num_train_epochs=config['epochs'], learning_rate=config['lr'],
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=config['grad_accum'],
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, optim='paged_adamw_8bit',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=5, save_strategy='epoch', save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, prediction_loss_only=True,
        report_to='none', seed=42, data_seed=42, remove_unused_columns=False,
        lr_scheduler_type='cosine', warmup_ratio=0.04, weight_decay=0.0, max_grad_norm=0.3,
        dataloader_num_workers=0, dataloader_pin_memory=False)
    signature = inspect.signature(TrainingArguments).parameters
    strategy_key = 'eval_strategy' if 'eval_strategy' in signature else 'evaluation_strategy'
    if strategy_key not in signature:
        raise RuntimeError('Installed TrainingArguments has no supported evaluation strategy parameter')
    params[strategy_key] = 'epoch'
    unsupported = sorted(set(params) - set(signature))
    if unsupported:
        print('TrainingArguments compatibility: removing unsupported keys:', unsupported, flush=True)
        params = {k: v for k, v in params.items() if k in signature}
    args = TrainingArguments(**params)
    trainer_args = {'model': model, 'args': args, 'train_dataset': train_data, 'eval_dataset': dev_data,
                    'data_collator': Collator(tokenizer)}
    ts = inspect.signature(Trainer).parameters
    trainer_args['processing_class' if 'processing_class' in ts else 'tokenizer'] = tokenizer
    trainer = Trainer(**trainer_args)
    checkpoint = get_last_checkpoint(str(dest)) if dest.is_dir() else None
    print(f'QLoRA: {trainable:,} trainable parameters; {len(train_data)} train examples; checkpoint={checkpoint}', flush=True)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    target = out / 'adapter_candidate'
    if target.exists():
        raise FileExistsError('Candidate directory exists without a completion record; preserving it. Use a new --output directory.')
    trainer.model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    write_json(done, {**start_record, 'trainable_parameters': trainable, 'train_metrics': result.metrics,
                      'eval_metrics': trainer.evaluate(), 'candidate_adapter': str(target),
                      'selected_by': 'masked development loss; downstream factual comparison is still required',
                      'human_verified_reference': False})
    verify_baseline(root, read_json(out / 'baseline_fingerprint.json'))
    print('Candidate saved; original adapter and GGUF hashes unchanged.', flush=True)
