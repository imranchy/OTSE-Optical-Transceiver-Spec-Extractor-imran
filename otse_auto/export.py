"""Optional safe CPU merge and separate GGUF export; never promotes over baseline."""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .core import BASE_MODEL, read_json, write_json


def find_tools(root, explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get('LLAMA_CPP_DIR'):
        candidates.append(Path(os.environ['LLAMA_CPP_DIR']))
    candidates.extend([root / 'llama.cpp', root / 'tools/llama.cpp', root / 'external/llama.cpp',
                       root.parent / 'llama.cpp', Path.home() / 'llama.cpp',
                       Path.home() / 'Downloads/llama.cpp', Path.home() / 'Downloads/Projects/llama.cpp'])
    for folder in candidates:
        script = folder / 'convert_hf_to_gguf.py'
        if not script.is_file():
            continue
        paths = [folder / rel for rel in ('build/bin/Release/llama-quantize.exe', 'build/bin/llama-quantize.exe',
                 'build/bin/llama-quantize', 'llama-quantize.exe', 'llama-quantize')]
        on_path = shutil.which('llama-quantize') or shutil.which('llama-quantize.exe')
        if on_path:
            paths.append(Path(on_path))
        quant = next((p for p in paths if p.is_file()), None)
        if quant:
            return script.resolve(), quant.resolve()
    return None


def export_candidate(root, out, explicit=None, automatic=False):
    tools = find_tools(root, explicit)
    status_path = out / 'reports/gguf_export.json'
    if not tools:
        note = {'exported': False, 'reason': 'Existing llama.cpp converter/quantizer not found automatically.',
                'next_command': 'python -m otse_auto export --llama-cpp-dir "C:\\path\\to\\llama.cpp"',
                'candidate_adapter_preserved': True}
        write_json(status_path, note)
        if not automatic:
            raise FileNotFoundError(note['reason'] + ' Supply --llama-cpp-dir. Training and labels are preserved.')
        print('Adapter complete. GGUF export deferred: llama.cpp tools not located.', flush=True)
        return note
    if not (out / 'adapter_candidate/adapter_config.json').is_file():
        raise FileNotFoundError('Train the candidate adapter before export')
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
    except ImportError:
        available_ram = None
    if shutil.disk_usage(out).free < 24 * 1024**3 or (available_ram is not None and available_ram < 16 * 1024**3):
        note = {'exported': False, 'reason': 'CPU export requires at least 24 GiB free disk and a conservative 16 GiB available RAM preflight.',
                'available_ram_bytes': available_ram, 'candidate_adapter_preserved': True}
        write_json(status_path, note)
        if not automatic:
            raise RuntimeError(note['reason'])
        print('GGUF export deferred: RAM/disk preflight. Adapter remains usable.', flush=True)
        return note
    merged = out / 'merged_fp16'
    merge_done = out / 'merge_complete.json'
    if not merge_done.is_file():
        if merged.exists():
            raise FileExistsError('Incomplete/existing merged directory retained. Use a fresh output location rather than overwriting it.')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16,
                                                   device_map='cpu', low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, str(out / 'adapter_candidate'), is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
        model.save_pretrained(merged, safe_serialization=True, max_shard_size='4GB')
        AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(merged)
        write_json(merge_done, {'merged': True, 'source_adapter': str(out / 'adapter_candidate')})
        del model, base
        import gc
        gc.collect()
    script, quant = tools
    intermediate = out / 'candidate-f16.gguf'
    final = out / 'candidate-Q4_K_M.gguf'
    for path, cmd in [(intermediate, [sys.executable, str(script), str(merged), '--outfile', str(intermediate) + '.partial', '--outtype', 'f16']),
                      (final, [str(quant), str(intermediate), str(final) + '.partial', 'Q4_K_M'])]:
        if path.is_file():
            continue
        partial = Path(str(path) + '.partial')
        if partial.exists():
            partial.unlink()  # Only this workflow's clearly marked incomplete output.
        subprocess.run(cmd, check=True)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError('Converter produced no GGUF file')
        with partial.open('rb') as h:
            if h.read(4) != b'GGUF':
                raise RuntimeError('Invalid GGUF magic; incomplete conversion retained')
        os.replace(partial, path)
    status = {'exported': True, 'candidate_gguf': str(final), 'baseline_overwritten': False,
              'warning': 'HF NF4 development scores do not establish Q4_K_M deployment performance. Run matched GGUF inference separately.'}
    write_json(status_path, status)
    print('Separate candidate GGUF saved: ' + str(final), flush=True)
    return status
