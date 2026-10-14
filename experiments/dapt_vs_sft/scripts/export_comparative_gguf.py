#!/usr/bin/env python3
"""Export OTSE DAPT and DAPT->SFT PEFT adapters as separate Q4_K_M GGUF models.

Run from the OTSE repository root on Windows:
  python experiments\dapt_vs_sft\scripts\export_comparative_gguf.py

The historical deployment/otse-qwen4b-Q4_K_M.gguf is NEVER modified.
Requires the existing llama.cpp converter and llama-quantize binary.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = "Qwen/Qwen3-4B-Instruct-2507"
NAMES = {
    "dapt": ("dapt_only", "otse-dapt-Q4_K_M.gguf", "otse-qwen4b-dapt"),
    "dapt_sft": ("dapt_then_sft", "otse-dapt-sft-Q4_K_M.gguf", "otse-qwen4b-dapt-sft"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None):
    print("RUN:", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def locate_quantizer(llama: Path) -> Path:
    candidates = [
        llama / "build/bin/Release/llama-quantize.exe",
        llama / "build/bin/llama-quantize.exe",
        llama / "build/bin/llama-quantize",
        llama / "llama-quantize.exe",
        llama / "llama-quantize",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    binary = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    if binary:
        return Path(binary)
    raise FileNotFoundError("llama-quantize binary not found. Build llama.cpp or supply --quantizer PATH.")


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def valid_gguf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as f:
        return f.read(4) == b"GGUF"


def merge_adapter(adapter: Path, merged: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base model on CPU (FP16)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )
    print("Loading adapter:", adapter, flush=True)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    print("Merging adapter...", flush=True)
    model = model.merge_and_unload(safe_merge=True)
    merged.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(merged), safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(str(merged))
    del model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="OTSE repository root")
    ap.add_argument("--llama-cpp-dir", type=Path, default=Path("deployment/llama.cpp"))
    ap.add_argument("--quantizer", type=Path)
    ap.add_argument("--convert-python", type=Path, help="Python with gguf converter dependencies; defaults to deployment/.venv-convert")
    ap.add_argument("--models", nargs="+", choices=list(NAMES), default=list(NAMES))
    ap.add_argument("--register-ollama", action="store_true", help="Run ollama create after exporting")
    ap.add_argument("--keep-intermediates", action="store_true", help="Keep merged HF and F16 GGUF files")
    args = ap.parse_args()
    root = args.root.resolve()
    llama = args.llama_cpp_dir if args.llama_cpp_dir.is_absolute() else root / args.llama_cpp_dir
    convert = llama / "convert_hf_to_gguf.py"
    quant = args.quantizer if args.quantizer else locate_quantizer(llama)
    if not quant.is_absolute():
        quant = (root / quant).resolve()
    if not convert.is_file():
        raise FileNotFoundError(f"GGUF converter missing: {convert}")
    if not quant.is_file():
        raise FileNotFoundError(f"Quantizer missing: {quant}")
    convert_py = args.convert_python
    if convert_py is None:
        candidate = root / "deployment/.venv-convert/Scripts/python.exe"
        convert_py = candidate if candidate.is_file() else Path(sys.executable)
    elif not convert_py.is_absolute():
        convert_py = root / convert_py
    if not convert_py.is_file():
        raise FileNotFoundError(f"Converter Python missing: {convert_py}")

    historical = root / "deployment/otse-qwen4b-Q4_K_M.gguf"
    if not valid_gguf(historical):
        raise FileNotFoundError("Historical GGUF missing or invalid; refusing export until baseline is identified.")
    baseline_hash = sha256(historical)
    print("Historical baseline SHA256:", baseline_hash, flush=True)
    deployment = root / "deployment"
    work = root / "experiments/dapt_vs_sft/outputs/gguf_export_work"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiments/dapt_vs_sft/results/gguf_export_manifest.json"
    manifest = {
        "base_model": BASE,
        "baseline_gguf": str(historical),
        "baseline_sha256": baseline_hash,
        "quantization": "Q4_K_M",
        "converter": str(convert),
        "quantizer": str(quant),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    for key in args.models:
        source_folder, out_name, ollama_name = NAMES[key]
        adapter = root / "experiments/dapt_vs_sft/outputs" / source_folder / "adapter_final"
        cfg_path = adapter / "adapter_config.json"
        weights_path = adapter / "adapter_model.safetensors"
        if not cfg_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Adapter config/weights missing: {adapter}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("base_model_name_or_path") != BASE:
            raise ValueError(f"Adapter {key} has incompatible base: {cfg.get('base_model_name_or_path')}")
        out = deployment / out_name
        if out == historical:
            raise RuntimeError("Refusing to overwrite historical baseline")
        phase = work / key
        merged = phase / "merged_fp16"
        f16 = phase / "model-f16.gguf"
        complete = phase / "merged_complete.json"
        phase.mkdir(parents=True, exist_ok=True)
        info = {
            "adapter": str(adapter),
            "adapter_config_sha256": sha256(cfg_path),
            "adapter_weights_sha256": sha256(weights_path),
            "output": str(out),
            "ollama_name": ollama_name,
        }
        print(f"\n=== EXPORT {key} ===", flush=True)
        start = time.monotonic()
        if valid_gguf(out):
            print("Existing valid GGUF detected; skipping export:", out, flush=True)
        else:
            if out.exists():
                raise FileExistsError(f"Existing invalid output will not be overwritten: {out}")
            if not complete.is_file():
                if merged.exists():
                    raise FileExistsError(f"Incomplete merged directory exists: {merged}; inspect before retrying")
                merge_adapter(adapter, merged)
                atomic_json(complete, {"adapter_sha256": info["adapter_weights_sha256"], "base": BASE})
            else:
                done = json.loads(complete.read_text(encoding="utf-8"))
                if done.get("adapter_sha256") != info["adapter_weights_sha256"]:
                    raise RuntimeError(f"Stale merged model for {key}; refusing reuse")
            if not valid_gguf(f16):
                partial = phase / "model-f16.gguf.partial"
                if partial.exists():
                    partial.unlink()
                run([str(convert_py), str(convert), str(merged), "--outfile", str(partial), "--outtype", "f16"])
                if not valid_gguf(partial):
                    raise RuntimeError(f"Invalid intermediate GGUF: {partial}")
                os.replace(partial, f16)
            partial_q = deployment / (out_name + ".partial")
            if partial_q.exists():
                partial_q.unlink()
            run([str(quant), str(f16), str(partial_q), "Q4_K_M"])
            if not valid_gguf(partial_q):
                raise RuntimeError(f"Invalid quantized GGUF: {partial_q}")
            os.replace(partial_q, out)
        info["gguf_sha256"] = sha256(out)
        info["gguf_bytes"] = out.stat().st_size
        info["export_seconds_this_run"] = round(time.monotonic() - start, 1)
        modelfile = root / f"Modelfile.{ollama_name}"
        # Exact raw-template setting seen in the historical `ollama show` output.
        modelfile.write_text(
            f'FROM "{out.as_posix()}"\nTEMPLATE {{{{ .Prompt }}}}\nPARAMETER num_ctx 8192\nPARAMETER temperature 0\n',
            encoding="utf-8",
        )
        info["modelfile"] = str(modelfile)
        if args.register_ollama:
            run(["ollama", "create", ollama_name, "-f", str(modelfile)], cwd=root)
        manifest["models"][key] = info
        atomic_json(manifest_path, manifest)
        print(f"EXPORTED {key}: {out} ({info['gguf_bytes']:,} bytes)", flush=True)
        if not args.keep_intermediates:
            if merged.is_dir():
                shutil.rmtree(merged)
            if f16.is_file():
                f16.unlink()
            if complete.is_file():
                complete.unlink()
    if sha256(historical) != baseline_hash:
        raise RuntimeError("Historical GGUF hash changed during export!")
    print("\nDONE. Export manifest:", manifest_path)
    print("Historical GGUF unchanged:", baseline_hash)


if __name__ == "__main__":
    main()
