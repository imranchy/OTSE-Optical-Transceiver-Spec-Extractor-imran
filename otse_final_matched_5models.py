"""OTSE matched silver-dev inference using historical teacher settings.

From the original repository root, evaluate ONLY the unadapted 4B and combine
its score with the prior four-model run (no historical inferences repeated):
    python otse_final_matched_5models.py --base-only --merge-from results/final_matched_20261014_180020

Optional first-task smoke test (does not merge or calculate F1):
    python otse_final_matched_5models.py --base-only --limit 1

Full fresh five-model run (only if explicitly desired):
    python otse_final_matched_5models.py

Resume an incomplete run with --run-dir <directory> and the same flags.
Never changes dataset, historical predictions, or evaluation code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from otse_auto.core import digest_obj, properties, read_jsonl, write_json
from otse_auto.evaluation import model_score

EXPECTED_DEV_SHA256 = "5c017f505415ad41786518c02acd9d9cbe57d48654a15294733ee6d4d1eb62f5"
MODELS = (
    ("teacher_14b", "qwen3:14b-otse8k"),
    ("base_4b", "qwen3:4b-instruct-2507-q4_K_M"),
    ("sft_only_4b", "otse-qwen4b-final:latest"),
    ("dapt_only_4b", "otse-qwen4b-dapt:latest"),
    ("dapt_sft_4b", "otse-qwen4b-dapt-sft:latest"),
)
OPTIONS = {"num_ctx": 8192, "num_predict": 1024, "temperature": 0,
           "seed": 42, "top_k": 20, "top_p": 0.95}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def request_json(url: str, payload=None, timeout=900):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as result:
            return json.loads(result.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Ollama HTTP {err.code}: {detail}") from err


def parse_strict(content):
    try:
        obj = json.loads(content.strip(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def hardware_snapshot():
    """Optional point-in-time device snapshot; not a utilization time series."""
    snapshot = {}
    try:
        import psutil
        snapshot["system_ram_used_bytes"] = psutil.virtual_memory().used
        snapshot["system_cpu_percent_instant"] = psutil.cpu_percent(interval=0.1)
        snapshot["process_rss_bytes"] = psutil.Process().memory_info().rss
    except (ImportError, OSError):
        pass
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        snapshot["nvidia_smi_gpu_snapshot"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return snapshot


def load_prior_four_run(prior_dir, root, dataset_sha, options, known, ollama_version):
    """Validate historical run compatibility; never changes its files."""
    prior_dir = prior_dir.resolve()
    manifest_file = prior_dir / "manifest.json"
    comparison_file = prior_dir / "final_comparison.json"
    if not manifest_file.is_file() or not comparison_file.is_file():
        raise RuntimeError(f"Expected manifest.json and final_comparison.json in {prior_dir}")
    prior = json.loads(manifest_file.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    required = {
        "dataset_sha256": dataset_sha,
        "evaluation_module_sha256": sha256(root / "otse_auto" / "evaluation.py"),
        "options": options,
        "api": "/api/chat",
        "think": False,
        "raw": False,
        "stream": False,
        "format": "per-field-group constrained schema",
        "selected_tasks": 45,
        "tasks": 45,
        "explicit_retries": 0,
    }
    for key, val in required.items():
        if prior.get(key) != val:
            raise RuntimeError(f"Incompatible previous benchmark: {key}: {prior.get(key)!r} != {val!r}")
    if prior.get("ollama_version") != ollama_version:
        raise RuntimeError("Ollama version changed since the four-model run; avoid mixing results.")
    prior_models = prior.get("models", {})
    old_labels = [label for label, _ in MODELS if label != "base_4b"]
    for label, model in MODELS:
        if label == "base_4b":
            continue
        record = prior_models.get(label, {})
        if record.get("name") != model or record.get("digest") != known[model].get("digest"):
            raise RuntimeError(f"The previous model identity/digest differs for {label}; cannot merge safely.")
        if not (prior_dir / f"score_{label}.json").is_file():
            raise RuntimeError(f"Missing prior score_{label}.json")
    summaries = comparison.get("summary", [])
    index = {r["configuration"]: r for r in summaries}
    if set(index) != set(old_labels) or len(summaries) != len(old_labels):
        raise RuntimeError("Previous comparison must contain exactly the four existing model summaries.")
    for label in old_labels:
        if index[label].get("model") != dict(MODELS)[label]:
            raise RuntimeError(f"Prior summary model mismatch for {label}")
    return prior_dir, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Only first N tasks, for optional smoke test")
    parser.add_argument("--run-dir", type=Path, help="Reuse a previous final-run directory (resume/checkpoint)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--base-only", action="store_true", help="Run only the 4B unadapted control")
    parser.add_argument("--merge-from", type=Path, help="Four-model results directory to combine with a base-only result")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or greater")
    if args.merge_from and not args.base_only:
        parser.error("--merge-from requires --base-only")
    if args.merge_from and args.limit:
        parser.error("A smoke test cannot be merged: omit --limit for the 45-task benchmark")
    chosen_models = (("base_4b", dict(MODELS)["base_4b"]),) if args.base_only else MODELS
    root = Path.cwd()
    dataset = root / "training_outputs" / "otse-auto-source-all-v1" / "dataset" / "dev.jsonl"
    if not dataset.is_file():
        parser.error(f"Run from the original OTSE repository root; missing {dataset}")
    actual_hash = sha256(dataset)
    if actual_hash != EXPECTED_DEV_SHA256:
        raise RuntimeError(f"Dataset hash MISMATCH: expected {EXPECTED_DEV_SHA256}, got {actual_hash}. STOP.")
    rows = read_jsonl(dataset)
    if len(rows) != 45 or len({(r["document_id"], r["field_group"]) for r in rows}) != 45:
        raise RuntimeError("Frozen benchmark must have exactly 45 unique extraction tasks")
    selected = rows[:args.limit] if args.limit else rows
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / "results" / f"final_matched_{stamp}"
        if run_dir.exists():
            raise FileExistsError(f"Destination exists: {run_dir}. Supply --run-dir to resume explicitly.")
    base = args.ollama_url.rstrip("/")
    tags = request_json(base + "/api/tags", timeout=20)
    known = {m["name"]: m for m in tags.get("models", [])}
    version = request_json(base + "/api/version", timeout=20).get("version", "unknown")
    for label, name in MODELS:
        if name not in known and (not args.base_only or label == "base_4b" or args.merge_from):
            # With --merge-from we verify the four historical model hashes too.
            raise RuntimeError(f"Missing Ollama model: {name}; available: {list(known)}")
    for label, name in chosen_models:
        if name not in known:
            raise RuntimeError(f"Missing Ollama model: {name}; available: {list(known)}")
    prior_dir = None
    prior_index = {}
    if args.merge_from:
        prior_dir, prior_index = load_prior_four_run(
            args.merge_from, root, actual_hash, OPTIONS, known, version
        )
    metadata = {
        "scope": "AUTOMATED_REFERENCE_DEV_PROXY_ONLY_NOT_FULL60",
        "human_verified": False, "independent_test_claim": False,
        "dataset_sha256": actual_hash, "dataset_path": str(dataset), "tasks": len(rows),
        "selected_tasks": len(selected), "documents": len({r["document_id"] for r in rows}),
        "evaluation_module_sha256": sha256(root / "otse_auto" / "evaluation.py"),
        "ollama_version": version, "api": "/api/chat", "stream": False,
        "think": False, "raw": False, "format": "per-field-group constrained schema",
        "options": OPTIONS, "explicit_retries": 0,
        "models": {label: {"name": name, "digest": known[name].get("digest"),
                           "size_bytes": known[name].get("size"),
                           "details": known[name].get("details")} for label, name in (MODELS if args.merge_from else chosen_models)},
        "merge_source": str(prior_dir) if prior_dir else None,
        "run_mode": "base_only" if args.base_only else "all_five",
        "warning": "Automated silver reference dev benchmark, not expert-gold independent test.",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != metadata:
            raise RuntimeError("Run directory manifest differs (model/runtime/data/evaluator/settings). Use a new run directory.")
    else:
        write_json(manifest_path, metadata)
    print(f"\nOTSE matched benchmark: {len(selected)} tasks x {len(chosen_models)} models", flush=True)
    print(f"Frozen dev dataset SHA256 verified: {actual_hash}", flush=True)
    print(f"Ollama {version}; context=8192; max generation=1024; thinking disabled", flush=True)
    print(f"Results: {run_dir}\n", flush=True)

    summary = []
    for model_idx, (label, model_name) in enumerate(chosen_models, 1):
        folder = run_dir / "predictions" / label
        folder.mkdir(parents=True, exist_ok=True)
        settings = {"runtime": "Ollama_Q4_K_M", "model": model_name,
                    "model_digest": known[model_name].get("digest", ""),
                    "ollama_version": version, "temperature": 0, "seed": 42,
                    "max_tokens": 1024, "context": 8192, "constrained_json": True,
                    "think": False, "top_k": 20, "top_p": 0.95,
                    "evidence": "exact_dev_messages"}
        print(f"\n[{model_idx}/{len(chosen_models)}] {label} ({model_name})", flush=True)
        print("Hardware before model:", hardware_snapshot(), flush=True)
        for i, row in enumerate(selected, 1):
            filename = f"{row['document_id']}__{row['field_group']}.json"
            path = folder / filename
            key = digest_obj([settings, row["source_view_sha256"]])
            if path.exists():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("prediction_key") != key or old.get("split") != "dev":
                    raise RuntimeError(f"Stale/conflicting checkpoint: {path}")
                print(f"  {i:02}/{len(selected)} cached {filename}", flush=True)
                continue
            defs = properties(row["field_group"])
            schema = {"type": "object", "properties": defs,
                      "required": list(defs), "additionalProperties": False}
            payload = {"model": model_name, "messages": row["messages"][:-1],
                       "stream": False, "think": False, "format": schema,
                       "keep_alive": "15m", "options": dict(OPTIONS)}
            print(f"  {i:02}/{len(selected)} {filename} ...", end=" ", flush=True)
            start = time.perf_counter()
            try:
                resp = request_json(base + "/api/chat", payload, timeout=900)
            except Exception as ex:
                print(f"ERROR: {ex}", flush=True)
                raise
            elapsed = time.perf_counter() - start
            message = resp.get("message") or {}
            content = message.get("content") or ""
            parsed = parse_strict(content)
            keys_ok = isinstance(parsed, dict) and set(parsed) == set(defs)
            # Full JSON-Schema checking is diagnostic; canonical model_score uses schema_keys_ok.
            full_schema_ok = False
            validation_error = None
            try:
                import jsonschema
                jsonschema.validate(instance=parsed, schema=schema)
                full_schema_ok = True
            except Exception as ex:
                validation_error = str(ex)[:500]
            usage = {"prompt_tokens": resp.get("prompt_eval_count"),
                     "completion_tokens": resp.get("eval_count"),
                     "prompt_eval_duration_ns": resp.get("prompt_eval_duration"),
                     "eval_duration_ns": resp.get("eval_duration")}
            eval_ns = usage["eval_duration_ns"]
            throughput = (usage["completion_tokens"] * 1e9 / eval_ns
                          if isinstance(eval_ns, (int, float)) and eval_ns > 0
                          and isinstance(usage["completion_tokens"], (int, float)) else None)
            result = {"document_id": row["document_id"], "field_group": row["field_group"],
                      "split": "dev", "prediction_key": key, "settings": settings,
                      "source_view_sha256": row["source_view_sha256"], "content": content,
                      "parsed": parsed, "schema_keys_ok": keys_ok,
                      "full_json_schema_ok": full_schema_ok, "schema_validation_error": validation_error,
                      "thinking_in_content": "<think>" in content,
                      "separate_thinking_present": bool(message.get("thinking")),
                      "separate_thinking": message.get("thinking"),
                      "done_reason": resp.get("done_reason"),
                      "possible_length_cutoff": resp.get("done_reason") == "length",
                      "request_wall_seconds": elapsed, "usage": usage,
                      "generation_tokens_per_second": throughput}
            write_json(path, result)
            print(f"{elapsed:.2f}s keys={keys_ok} schema={full_schema_ok} "
                  f"think={'<think>' in content} done={resp.get('done_reason')} "
                  f"tokens={usage['completion_tokens']}", flush=True)
        print("Hardware after model:", hardware_snapshot(), flush=True)
        # Scoring only after a complete 45-task set for this model.
        if len(selected) == len(rows):
            score, by_field, by_doc, _, detail = model_score(rows, folder)
            items = [json.loads((folder / f"{r['document_id']}__{r['field_group']}.json").read_text(encoding="utf-8")) for r in rows]
            score["full_json_schema_failures"] = sum(not p["full_json_schema_ok"] for p in items)
            score["reasoning_contaminated_groups"] = sum(p["thinking_in_content"] for p in items)
            score["possible_length_cutoff_groups"] = sum(p["possible_length_cutoff"] for p in items)
            token_total = sum(p["usage"]["completion_tokens"] or 0 for p in items)
            eval_seconds = sum((p["usage"]["eval_duration_ns"] or 0) / 1e9 for p in items)
            score["generated_tokens_total"] = token_total
            score["generation_throughput_tokens_per_second"] = token_total / eval_seconds if eval_seconds else None
            score["mean_wall_seconds_per_task"] = score["inference_request_seconds_total"] / len(rows)
            write_json(run_dir / f"score_{label}.json", {"model": model_name, "score": score,
                                                        "by_field": by_field, "per_task_field_details": detail})
            summary.append({"configuration": label, "model": model_name, **score})
            print(f"SCORE {label}: precision={score.get('precision')} recall={score.get('recall')} "
                  f"micro_F1={score.get('micro_f1')} schema_key_failures={score['schema_key_failure_groups']}", flush=True)
        else:
            print("Partial smoke test: no canonical F1 calculated.", flush=True)
    if prior_dir and len(selected) == len(rows):
        # Incorporate earlier score reports unchanged. Only the base model is inferred.
        for label in prior_index:
            shutil.copy2(prior_dir / f"score_{label}.json", run_dir / f"score_{label}.json")
        summary = [
            next(r for r in summary if r["configuration"] == label)
            if label == "base_4b" else prior_index[label]
            for label, _ in MODELS
        ]
        print("Merged four verified historical scores; old predictions remain in:", prior_dir, flush=True)
    if summary:
        write_json(run_dir / "final_comparison.json", {"manifest": metadata, "summary": summary})
        with (run_dir / "final_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["configuration", "model", "precision", "recall", "micro_f1", "macro_field_f1",
                      "schema_key_failure_groups", "full_json_schema_failures",
                      "reasoning_contaminated_groups", "possible_length_cutoff_groups",
                      "inference_request_seconds_total", "mean_wall_seconds_per_task",
                      "generated_tokens_total", "generation_throughput_tokens_per_second"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        print("\nFINAL COMPARISON:", flush=True)
        for r in summary:
            print(f"  {r['configuration']:16s} F1={r['micro_f1']} P={r['precision']} R={r['recall']} "
                  f"schema_fail={r['schema_key_failure_groups']} mean_s={r['mean_wall_seconds_per_task']:.2f}", flush=True)
    print(f"\nComplete. Output: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
