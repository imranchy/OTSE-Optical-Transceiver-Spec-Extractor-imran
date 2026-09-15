"""Streamlit UI for the transceiver datasheet extraction pipeline.

Uses Ollama's local OpenAI-compatible API.
"""

import json
import tempfile
import time
from pathlib import Path

import requests
import streamlit as st
from pydantic import ValidationError

from pipeline.pdf_parser import parse_pdf
from pipeline.atom_extractor import extract_atoms, BASE_URL
import re
from pipeline.mode_linker import synthesize_modes, store_modes, separate_atoms, _filter_sections_by_provenance
from pipeline.runtime_store import (
    list_runs,
    load_run,
    store_specs,
    store_run,
    update_run_timings,
)
from generate_report import generate_html
from generate_modes_report import generate_modes_html
from transceiver_models import TransceiverSpecs

REPORTS_DIR = Path("reports")

st.set_page_config(page_title="Transceiver Spec Extractor", layout="wide")
st.title("Transceiver Datasheet Extractor")


# --------------------------------------------------------------------------- #
# Server / model helpers (from Chatbot/llm.py pattern)
# --------------------------------------------------------------------------- #

def _server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/api/version", timeout=5).ok
    except requests.RequestException:
        return False


def _list_models() -> list[str]:
    resp = requests.get(f"{BASE_URL}/v1/models", timeout=15)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


# --------------------------------------------------------------------------- #
# Sidebar: server status + model picker
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("LLM Server")

    if not _server_up():
        st.error(
            "Ollama is not reachable on port 11434.\n\n"
            "Start it with:\n```\nollama serve\n```"
        )
        st.stop()

    st.success("Server online")

    models = _list_models()
    if not models:
        st.error("No Ollama models found. Install the target model with:\n```\nollama pull qwen3:14b\n```")
        st.stop()

    default_model = "qwen3:14b"
    default_index = models.index(default_model) if default_model in models else 0
    model_id = st.selectbox("Model", models, index=default_index)

    st.divider()
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.slider("Max tokens", 1024, 8192, 4096, 512)
    st.caption("Schema-constrained extraction runs with Qwen reasoning disabled.")

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #

tab_extract, tab_history = st.tabs(["Extract", "History"])

with tab_extract:
    uploaded = st.file_uploader("Upload a PDF datasheet", type=["pdf"])

    # --- Step 1: Parse PDF ---
    if uploaded:
        file_key = uploaded.name + str(uploaded.size)
        if st.session_state.get("_parse_key") != file_key:
            st.session_state.pop("parsed", None)
            st.session_state.pop("edited_md", None)
            st.session_state.pop("extraction_result", None)
            st.session_state.pop("timings_seconds", None)
            st.session_state.pop("current_specs", None)
            st.session_state.pop("modes_result", None)

        if st.button("Parse PDF", type="secondary"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            with st.spinner("Parsing..."):
                stage_started = time.perf_counter()
                parsed = parse_pdf(tmp_path)
                parse_seconds = time.perf_counter() - stage_started
            st.session_state["parsed"] = parsed
            st.session_state["edited_md"] = parsed.markdown
            st.session_state["timings_seconds"] = {"parse_pdf": parse_seconds}
            st.session_state["_parse_key"] = file_key
            st.session_state.pop("extraction_result", None)
            st.success(f"PDF parsed in {parse_seconds:.3f} seconds")

    # --- Step 2: Review / edit parsed markdown ---
    if "parsed" in st.session_state:
        parsed = st.session_state["parsed"]
        st.subheader("Step 1 — Parsed Content")
        st.caption(f"{parsed.page_count} pages | {len(parsed.tables)} tables | {len(parsed.markdown):,} chars")

        edited_md = st.text_area(
            "Edit the markdown before extraction",
            value=st.session_state.get("edited_md", parsed.markdown),
            height=400,
            key="md_editor",
        )
        st.session_state["edited_md"] = edited_md

        headings = [re.sub(r"^#{1,6}\s+", "", line.strip()) for line in edited_md.splitlines() if re.match(r"^#{1,6}\s+", line)]
        with st.expander(f"Identified Headings ({len(headings)})"):
            if headings:
                for h in headings:
                    st.write(f"- {h}")
            else:
                st.caption("No headings found")

        # Show provenance-filtered content after extraction
        if "extraction_result" in st.session_state:
            ex_specs, ex_raw = st.session_state["extraction_result"]
            ex_data = st.session_state.get("current_specs") or (
                json.loads(ex_specs.model_dump_json()) if ex_specs else ex_raw
            )
            provenance = ex_data.get("provenance_datasheet_sections")
            if provenance and isinstance(provenance, (str, list)):
                filtered_md = _filter_sections_by_provenance(edited_md, provenance)
                prov_headings = len([line for line in filtered_md.splitlines() if re.match(r"^#{1,6}\s+", line)])
                with st.expander(f"Provenance Sections ({len(filtered_md):,} chars · {prov_headings} sections)", expanded=False):
                    st.text_area(
                        "Filtered content (read-only)",
                        value=filtered_md,
                        height=400,
                        disabled=True,
                        key="provenance_viewer",
                        label_visibility="collapsed",
                    )

        # --- Step 3: Extract ---
        if st.button("Extract Parameters", type="primary"):
            # Patch the parsed object with edited markdown
            parsed.markdown = st.session_state["edited_md"]

            with st.status("Extracting...", expanded=True) as status:
                st.write(f"Sending to `{model_id}` ({len(parsed.markdown):,} chars)...")
                stage_started = time.perf_counter()
                specs, raw_dict = extract_atoms(
                    parsed,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                extract_seconds = time.perf_counter() - stage_started
                timings = dict(st.session_state.get("timings_seconds", {}))
                timings["extract_atoms"] = extract_seconds
                st.write(f"Extraction completed in {extract_seconds:.3f} seconds")

                st.write("Storing run...")
                stage_started = time.perf_counter()
                run_dir = store_run(
                    parsed,
                    raw_dict,
                    specs,
                    model_id=model_id,
                    timings_seconds=timings,
                )
                timings["store_run"] = time.perf_counter() - stage_started
                timings["total"] = sum(
                    seconds for name, seconds in timings.items() if name != "total"
                )
                update_run_timings(run_dir, timings)

                # Persist content & provenance stats into meta
                result_data = json.loads(specs.model_dump_json()) if specs else raw_dict
                md_text = st.session_state["edited_md"]
                all_headings = [l for l in md_text.splitlines() if re.match(r"^#{1,6}\s+", l)]
                prov = result_data.get("provenance_datasheet_sections")
                if prov and isinstance(prov, (str, list)):
                    filt = _filter_sections_by_provenance(md_text, prov)
                    prov_headings = [l for l in filt.splitlines() if re.match(r"^#{1,6}\s+", l)]
                else:
                    filt = md_text
                    prov_headings = all_headings
                content_stats = {
                    "content_chars": len(md_text),
                    "content_headings": len(all_headings),
                    "provenance_chars": len(filt),
                    "provenance_headings": len(prov_headings),
                }
                meta_path = run_dir / "meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["content_stats"] = content_stats
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                st.session_state["timings_seconds"] = timings
                st.session_state["_last_run_dir"] = str(run_dir)
                st.write(f"Saved to `{run_dir}` in {timings['store_run']:.3f} seconds")
                st.write(f"Total processing time: {timings['total']:.3f} seconds")

                if specs:
                    status.update(label="Extraction complete", state="complete")
                else:
                    status.update(label="Extraction complete (validation failed)", state="error")

            st.session_state["extraction_result"] = (specs, raw_dict)
            st.session_state.pop("modes_result", None)
            st.session_state.pop("current_specs", None)

    # --- Display results ---
    if "extraction_result" in st.session_state:
        specs, raw_dict = st.session_state["extraction_result"]
        run_dir = Path(st.session_state["_last_run_dir"])

        # Use saved edits if available, otherwise the extraction output
        display_data = st.session_state.get("current_specs") or (
            json.loads(specs.model_dump_json()) if specs else raw_dict
        )

        st.subheader("Extracted Parameters")
        col1, col2, col3 = st.columns(3)
        scalar_fields = {}
        list_fields = {}
        null_fields = []
        for k, v in display_data.items():
            if v is None:
                null_fields.append(k)
            elif isinstance(v, list):
                list_fields[k] = v
            else:
                scalar_fields[k] = v

        with col1:
            st.markdown("**Scalar Values**")
            for k, v in scalar_fields.items():
                st.write(f"- **{k}:** {v}")
        with col2:
            st.markdown("**List Values**")
            for k, v in list_fields.items():
                st.write(f"- **{k}:** {v}")
        with col3:
            st.markdown("**Not Found (null)**")
            for k in null_fields:
                st.write(f"- {k}")

        with st.expander("Edit Parameters (JSON)"):
            editor_key = f"review_editor_{run_dir.name}"
            edited_specs_json = st.text_area(
                "Parameters (JSON)",
                value=json.dumps(display_data, indent=2),
                height=600,
                key=editor_key,
                label_visibility="collapsed",
            )
            if st.button("Save Changes"):
                try:
                    edited_data = json.loads(edited_specs_json)
                    if not isinstance(edited_data, dict):
                        raise ValueError("Parameters must be a JSON object.")
                    validated = TransceiverSpecs.model_validate(edited_data)
                    saved = store_specs(run_dir, validated)
                    st.session_state["current_specs"] = saved
                    st.success("Changes saved to specs.json.")
                    st.rerun()
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    st.error(f"Could not save: {exc}")

        # --- Step 4: Mode Synthesis ---
        st.divider()
        st.subheader("Step 2 — Application-Mode Extraction")
        if st.button("Synthesize Modes", type="primary"):
            source = display_data
            multi, general = separate_atoms(source)
            st.session_state.pop("modes_result", None)

            if multi:
                st.caption("Mode rows are extracted directly from the source tables and validated against a separate schema.")
                with st.expander("Multi-valued atoms"):
                    st.json(multi)

            with st.status("Extracting application-mode rows...", expanded=True) as status:
                stage_started = time.perf_counter()
                modes_result, _, _ = synthesize_modes(
                    source,
                    st.session_state["parsed"],
                    run_dir=run_dir,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                mode_seconds = time.perf_counter() - stage_started
                timings = dict(st.session_state.get("timings_seconds", {}))
                timings["synthesize_modes"] = mode_seconds
                st.write(f"Mode synthesis completed in {mode_seconds:.3f} seconds")
                if modes_result:
                    stage_started = time.perf_counter()
                    store_modes(run_dir, modes_result, multi, general)
                    timings["store_modes"] = time.perf_counter() - stage_started
                    st.write(f"Modes saved in {timings['store_modes']:.3f} seconds")
                    status.update(label=f"{len(modes_result.get('modes', []))} modes identified", state="complete")
                else:
                    status.update(label="Mode synthesis failed", state="error")

                timings["total"] = sum(
                    seconds for name, seconds in timings.items() if name != "total"
                )
                update_run_timings(run_dir, timings)
                st.session_state["timings_seconds"] = timings

            st.session_state["modes_result"] = modes_result

        # --- Generate Report button (after mode synthesis) ---
        if st.session_state.get("modes_result"):
            run_dir = Path(st.session_state["_last_run_dir"])
            if st.button("Generate Report", type="secondary"):
                REPORTS_DIR.mkdir(exist_ok=True)
                pipeline_path = REPORTS_DIR / f"{run_dir.name}_pipeline.html"
                html = generate_html(run_dir)
                pipeline_path.write_text(html, encoding="utf-8")
                modes_path = REPORTS_DIR / f"{run_dir.name}_modes.html"
                modes_html = generate_modes_html(run_dir)
                modes_path.write_text(modes_html, encoding="utf-8")
                st.success(f"Reports saved: `{pipeline_path}` and `{modes_path}`")

            modes_result = st.session_state["modes_result"]
            with st.expander("Raw LLM Mode Output"):
                st.json(modes_result)
            st.subheader("Configuration Modes")
            for i, mode in enumerate(modes_result.get("modes", [])):
                label = mode.get("label", f"Mode {i+1}")
                with st.expander(f"**{label}**", expanded=i == 0):
                    st.json(mode)

with tab_history:
    runs = list_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        selected = st.selectbox("Select a run", [r.name for r in reversed(runs)])
        if selected:
            run_data = load_run(Path("runs") / selected)
            if "meta" in run_data:
                st.json(run_data["meta"])
            if "specs" in run_data:
                st.subheader("Validated Specs")
                st.json(run_data["specs"])
            elif "raw_llm" in run_data:
                st.subheader("Raw LLM Output (no validated specs)")
                st.json(run_data["raw_llm"])
            if "raw_llm_modes_response" in run_data:
                with st.expander("Raw Mode LLM Response"):
                    st.json(run_data["raw_llm_modes_response"])
            if "raw_llm_modes" in run_data:
                with st.expander("Raw LLM Mode Output"):
                    st.json(run_data["raw_llm_modes"])

            if st.button("Generate Report", key="hist_report"):
                REPORTS_DIR.mkdir(exist_ok=True)
                sel_run_dir = Path("runs") / selected
                pipeline_path = REPORTS_DIR / f"{selected}_pipeline.html"
                html = generate_html(sel_run_dir)
                pipeline_path.write_text(html, encoding="utf-8")
                modes_path = REPORTS_DIR / f"{selected}_modes.html"
                modes_html = generate_modes_html(sel_run_dir)
                modes_path.write_text(modes_html, encoding="utf-8")
                st.success(f"Reports saved: `{pipeline_path}` and `{modes_path}`")
