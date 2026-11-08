# =============================================================================
# 11 — RE-EXTRACT THE DeepSeek-R1-Distill-Llama-8B ACTIVATIONS
#
# WHY. For the three Qwen models, recomputing the extractor's window centre
# from the traces, labels and tokenizer reproduces every stored token_idx.
# For DeepSeek-R1-Distill-Llama-8B it reproduces 3 of 1,021 revision and 2 of
# 505 filler windows, under every text / cleaning / prompt variant tested
# (analyses/diagnose_llama8b_window_alignment.py). Row mapping is correct and
# the traces and labels are older than the activation files; the cause is
# unknown. So it is not known which token position the stored Llama-8B windows
# represent. This re-runs extraction for Llama-8B with the current code.
#
# WHAT. Notebooks 03 (revision windows), 05 (filler windows) and 06 (Table 1
# probe, Parts A and B), copied VERBATIM from pipeline/. The only changes are
# marked "# CHANGED": the model (Llama-8B) and the output folders / result
# files, which are NEW, so nothing existing is overwritten:
#     activations_reextract_2026_11/DeepSeek-R1-Distill-Llama-8B/
#     activations_filler_reextract_2026_11/DeepSeek-R1-Distill-Llama-8B/
#     results_v3/task1_residual_cis_nboot2000_v3_llama8b_reextract.json
#     results_v3/task1_residual_pointfix_llama8b_reextract.json
#     hand_labeling_export/llama8b_reextract_verification.json
#
# STAGES (set STAGES below; run in this order, each resumes if interrupted):
#   "revision"  GPU. Notebook 03. Skips problems already in its extraction log.
#   "filler"    GPU. Notebook 05. Skips problems whose .npz already exists.
#   "verify"    CPU. Checks the new windows (recomputed centre == stored, for
#               every event), where they sit relative to the marker, and how
#               they compare with the OLD windows (does the old centre vector
#               appear anywhere in the new window?).
#   "probe"     CPU, hours. Notebook 06 Part A then Part B on the new windows.
#   "direction" CPU. Notebook 08 cell 1: the Llama-8B steering direction from
#               the new windows (new file), compared with the published one.
#               Then 09_necessity_rerun.py with LLAMA_DIRECTION = "reextract",
#               STAGE = "steering", then "analyze".
#   Then: analyses/probe_robustness_checks.py with ACTIVATIONS = "llama8b_reextract".
#
# DISK. Notebook 03 also stores control windows and norms; it prints its own
# storage estimate before starting (Llama-8B: 32 layers x 61 tokens x 4096,
# float16, per window, for revision AND control windows). Check Drive space.
# =============================================================================
STAGES = ["revision", "filler"]          # then ["verify"], ["probe"], ["direction"]

import os, gc, json
from pathlib import Path
DATA_ROOT = Path(os.environ.get("REVISION_DATA_ROOT", "/content/gdrive/MyDrive/llm_planning_study"))
if not DATA_ROOT.exists():
    from google.colab import drive; drive.mount("/content/gdrive")
print("DATA_ROOT =", DATA_ROOT, " STAGES =", STAGES)
NEW_REV = "activations_reextract_2026_11"
NEW_FIL = "activations_filler_reextract_2026_11"
LLAMA   = "DeepSeek-R1-Distill-Llama-8B"
if "revision" in STAGES or "filler" in STAGES:
    import torch


# #############################################################################
# STAGE "revision" — notebook 03, verbatim (Llama-8B, new output folder)
# #############################################################################
if "revision" in STAGES:
    # ---- 03, cell 1 (verbatim except where marked) ----
    # =============================================================================
    # 03 — EXTRACT REVISION ACTIVATIONS
    #
    # PURPOSE: Re-run each model with forward hooks registered. For each labeled
    #          revision token, extract ±30 token windows of residual stream
    #          activations, attention patterns, and layer-wise norms.
    #          Save compressed numpy arrays to Drive after every problem.
    #
    # INPUTS:  DATA_ROOT/traces/{model}/{problem_id}.json   (notebook 01)
    #          DATA_ROOT/labels/{model}/{problem_id}.json   (notebook 02)
    #
    # OUTPUTS: DATA_ROOT/activations/{model}/
    #              {problem_id}.npz           (compressed numpy arrays)
    #              extraction_log.json        (progress tracker)
    #          DATA_ROOT/activations_summary_all_models.json
    #
    # SPECIAL: Llama-3.1-8B-Instruct gets different handling —
    #          500 random token windows instead of revision windows.
    #          Set IS_BASELINE_MODEL = True for that run.
    #
    # RUNTIME: GPU required. One session per model.
    #          Estimated: 3-5 hours per model on A100 80GB.
    #
    # RUN ORDER:
    #   1. DeepSeek-R1-Distill-Llama-8B    IS_BASELINE_MODEL = False
    #   2. Llama-3.1-8B-Instruct            IS_BASELINE_MODEL = True
    #   3. DeepSeek-R1-Distill-Qwen-7B     IS_BASELINE_MODEL = False
    #   4. DeepSeek-R1-Distill-Qwen-14B    IS_BASELINE_MODEL = False
    #   5. DeepSeek-R1-Distill-Qwen-1.5B   IS_BASELINE_MODEL = False
    # =============================================================================

    # ---- 03, cell 3 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 2 — Imports and config
    # -----------------------------------------------------------------------------
    import gc
    import json
    import time
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path
    from datetime import datetime
    from collections import defaultdict
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # ── MODEL SELECTION ───────────────────────────────────────────────────────────
    #MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    #MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    #MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    #MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # CHANGED: Llama-8B
    IS_BASELINE_MODEL = False   # True ONLY for Llama-3.1-8B-Instruct

    # ── EXTRACTION CONFIG ─────────────────────────────────────────────────────────
    WINDOW_TOKENS     = 30      # ±30 tokens around each revision
    N_BASELINE_WINDOWS = 500    # random windows for Llama baseline
    RANDOM_SEED        = 42

    # ── QUANTIZATION ──────────────────────────────────────────────────────────────
    USE_4BIT = "32B" in MODEL_NAME or "32b" in MODEL_NAME

    # ── PATHS ─────────────────────────────────────────────────────────────────────
    GDRIVE_BASE    = DATA_ROOT
    MODEL_SHORT    = MODEL_NAME.split("/")[-1]
    TRACE_DIR      = GDRIVE_BASE / "traces"  / MODEL_SHORT
    LABEL_DIR      = GDRIVE_BASE / "labels"  / MODEL_SHORT
    ACTIV_DIR      = GDRIVE_BASE / NEW_REV / MODEL_SHORT   # CHANGED: new folder
    LOG_FILE       = ACTIV_DIR / "extraction_log.json"
    SUMMARY_FILE   = GDRIVE_BASE / NEW_REV / "activations_summary_all_models.json"   # CHANGED

    print(f"Model:        {MODEL_NAME}")
    print(f"Baseline:     {IS_BASELINE_MODEL}")
    print(f"Window:       ±{WINDOW_TOKENS} tokens")
    print(f"Output dir:   {ACTIV_DIR}")

    # ---- 03, cell 4 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 3 — Mount Google Drive
    # -----------------------------------------------------------------------------
    ACTIV_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Activation directory ready: {ACTIV_DIR}")

    # ---- 03, cell 5 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 4 — Load model and tokenizer
    # -----------------------------------------------------------------------------
    def load_model_and_tokenizer(model_name: str, use_4bit: bool):
        print(f"\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model ({'4-bit' if use_4bit else 'fp16'})...")
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

        model.eval()

        # Detect layer count and hidden dim automatically
        config     = model.config
        n_layers   = config.num_hidden_layers
        hidden_dim = config.hidden_size
        print(f"Model loaded. Layers: {n_layers}, Hidden dim: {hidden_dim}")
        return model, tokenizer, n_layers, hidden_dim


    model, tokenizer, N_LAYERS, HIDDEN_DIM = load_model_and_tokenizer(
        MODEL_NAME, USE_4BIT
    )

    # ---- 03, cell 6 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 5 — Hook registration
    # Hook into residual stream AFTER each transformer block.
    # We capture output[0] which is the hidden state tensor.
    # Hooks are registered once and stay active for all forward passes.
    # -----------------------------------------------------------------------------
    #------------------------------------------------------------------------
    class ActivationStore:
        """
        Thread-safe store for activations captured by forward hooks.
        Cleared before each forward pass via .clear().
        """
        def __init__(self):
            self.data = {}

        def clear(self):
            self.data.clear()

        def __setitem__(self, key, value):
            self.data[key] = value

        def __getitem__(self, key):
            return self.data[key]

        def keys(self):
            return self.data.keys()

    store = ActivationStore()


    hooks = []

    def register_hooks(model, store):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            raise RuntimeError(f"Cannot find layers in {type(model)}")

        hooks = []
        for i, layer in enumerate(layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    hidden_state = output[0]
                    if hidden_state.dim() == 3:
                        store[layer_idx] = hidden_state[0].detach().cpu().float()
                    elif hidden_state.dim() == 2:
                        store[layer_idx] = hidden_state.detach().cpu().float()
                    else:
                        print(f"WARNING layer {layer_idx}: unexpected shape {hidden_state.shape}")
                return hook
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        print(f"Registered {len(hooks)} hooks.")
        return hooks

    hooks = register_hooks(model, store)

    # Verify immediately
    store.clear()
    test_input = tokenizer("test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model(**test_input)
    print(f"Verification — keys in store: {len(store.keys())} layers")
    assert len(store.keys()) == N_LAYERS, f"Expected {N_LAYERS}, got {len(store.keys())}"
    print("Hooks verified OK.")

    # ---- 03, cell 7 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 6 — Text utilities
    # -----------------------------------------------------------------------------
    def clean_text(text: str) -> str:
        return (text.replace("Ġ", " ")
                    .replace("Ċ", "\n")
                    .replace("ĉ", "\t"))


    def get_analysis_text(trace: dict) -> str:
        think = clean_text(trace.get("think_content", ""))
        clean = clean_text(trace.get("response_clean", ""))
        return think if think.strip() else clean


    def word_idx_to_token_idx(
        text: str,
        word_idx: int,
        tokenizer,
        prompt_len: int
    ) -> int:
        """
        Convert a word index in the analysis text to a token index in the
        full model input (which includes the prompt prefix).

        Strategy:
        1. Split text into words
        2. Reconstruct text up to word_idx
        3. Tokenize that prefix to get token count
        4. Add prompt_len offset

        This is approximate but sufficient for ±30 token windows.
        """
        words  = text.split()
        prefix = " ".join(words[:word_idx])
        prefix_tokens = tokenizer(prefix, return_tensors="pt",
                                  add_special_tokens=False)
        n_prefix_tokens = prefix_tokens["input_ids"].shape[1]
        return prompt_len + n_prefix_tokens

    # ---- 03, cell 8 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 7 — Prompt builder (identical to notebook 01 to ensure same tokenization)
    # -----------------------------------------------------------------------------
    def build_prompt(problem: dict, tokenizer, model_name: str) -> str:
        is_reasoning = any(tag in model_name for tag in ["R1", "QwQ"])

        if problem.get("difficulty") == "hard":
            instruction = (
                "Solve the following competition mathematics problem carefully. "
                "Show your reasoning step by step and verify your answer."
            )
        else:
            instruction = (
                "Solve the following mathematics competition problem. "
                "Show your working step by step."
            )

        if is_reasoning:
            messages = [{"role": "user",
                         "content": f"{instruction}\n\nProblem: {problem['problem']}"}]
        else:
            messages = [
                {"role": "system",
                 "content": "You are an expert mathematician. Solve problems with "
                            "clear step-by-step reasoning."},
                {"role": "user",
                 "content": f"{instruction}\n\nProblem: {problem['problem']}"}
            ]

        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---- 03, cell 9 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 8 — Core extraction: run model and extract windows
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def extract_windows_for_problem(
        trace: dict,
        label: dict,
        model,
        tokenizer,
        store: ActivationStore,
        n_layers: int,
        window_tokens: int
    ) -> dict | None:
        """
        Extracts activation windows for a given problem based on revision labels.
        """
        pid        = trace["problem_id"]
        analysis   = get_analysis_text(trace)
        prompt     = build_prompt(trace, tokenizer, MODEL_NAME)
        inputs     = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate full input IDs for the model run
        full_text   = prompt + analysis
        full_inputs = tokenizer(full_text, return_tensors="pt", truncation=False).to(model.device)
        seq_len     = full_inputs["input_ids"].shape[1]

        # Convert revision word indices to token indices
        revision_entries = []
        for rev in label.get("revisions", []):
            word_idx       = rev.get("revision_word_idx", 0)
            ctrl_word_idx  = rev.get("control_word_idx", 0)

            token_idx      = word_idx_to_token_idx(
                analysis, word_idx, tokenizer, prompt_len
            )
            ctrl_token_idx = word_idx_to_token_idx(
                analysis, ctrl_word_idx, tokenizer, prompt_len
            )

            # Clamp to valid range
            token_idx      = max(window_tokens, min(token_idx,      seq_len - window_tokens - 1))
            ctrl_token_idx = max(window_tokens, min(ctrl_token_idx, seq_len - window_tokens - 1))
            revision_entries.append({
                "token_idx":      token_idx,
                "ctrl_token_idx": ctrl_token_idx,   # ← this was missing
                "verdict":        rev.get("verdict", "UNKNOWN"),
                "confidence":     rev.get("confidence", 0.0),
                "word_idx":       word_idx,
            })
        if not revision_entries:
            return None # No valid revisions to extract

        # Run model forward pass to populate store
        store.clear()
        _ = model(**full_inputs, output_hidden_states=False)

        # Instead of pre-allocating the full array upfront:
        # OLD — causes OOM on large n_revisions:
        # revision_windows = np.zeros((n_revisions, n_layers, window_size, HIDDEN_DIM), dtype=np.float16)

        # NEW — build list, stack at end:
        revision_windows_list = []
        control_windows_list  = [] # Control windows are not defined in the trace, commenting out
        revision_norms_list   = []
        control_norms_list    = [] # Control windows are not defined in the trace, commenting out

        window_size = 2 * window_tokens + 1

        for r, entry in enumerate(revision_entries):
            t_rev  = entry["token_idx"]
            t_ctrl = entry["ctrl_token_idx"] # Control tokens are not defined in the trace, commenting out

            rev_win_layers  = np.zeros((n_layers, window_size, HIDDEN_DIM), dtype=np.float16)
            ctrl_win_layers = np.zeros((n_layers, window_size, HIDDEN_DIM), dtype=np.float16) # Control windows are not defined in the trace, commenting out
            rev_norms       = np.zeros(n_layers, dtype=np.float32)
            ctrl_norms      = np.zeros(n_layers, dtype=np.float32) # Control windows are not defined in the trace, commenting out

            for l in range(n_layers):
                hidden = store[l]   # (seq_len, hidden_dim)

                rev_window = hidden[t_rev - window_tokens: t_rev + window_tokens + 1]
                if rev_window.shape[0] == window_size:
                    rev_win_layers[l] = rev_window.numpy().astype(np.float16)

                ctrl_window = hidden[t_ctrl - window_tokens: t_ctrl + window_tokens + 1] # Control windows are not defined in the trace, commenting out
                if ctrl_window.shape[0] == window_size:
                     ctrl_win_layers[l] = ctrl_window.numpy().astype(np.float16) # Control windows are not defined in the trace, commenting out

                if l > 0:
                    prev = store[l-1]
                    rev_norms[l]  = float(torch.norm(hidden[t_rev]  - prev[t_rev]).item())
                    # ctrl_norms[l] = float(torch.norm(hidden[t_ctrl] - prev[t_ctrl]).item()) # Control windows are not defined in the trace, commenting out

            revision_windows_list.append(rev_win_layers)
            control_windows_list.append(ctrl_win_layers) # Control windows are not defined in the trace, commenting out
            revision_norms_list.append(rev_norms)
            control_norms_list.append(ctrl_norms) # Control windows are not defined in the trace, commenting out

        # Stack into final arrays
        revision_windows = np.stack(revision_windows_list)  # (n_rev, n_layers, 61, hidden_dim)
        control_windows  = np.stack(control_windows_list) # Control windows are not defined in the trace, commenting out
        revision_norms   = np.stack(revision_norms_list)    # (n_rev, n_layers)
        control_norms    = np.stack(control_norms_list) # Control windows are not defined in the trace, commenting out

        store.clear()
        torch.cuda.empty_cache()
        gc.collect()
        revision_meta = [
            {
                "token_idx":      entry["token_idx"],
                "ctrl_token_idx": entry["ctrl_token_idx"],
                "verdict":        entry.get("verdict", "UNKNOWN"),
                "confidence":     entry.get("confidence", 0.0),
                "word_idx":       entry.get("word_idx", 0),
            }
            for entry in revision_entries
        ]

        return {
            "problem_id":       pid,
            "model":            MODEL_SHORT,
            "difficulty":       trace.get("difficulty", "unknown"),
            "is_baseline":      False,
            "n_revisions":      len(revision_entries),
            "n_layers":         n_layers,
            "hidden_dim":       HIDDEN_DIM,
            "window_tokens":    window_tokens,
            "revision_meta":    revision_meta,
            "revision_windows": np.nan_to_num(revision_windows, nan=0.0, posinf=0.0, neginf=0.0),
            "control_windows":  np.nan_to_num(control_windows,  nan=0.0, posinf=0.0, neginf=0.0),
            "revision_norms":   revision_norms,
            "control_norms":    control_norms,
        }

    # ---- 03, cell 10 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 9 — Baseline extraction for Llama (random windows)
    # Only runs when IS_BASELINE_MODEL = True
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def extract_baseline_windows(
        trace: dict,
        model,
        tokenizer,
        store: ActivationStore,
        n_layers: int,
        window: int,
        n_windows: int,
        rng: random.Random
    ) -> dict | None:
        """
        For Llama baseline: extract N random token windows from each trace.
        No revision labels needed — just random positions in the response.
        """
        pid         = trace["problem_id"]
        analysis    = get_analysis_text(trace)
        prompt      = build_prompt(trace, tokenizer, MODEL_NAME)
        inputs      = tokenizer(prompt, return_tensors="pt",
                                truncation=False).to(model.device)
        prompt_len  = inputs["input_ids"].shape[1]

        full_text   = prompt + analysis
        full_inputs = tokenizer(full_text, return_tensors="pt",
                                truncation=False).to(model.device)
        seq_len     = full_inputs["input_ids"].shape[1]

        if seq_len <= 2 * window + prompt_len + 10:
            return None   # trace too short for any window

        store.clear()
        _ = model(**full_inputs, output_hidden_states=False)

        # Sample random positions in the response portion only
        valid_range  = range(prompt_len + window, seq_len - window - 1)
        if len(valid_range) < n_windows:
            positions = list(valid_range)
        else:
            positions = rng.sample(list(valid_range), n_windows)

        window_size = 2 * window + 1
        n_pos       = len(positions)

        baseline_windows = np.zeros(
            (n_pos, n_layers, window_size, HIDDEN_DIM), dtype=np.float16
        )
        baseline_norms   = np.zeros((n_pos, n_layers), dtype=np.float32)

        for p_idx, t in enumerate(positions):
            for l in range(n_layers):
                hidden = store[l]
                win    = hidden[t - window: t + window + 1]
                if win.shape[0] == window_size:
                    baseline_windows[p_idx, l] = win.numpy().astype(np.float16)
                if l > 0:
                    prev    = store[l-1]
                    delta   = hidden[t] - prev[t]
                    baseline_norms[p_idx, l] = float(torch.norm(delta).item())

        store.clear()
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "problem_id":       pid,
            "model":            MODEL_SHORT,
            "difficulty":       trace.get("difficulty", "unknown"),
            "is_baseline":      True,
            "n_windows":        n_pos,
            "n_layers":         n_layers,
            "hidden_dim":       HIDDEN_DIM,
            "window_tokens":    window,
            "positions":        positions,
            "baseline_windows": baseline_windows,   # (n_pos, n_layers, 61, hidden_dim)
            "baseline_norms":   baseline_norms,     # (n_pos, n_layers)
        }

    # ---- 03, cell 11 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 10 — Save and load functions
    # -----------------------------------------------------------------------------
    def save_activations(result: dict, activ_dir: Path):
        """
        Save activation arrays as compressed .npz and metadata as .json.
        Keeps arrays and metadata separate for efficient loading downstream.
        """
        pid = result["problem_id"]

        # Save arrays
        array_keys = [k for k, v in result.items() if isinstance(v, np.ndarray)]
        arrays     = {k: result[k] for k in array_keys}
        np.savez_compressed(activ_dir / f"{pid}.npz", **arrays)

        # Save metadata (everything that's not an array)
        meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        with open(activ_dir / f"{pid}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


    def load_activations(pid: str, activ_dir: Path) -> tuple[dict, dict]:
        """
        Load arrays and metadata for one problem.
        """
        arrays = dict(np.load(activ_dir / f"{pid}.npz"))
        with open(activ_dir / f"{pid}_meta.json") as f:
            meta = json.load(f)
        return arrays, meta

    # ---- 03, cell 12 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 11 — Checkpoint log
    # -----------------------------------------------------------------------------
    def load_extraction_log() -> dict:
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                log = json.load(f)
            print(f"Resuming. Done: {len(log['completed'])} | "
                  f"Skipped: {len(log['skipped'])} | Failed: {len(log['failed'])}")
        else:
            log = {
                "model":       MODEL_SHORT,
                "is_baseline": IS_BASELINE_MODEL,
                "started":     datetime.now().isoformat(),
                "completed":   [],
                "skipped":     [],
                "failed":      [],
            }
            print("Starting fresh extraction run.")
        return log


    def save_extraction_log(log: dict):
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    # ---- 03, cell 13 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 12 — Build work queue
    # For reasoning models: pair trace files with label files.
    # For baseline model: use all non-truncated trace files directly.
    # -----------------------------------------------------------------------------
    def build_work_queue(
        trace_dir: Path,
        label_dir: Path,
        is_baseline: bool,
        completed: list
    ) -> list[tuple]:
        """
        Returns list of (trace, label_or_None) tuples to process.
        """
        done = set(completed)
        trace_files = [f for f in trace_dir.glob("*.json")
                       if f.name != "run_log.json"]

        queue = []
        print(len(trace_files))

        if is_baseline:
            # Baseline: all non-truncated traces, no labels needed
            for tf in trace_files:
                with open(tf) as f:
                    trace = json.load(f)
                if 'problem_id' not in trace:
                    print(f"Skipping trace {tf.name}: 'problem_id' key missing.")
                    continue
                if trace.get("truncated"):
                    continue
                if trace["problem_id"] in done:
                    continue
                queue.append((trace, None))
        else:
            # Reasoning models: only traces with a valid label file
            label_files = {f.stem: f for f in label_dir.glob("*.json")
                           if f.name != "labeling_log.json"
                           and f.name != "label_summary.json"}

            for tf in trace_files:
                with open(tf) as f:
                    trace = json.load(f)

                if 'problem_id' not in trace:
                    print(f"Skipping trace {tf.name}: 'problem_id' key missing.")
                    continue

                pid = trace["problem_id"]

                if pid in done:
                    continue
                if pid not in label_files:
                    continue   # no label — skipped in notebook 02

                with open(label_files[pid]) as f:
                    label = json.load(f)

                if label.get("skipped"):
                    continue   # notebook 02 decided to skip this trace

                queue.append((trace, label))

        return queue


    log   = load_extraction_log()
    queue = build_work_queue(
        TRACE_DIR, LABEL_DIR,
        IS_BASELINE_MODEL,
        log["completed"] + log["skipped"]
    )
    print(f"Work queue: {len(queue)} problems to extract")

    # ---- 03, cell 14 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 13 — Main extraction loop
    # -----------------------------------------------------------------------------
    rng = random.Random(RANDOM_SEED)

    # Estimate storage before starting
    if not IS_BASELINE_MODEL and queue:
        sample_label = queue[0][1]
        avg_rev_est  = sum(
            l.get("n_verified", 0) for _, l in queue if l
        ) / max(len(queue), 1)
        window_size  = 2 * WINDOW_TOKENS + 1
        est_gb       = (len(queue) * avg_rev_est * N_LAYERS
                        * window_size * HIDDEN_DIM * 2) / 1024**3
        print(f"Estimated storage: {est_gb:.1f} GB")
        print(f"  ({len(queue)} problems × {avg_rev_est:.1f} revisions × "
              f"{N_LAYERS} layers × {window_size} tokens × {HIDDEN_DIM} dims × 2 bytes)")

    print(f"\nStarting extraction: {len(queue)} problems")
    print(f"Model: {MODEL_NAME}")
    print(f"Baseline mode: {IS_BASELINE_MODEL}")
    print("=" * 65)

    total_revisions_extracted = 0
    for i, (trace, label) in enumerate(queue):
        pid  = trace["problem_id"]
        diff = trace.get("difficulty", "?")
        print(f"\n[{i+1}/{len(queue)}] {pid}  diff={diff}")

        t0 = time.time()
        try:
            if IS_BASELINE_MODEL:
                result = extract_baseline_windows(
                    trace, model, tokenizer, store,
                    N_LAYERS, WINDOW_TOKENS,
                    N_BASELINE_WINDOWS // max(len(queue), 1) + 1,
                    rng
                )
            else:
                result = extract_windows_for_problem(
                    trace, label, model, tokenizer, store,
                    N_LAYERS, WINDOW_TOKENS
                )

            if result is None:
                log["skipped"].append(pid)
                save_extraction_log(log)
                print(f"  SKIPPED: no valid windows")
                continue

            save_activations(result, ACTIV_DIR)
            log["completed"].append(pid)
            save_extraction_log(log)

            elapsed = time.time() - t0
            if IS_BASELINE_MODEL:
                n_extracted = result["n_windows"]
            else:
                n_extracted = result["n_revisions"]
                total_revisions_extracted += n_extracted

            # Estimate file size
            total_elements = sum(
                v.size for v in result.values() if isinstance(v, np.ndarray)
            )
            est_mb = total_elements * 2 / 1024**2   # float16 = 2 bytes

            print(f"  ✓ {n_extracted} windows | "
                  f"~{est_mb:.1f} MB | {elapsed:.1f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            log["failed"].append({"problem_id": pid, "error": str(e)})
            save_extraction_log(log)
            # Clear memory even on failure
            store.clear()
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "=" * 65)
    print(f"Extraction complete.")
    print(f"  Completed : {len(log['completed'])}")
    print(f"  Skipped   : {len(log['skipped'])}")
    print(f"  Failed    : {len(log['failed'])}")
    if not IS_BASELINE_MODEL:
        print(f"  Total revision windows extracted: {total_revisions_extracted}")

    # ---- 03, cell 15 (verbatim except where marked) ----
    # -----------------------------------------------------------------------------
    # CELL 14 — Storage audit and cross-model summary
    # Run after extraction completes. Checks actual disk usage and validates
    # all .npz files are readable before moving on.
    # -----------------------------------------------------------------------------
    def storage_audit(activ_dir: Path, model_short: str) -> dict:
        npz_files = list(activ_dir.glob("*.npz"))
        if not npz_files:
            print("No .npz files found.")
            return {}

        total_bytes     = 0
        total_revisions = 0
        total_problems  = 0
        corrupt_files   = []
        by_diff         = defaultdict(lambda: {"problems": 0, "revisions": 0, "bytes": 0})

        for npz_path in npz_files:
            pid       = npz_path.stem
            meta_path = activ_dir / f"{pid}_meta.json"

            # Check file is readable
            try:
                arrays = np.load(npz_path)
                _ = list(arrays.keys())   # force load
            except Exception as e:
                corrupt_files.append({"pid": pid, "error": str(e)})
                continue

            file_bytes = npz_path.stat().st_size
            total_bytes += file_bytes

            # Load metadata
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                diff = meta.get("difficulty", "unknown")
                n_rev = meta.get("n_revisions", meta.get("n_windows", 0))
            else:
                diff  = "unknown"
                n_rev = 0

            by_diff[diff]["problems"]   += 1
            by_diff[diff]["revisions"]  += n_rev
            by_diff[diff]["bytes"]      += file_bytes
            total_revisions += n_rev
            total_problems  += 1

        total_gb = total_bytes / 1024**3

        print(f"\n{'='*65}")
        print(f"STORAGE AUDIT — {model_short}")
        print(f"{'='*65}")
        print(f"  Total problems:   {total_problems}")
        print(f"  Total revisions:  {total_revisions}")
        print(f"  Total size:       {total_gb:.2f} GB")
        print(f"  Corrupt files:    {len(corrupt_files)}")

        for diff, stats in by_diff.items():
            gb = stats["bytes"] / 1024**3
            print(f"\n  {diff.upper()}")
            print(f"    Problems:   {stats['problems']}")
            print(f"    Revisions:  {stats['revisions']}")
            print(f"    Size:       {gb:.2f} GB")

        if corrupt_files:
            print(f"\n  ⚠ CORRUPT FILES — re-extract these:")
            for c in corrupt_files:
                print(f"    {c['pid']}: {c['error']}")

        summary = {
            "model":           model_short,
            "timestamp":       datetime.now().isoformat(),
            "total_problems":  total_problems,
            "total_revisions": total_revisions,
            "total_gb":        round(total_gb, 2),
            "corrupt_files":   corrupt_files,
            "by_difficulty":   {
                d: {k: v for k, v in s.items() if k != "bytes"}
                for d, s in by_diff.items()
            }
        }

        # Save per-model audit
        audit_path = activ_dir / "storage_audit.json"
        with open(audit_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Audit saved: {audit_path}")

        # Append to cross-model summary
        if SUMMARY_FILE.exists():
            with open(SUMMARY_FILE) as f:
                cross = json.load(f)
        else:
            cross = {}
        cross[model_short] = summary
        with open(SUMMARY_FILE, "w") as f:
            json.dump(cross, f, indent=2)
        print(f"  Cross-model summary updated: {SUMMARY_FILE}")

        return summary


    storage_audit(ACTIV_DIR, MODEL_SHORT)
    for _h in hooks: _h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()


# #############################################################################
# STAGE "filler" — notebook 05, verbatim (Llama-8B, new output folder)
# #############################################################################
if "filler" in STAGES:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # ---- 05, cell 2 (verbatim except where marked) ----
    # =============================================================================
    # 05 -- FILLER ACTIVATION EXTRACTION
    #
    # Extracts activation windows at confirmed-filler positions
    # (rejects/{model}_confirmed_filler.json, from notebook 04) with the same
    # hooks and window logic as notebook 03, writing
    # activations_filler/{model}/{problem_id}.npz.
    #
    # NOTE ON DATA COMPLETENESS: if ANY layer's window comes up short near a
    # sequence boundary for a given filler event, that event is dropped entirely
    # (not zero-padded) -- see the `ok` flag in the extraction loop. This means
    # activations_filler/ has FEWER events than rejects/{model}_confirmed_filler.json
    # lists. Do not assume 1:1 correspondence between the two without checking.
    #
    # NOTE ON METADATA: filler_meta only carries marker, token_idx, matched_text.
    # char_start / type_hint / filler_confidence are NOT duplicated into the
    # activation metadata -- cross-reference rejects/{model}_confirmed_filler.json
    # by matched_text/token position if those fields are needed downstream.
    # =============================================================================

    import torch, json, gc, numpy as np
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_SHORT = MODEL_NAME.split("/")[-1]
    WINDOW      = 30

    GDRIVE_BASE = DATA_ROOT

    # ---- 05, cell 3 (verbatim except where marked) ----
    REJECT_BASE = GDRIVE_BASE / "rejects"
    TRACE_DIR   = GDRIVE_BASE / "traces" / MODEL_SHORT
    OUT_DIR     = GDRIVE_BASE / NEW_FIL / MODEL_SHORT   # CHANGED: new folder
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    # ---- 05, cell 4 (verbatim except where marked) ----
    model.eval()
    N_LAYERS   = model.config.num_hidden_layers
    HIDDEN_DIM = model.config.hidden_size

    store = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def mk(idx):
            def hook(mod, inp, out):
                h = out[0]
                store[idx] = (h[0] if h.dim() == 3 else h).detach().cpu().float()
            return hook
        hooks.append(layer.register_forward_hook(mk(i)))
    print(f"{N_LAYERS} hooks registered")

    # ---- 05, cell 5 (verbatim except where marked) ----
    def clean_text(t):
        return t.replace("Ġ"," ").replace("Ċ","\n").replace("ĉ","\t")
    def get_analysis_text(tr):
        th = clean_text(tr.get("think_content",""))
        cl = clean_text(tr.get("response_clean",""))
        return th if th.strip() else cl
    def build_prompt(tr):
        instr = ("Solve the following competition mathematics problem carefully. "
                 "Show your reasoning step by step and verify your answer."
                 if tr.get("difficulty")=="hard" else
                 "Solve the following mathematics competition problem. "
                 "Show your working step by step.")
        msgs = [{"role":"user","content":f"{instr}\n\nProblem: {tr['problem']}"}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    def word_to_token(text, widx, plen):
        pref = " ".join(text.split()[:widx])
        return plen + tokenizer(pref, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].shape[1]

    with open(REJECT_BASE / f"{MODEL_SHORT}_confirmed_filler.json") as f:
        fillers = json.load(f)

    by_problem = {}
    for r in fillers:
        by_problem.setdefault(r["problem_id"], []).append(r)

    # ---- 05, cell 6 (verbatim except where marked) ----
    print(f"{len(fillers)} filler positions across {len(by_problem)} problems")

    for i, (pid, entries) in enumerate(sorted(by_problem.items())):
        if (OUT_DIR / f"{pid}.npz").exists():
            continue
        try:
            with open(TRACE_DIR / f"{pid}.json") as f:
                trace = json.load(f)
            analysis = get_analysis_text(trace)
            prompt   = build_prompt(trace)
            plen = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            full = tokenizer(prompt + analysis, return_tensors="pt").to(model.device)
            slen = full["input_ids"].shape[1]

            store.clear()
            with torch.no_grad():
                _ = model(**full)

            wins, metas = [], []
            for e in entries:
                t = word_to_token(analysis, e["word_idx"], plen)
                t = max(WINDOW, min(t, slen - WINDOW - 1))
                w = np.zeros((N_LAYERS, 2*WINDOW+1, HIDDEN_DIM), dtype=np.float16)
                ok = True
                for l in range(N_LAYERS):
                    sl = store[l][t-WINDOW : t+WINDOW+1]
                    if sl.shape[0] != 2*WINDOW+1:
                        ok = False; break
                    w[l] = sl.numpy().astype(np.float16)
                if ok:
                    wins.append(np.nan_to_num(w, nan=0., posinf=0., neginf=0.))
                    metas.append({"marker": e["marker"], "token_idx": int(t),
                                  "matched_text": e["matched_text"]})
            store.clear(); torch.cuda.empty_cache(); gc.collect()

            if not wins:
                continue
            np.savez_compressed(OUT_DIR / f"{pid}.npz",
                                filler_windows=np.stack(wins))
            with open(OUT_DIR / f"{pid}_meta.json", "w") as f:
                json.dump({"problem_id": pid, "n_layers": N_LAYERS,
                           "hidden_dim": HIDDEN_DIM, "difficulty": trace.get("difficulty"),
                           "filler_meta": metas}, f, indent=2)
            print(f"[{i+1}/{len(by_problem)}] {pid}: {len(wins)} filler windows")
        except Exception as ex:
            print(f"[{i+1}] {pid} FAILED: {ex}")
            store.clear(); torch.cuda.empty_cache(); gc.collect()

    print("Filler extraction done ->", OUT_DIR)
    for _h in hooks: _h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()


# #############################################################################
# STAGE "verify" — CPU, tokenizer only. Changes nothing.
# #############################################################################
if "verify" in STAGES:
    import numpy as np
    from collections import Counter
    from transformers import AutoTokenizer
    CENTER = 30
    tok = AutoTokenizer.from_pretrained(f"deepseek-ai/{LLAMA}", trust_remote_code=True)
    pf = json.load(open(DATA_ROOT / "results_v3" / "task1_residual_pointfix.json"))
    L = int(pf[LLAMA]["peak_layer"])
    import shutil, time, tempfile
    LOCAL_TMP = Path(tempfile.gettempdir()) / "verify_npz"; LOCAL_TMP.mkdir(exist_ok=True)
    def load_layer(path, key, layer):
        """One layer of an .npz on Drive. Copies the file to local disk first and
        retries on Drive I/O errors (Errno 5 / 107 / 103 / 108), remounting
        between attempts. Returns None if it still cannot be read."""
        for attempt in range(5):
            tmp = LOCAL_TMP / path.name
            try:
                shutil.copyfile(path, tmp)
                with np.load(tmp) as z:
                    return z[key][:, layer, :, :].astype(np.float32)
            except Exception as e:
                err = getattr(e, "errno", None) or getattr(getattr(e, "__context__", None), "errno", None)
                print(f"    read failed ({type(e).__name__}, errno {err}) on {path.name}, attempt {attempt+1}/5", flush=True)
                time.sleep(10 * (attempt + 1))
                try:
                    from google.colab import drive; drive.mount("/content/gdrive", force_remount=True)
                except Exception:
                    pass
            finally:
                try: tmp.unlink()
                except Exception: pass
        return None
    def rj(p, d=None):
        try: return json.load(open(p))
        except Exception: return d
    def clean_text(t): return t.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    def analysis(tr):
        th = clean_text(tr.get("think_content", "") or ""); cl = clean_text(tr.get("response_clean", "") or "")
        return th if th.strip() else cl
    def prompt_of(tr):
        instr = ("Solve the following competition mathematics problem carefully. "
                 "Show your reasoning step by step and verify your answer."
                 if tr.get("difficulty") == "hard" else
                 "Solve the following mathematics competition problem. Show your working step by step.")
        return tok.apply_chat_template([{"role": "user", "content": f"{instr}\n\nProblem: {tr['problem']}"}],
                                       tokenize=False, add_generation_prompt=True)
    rejects = {}
    for r in rj(DATA_ROOT / "rejects" / f"{LLAMA}_confirmed_filler.json", []) or []:
        rejects.setdefault(str(r["problem_id"]), []).append(r)
    report = {}
    for arm, old_dir, new_dir, wkey, mkey in (
            ("revision", DATA_ROOT / "activations" / LLAMA, DATA_ROOT / NEW_REV / LLAMA, "revision_windows", "revision_meta"),
            ("filler", DATA_ROOT / "activations_filler" / LLAMA, DATA_ROOT / NEW_FIL / LLAMA, "filler_windows", "filler_meta")):
        st = Counter(); d_marker, d_old_new, cos_same, best_off, best_cos = [], [], [], [], []
        unreadable = []
        new_metas = sorted(new_dir.glob("*_meta.json"))
        n_old = sum(len((rj(p, {}) or {}).get(mkey) or []) for p in old_dir.glob("*_meta.json"))
        for mp in new_metas:
            pid = mp.name[:-len("_meta.json")]
            nm = (rj(mp, {}) or {}).get(mkey) or []
            om = (rj(old_dir / mp.name, {}) or {}).get(mkey) or []
            tr = rj(DATA_ROOT / "traces" / LLAMA / f"{pid}.json")
            if tr is None: st["no trace"] += 1; continue
            text = analysis(tr); prompt = prompt_of(tr)
            enc = tok(prompt + text, return_offsets_mapping=True); offs = enc["offset_mapping"]; seq = len(enc["input_ids"])
            plen = len(tok(prompt)["input_ids"]); plen_c = len(prompt)
            if arm == "revision":
                srows = [r for r in ((rj(DATA_ROOT / "labels" / LLAMA / f"{pid}.json", {}) or {}).get("revisions") or [])]
            else:
                srows = rejects.get(pid, [])
            NW = load_layer(new_dir / f"{pid}.npz", wkey, L)
            if NW is None:
                st["NEW file unreadable after 5 tries (skipped; re-run verify)"] += 1; unreadable.append(f"new/{pid}"); continue
            OW = None
            if (old_dir / f"{pid}.npz").exists():
                OW = load_layer(old_dir / f"{pid}.npz", wkey, L)
                if OW is None:
                    st["old file unreadable after 5 tries (old-vs-new skipped for it)"] += 1; unreadable.append(f"old/{pid}")
            for i, m in enumerate(nm):
                # 1. recomputed centre == stored (the check Llama-8B failed before)
                src = srows[i] if (arm == "revision" and i < len(srows)) else None
                if arm == "filler":   # 05 may drop events; match on matched_text in order
                    src = next((r for r in srows if r.get("matched_text") == m.get("matched_text")
                                and not r.get("_used")), None)
                    if src is not None: src["_used"] = True
                w = (src or {}).get("revision_word_idx" if arm == "revision" else "word_idx")
                if w is not None:
                    t = plen + len(tok(" ".join(text.split()[:int(w)]), add_special_tokens=False)["input_ids"])
                    t = max(CENTER, min(t, seq - CENTER - 1))
                    st[f"recomputed centre {'==' if t == int(m['token_idx']) else '!='} stored"] += 1
                # 2. where the new centre sits relative to the marker's first token
                cs = (src or {}).get("char_start")
                if cs is not None:
                    pos = plen_c + int(cs)
                    mt = next((k for k, (a, b) in enumerate(offs) if a <= pos < b and b > a), None)
                    if mt is not None: d_marker.append(int(m["token_idx"]) - mt)
                # 3. old vs new window for the same event (same file, same row)
                if OW is not None and i < len(om) and i < OW.shape[0]:
                    d_old_new.append(int(om[i]["token_idx"]) - int(m["token_idx"]))
                    oc = OW[i, CENTER] / (np.linalg.norm(OW[i, CENTER]) + 1e-9)
                    nw = NW[i] / (np.linalg.norm(NW[i], axis=1, keepdims=True) + 1e-9)
                    cos = nw @ oc
                    cos_same.append(float(cos[CENTER])); k = int(cos.argmax())
                    best_off.append(k - CENTER); best_cos.append(float(cos[k]))
        def dist(a):
            if not a: return None
            a = np.array(a); return {"n": int(len(a)), "median": float(np.median(a)), "min": float(a.min()),
                                     "max": float(a.max()), "most_common": Counter(a.round(3).tolist()).most_common(8)}
        bc = np.array(best_cos) if best_cos else np.array([])
        report[arm] = {"events_new": sum(len((rj(p, {}) or {}).get(mkey) or []) for p in new_metas),
                       "events_old": n_old, "checks": dict(st),
                       "new_centre_minus_marker_first_token": dist(d_marker),
                       "share_new_centre_in_-3..+1": (float(np.mean([(-3 <= x <= 1) for x in d_marker])) if d_marker else None),
                       "old_token_idx_minus_new": dist(d_old_new),
                       "cos_old_centre_vs_new_centre": dist(cos_same),
                       "old_centre_found_in_new_window (cos>0.99)": (float(np.mean(bc > 0.99)) if len(bc) else None),
                       "offset_of_best_match_in_new_window": dist(best_off),
                       "unreadable_files": unreadable}
        print(f"\n  {arm}: {json.dumps(report[arm], default=str)}")
    json.dump(report, open(DATA_ROOT / "hand_labeling_export" / "llama8b_reextract_verification.json", "w"),
              indent=1, default=str)
    print("\n  wrote hand_labeling_export/llama8b_reextract_verification.json")


# #############################################################################
# STAGE "direction" — notebook 08 cell 1 (steering direction), verbatim, on the
# re-extracted Llama-8B windows. CPU. Writes a NEW file,
# hand_labeling_export/steering_directions_llama8b_reextract.json; the
# published steering_directions.json is untouched. Then compares the two.
# The strength multiplier (CHOSEN_ALPHA_MULTIPLIER, Llama-8B 1.5) is NOT
# re-chosen: 09 applies the published multiplier to the new direction's norm.
# #############################################################################
if "direction" in STAGES:
    # ---- 08, cell 1 (verbatim except where marked) ----
    # =============================================================================
    # EXTRACT STEERING DIRECTIONS -- diff-of-means from existing saved activations
    #
    # direction = mean(revision_windows[center]) - mean(filler_windows[center])
    #
    # Uses revision_windows (activations/) and filler_windows (activations_filler/)
    # -- the SAME classes as the probe's MATCHED contrast (revision-marker vs
    # filler-marker, same lexical marker family) -- not the generic
    # 'control_windows' field also present in activations/*.npz, which is a
    # weaker, non-lexically-matched contrast. Staying consistent with the
    # project's established lexical-confound-aware approach.
    #
    # Layer: does NOT depend on the probe's saved results JSON (path/existence
    # unconfirmed, would be a fragile dependency). Instead uses a simple,
    # self-contained heuristic grounded in the already-established finding that
    # the accuracy plateau begins ~32-38% of network depth across all four
    # models: picks a layer at 65% depth (comfortably inside the plateau, not
    # near either edge). n_layers is read directly from the activation array
    # shape, so this needs no external file at all.
    #
    # Output: steering_directions.json -- one entry per model with the direction
    # (unit-normalized), the chosen layer, and the raw (pre-normalization) norm
    # and mean per-position activation norm at that layer (both useful later for
    # picking sensible alpha values relative to the activation scale).
    # =============================================================================

    from pathlib import Path
    import json
    import numpy as np

    GDRIVE_BASE = DATA_ROOT
    ACTIV_BASE = GDRIVE_BASE / NEW_REV   # CHANGED
    FILLER_BASE = GDRIVE_BASE / NEW_FIL   # CHANGED
    EXPORT_DIR = GDRIVE_BASE / "hand_labeling_export"
    OUT_PATH = EXPORT_DIR / "steering_directions_llama8b_reextract.json"   # CHANGED: new file

    MODELS = ["DeepSeek-R1-Distill-Llama-8B"]   # CHANGED: Llama-8B only
    CENTER = 30                # same center-token convention used throughout this project
    PLATEAU_DEPTH_FRACTION = 0.65  # heuristic: comfortably inside the ~32-38%-onset plateau


    def peek_n_layers(base_dir, model, key):
        """Reads just ONE file's shape to determine n_layers, without loading
        everything. Needed before we know which layer to slice down to."""
        model_dir = base_dir / model
        if not model_dir.exists():
            return None
        for f in sorted(model_dir.glob("*.npz")):
            with np.load(f) as z:
                if key in z:
                    return z[key].shape[1]
        return None


    def load_sliced_windows(base_dir, model, key, layer, center):
        """Loads each file's array, immediately slices down to [:, layer, center, :],
        and discards the full array before moving to the next file. Peak memory is
        one file's full array at a time (transient), not every file's full array
        held simultaneously -- the previous version's np.concatenate(arrays) after
        the loop kept everything in memory at once, which is what blew up on
        Qwen-14B (48 layers x 5120 hidden dim -- ~51GB combined across both
        revision and filler arrays, held simultaneously, well past typical Colab
        RAM). This version only ever holds the small (n_events_in_file, hidden_dim)
        slices, which are orders of magnitude smaller."""
        slices = []
        model_dir = base_dir / model
        if not model_dir.exists():
            return None
        for f in sorted(model_dir.glob("*.npz")):
            with np.load(f) as z:
                if key not in z:
                    continue
                arr = z[key]                       # full array for this file, transient
                sliced = arr[:, layer, center, :].astype(np.float32).copy()
                slices.append(sliced)
                del arr                            # explicit, though scope-exit would do this anyway
        if not slices:
            return None
        return np.concatenate(slices, axis=0)      # concatenating small slices is cheap


    def extract_direction_for_model(model):
        n_layers = peek_n_layers(ACTIV_BASE, model, "revision_windows")
        if n_layers is None:
            print(f"  {model}: no revision_windows files found -- skipping")
            return None

        layer = int(round(PLATEAU_DEPTH_FRACTION * (n_layers - 1)))
        layer = max(0, min(layer, n_layers - 1))

        rev_vecs = load_sliced_windows(ACTIV_BASE, model, "revision_windows", layer, CENTER)
        fil_vecs = load_sliced_windows(FILLER_BASE, model, "filler_windows", layer, CENTER)

        if rev_vecs is None or fil_vecs is None:
            print(f"  {model}: missing revision_windows or filler_windows -- skipping")
            return None

        rev_mean = rev_vecs.mean(axis=0)
        fil_mean = fil_vecs.mean(axis=0)
        raw_direction = rev_mean - fil_mean
        raw_norm = float(np.linalg.norm(raw_direction))

        if raw_norm == 0:
            print(f"  {model}: degenerate zero-norm direction at layer {layer} -- skipping")
            return None

        unit_direction = raw_direction / raw_norm

        # mean activation norm at this layer/position -- reference scale for alpha calibration
        all_vecs = np.concatenate([rev_vecs, fil_vecs], axis=0)
        mean_activation_norm = float(np.linalg.norm(all_vecs, axis=1).mean())

        return {
            "model": model,
            "n_layers": n_layers,
            "layer": layer,
            "layer_fraction": layer / (n_layers - 1),
            "n_revision_events": int(rev_vecs.shape[0]),
            "n_filler_events": int(fil_vecs.shape[0]),
            "raw_direction_norm": raw_norm,
            "mean_activation_norm_at_layer": mean_activation_norm,
            "unit_direction": unit_direction.tolist(),
        }


    def main():
        results = {}
        for model in MODELS:
            print(f"\n{model}")
            r = extract_direction_for_model(model)
            if r is not None:
                results[model] = r
                print(f"  layer={r['layer']}/{r['n_layers']} ({r['layer_fraction']:.0%} depth)  "
                      f"n_revision={r['n_revision_events']}  n_filler={r['n_filler_events']}")
                print(f"  raw direction norm={r['raw_direction_norm']:.3f}  "
                      f"mean activation norm at this layer={r['mean_activation_norm_at_layer']:.3f}")
                print(f"  (ratio: direction is {r['raw_direction_norm']/r['mean_activation_norm_at_layer']:.1%} "
                      f"of typical activation magnitude at this layer -- a sanity check, not itself alpha)")

        with open(OUT_PATH, "w") as f:
            json.dump(results, f)
        print(f"\nWrote {len(results)} model directions to {OUT_PATH}")


    main()   # CHANGED: called directly (was under if __name__ == "__main__")
    import numpy as np
    _old = json.load(open(DATA_ROOT / "hand_labeling_export" / "steering_directions.json"))
    _old = _old.get("models", _old)
    _new = json.load(open(DATA_ROOT / "hand_labeling_export" / "steering_directions_llama8b_reextract.json"))
    o, n = _old[LLAMA], _new[LLAMA]
    uo, un = np.array(o["unit_direction"]), np.array(n["unit_direction"])
    print(f"\n  Llama-8B steering direction, original vs re-extracted windows:")
    print(f"    layer                {o.get('layer')}  vs  {n['layer']}")
    print(f"    events (rev/fil)     {o.get('n_revision_events')}/{o.get('n_filler_events')}  vs  "
          f"{n['n_revision_events']}/{n['n_filler_events']}")
    print(f"    raw direction norm   {o['raw_direction_norm']:.4f}  vs  {n['raw_direction_norm']:.4f}"
          f"   (alpha = 1.5 x this: {1.5*o['raw_direction_norm']:.3f} vs {1.5*n['raw_direction_norm']:.3f})")
    print(f"    cosine(original unit direction, new unit direction) = {float(uo @ un):+.4f}")


# #############################################################################
# STAGE "probe" — notebook 06 Parts A and B, verbatim (Llama-8B, new folders,
# separate result files). CPU.
# #############################################################################
if "probe" in STAGES:
    # ---- 06, cell 1 (verbatim except where marked) ----
    # =============================================================================
    # 06 PART A -- MATCHED-MARKER PROBE (Table 1: matched, null, all CIs)
    #
    # For each model: pick the layer with the highest matched accuracy (revision
    # vs filler windows, centre position = token before the marker), then
    # bootstrap matched, null (filler vs filler) and their within-replicate
    # difference, N_BOOT = 2000.
    #
    #   * Group-aware CV (StratifiedGroupKFold on original event ids), so a
    #     resampled duplicate never sits on both sides of a fold.
    #   * StandardScaler -> PCA(50, fit inside each fold) -> logistic regression.
    #   * Matched / null point estimates are the mean of N_PT_DRAWS = 30 balanced
    #     draws.
    #   * The full per-layer accuracy profile is saved with each model.
    #
    # The residual point estimate written here is the UNPAIRED difference
    # (matched_pt - null_pt). Part B recomputes it paired, which is what Table 1
    # reports.
    #
    # CPU, hours. Resumable: models already in OUTFILE are skipped.
    # =============================================================================

    from pathlib import Path
    GDRIVE_BASE = DATA_ROOT
    assert (GDRIVE_BASE / "activations").exists(),        "activations/ missing -- bad mount"
    assert (GDRIVE_BASE / "activations_filler").exists(), "fillers missing -- bad mount"
    print("Drive mounted, data visible.")

    import json, gc
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    import warnings
    warnings.filterwarnings("ignore")

    ACTIV_BASE  = GDRIVE_BASE / NEW_REV   # CHANGED
    FILLER_BASE = GDRIVE_BASE / NEW_FIL   # CHANGED
    RESULTS_DIR = GDRIVE_BASE / "results_v3"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    MODELS = ["DeepSeek-R1-Distill-Llama-8B"]   # CHANGED: Llama-8B only

    CENTER       = 30
    SEED         = 42
    N_PCA        = 50          # pre-committed headline dimensionality
    MAX_ITER     = 2000
    N_BOOT       = 2000
    N_PT_DRAWS   = 30          # averaged point-estimate draws
    FLAT_WINDOW  = 0.02        # layers within this of peak are "flat with it"

    OUTFILE = RESULTS_DIR / "task1_residual_cis_nboot2000_v3_llama8b_reextract.json"   # CHANGED

    # ---- 06, cell 2 (verbatim except where marked) ----
    def load_rev(m):
        rev = []
        for p in sorted((ACTIV_BASE / m).glob("*.npz")):
            rw = np.load(p).get("revision_windows")
            if rw is not None:
                rev.extend(rw[i] for i in range(rw.shape[0]))
        return rev

    def load_fil(m):
        fil = []
        for p in sorted((FILLER_BASE / m).glob("*.npz")):
            w = np.load(p).get("filler_windows")
            if w is not None:
                fil.extend(w[i] for i in range(w.shape[0]))
        return fil


    # -----------------------------------------------------------------------
    # cv_accuracy threads original-event-id groups through every
    # subsampling step and uses StratifiedGroupKFold. Every array of features
    # passed in (Xp, Xn) must be accompanied by a same-length array of original
    # event IDs (gp, gn) so duplicates introduced by bootstrap resampling never
    # get split across train/test.
    # -----------------------------------------------------------------------
    def cv_accuracy(Xp, Xn, gp, gn, seed):
        """Balanced 5-fold, group-aware CV accuracy. PCA fit inside each fold."""
        n = min(len(Xp), len(Xn))
        if n < 20:
            return 0.5
        rng = np.random.RandomState(seed)
        idx_p = rng.choice(len(Xp), n, False)
        idx_n = rng.choice(len(Xn), n, False)
        X = np.vstack([Xp[idx_p], Xn[idx_n]]).astype(np.float32)
        X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
        y = np.array([1]*n + [0]*n)
        g = np.concatenate([gp[idx_p], gn[idx_n]])

        ncomp = min(N_PCA, len(y) - 1)
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n_components=ncomp, random_state=seed)),
            ("clf", LogisticRegression(max_iter=MAX_ITER, random_state=seed)),
        ])
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in cv.split(X, y, groups=g):
            pipe.fit(X[tr], y[tr])
            accs.append((pipe.predict(X[te]) == y[te]).mean())
        return float(np.mean(accs))


    def averaged_point_estimate(Xp, Xn, gp, gn, base_seed, n_draws=N_PT_DRAWS):
        """Point estimate = mean cv_accuracy over n_draws independent
        balanced subsample draws, so it isn't hostage to one lucky split."""
        vals = [cv_accuracy(Xp, Xn, gp, gn, base_seed + 10_000 + d) for d in range(n_draws)]
        return float(np.mean(vals)), vals


    def peak_layer(rev, fil, nl):
        """Returns the peak layer and the full per-layer accuracy profile."""
        profile = []
        best_l, best_a = 0, 0.0
        rev_ids = np.arange(len(rev))
        fil_ids = np.arange(len(fil)) + 10**7   # disjoint id space from revisions
        for l in range(nl):
            Xp = np.array([w[l, CENTER] for w in rev], np.float32)
            Xn = np.array([w[l, CENTER] for w in fil], np.float32)
            a = cv_accuracy(Xp, Xn, rev_ids, fil_ids, SEED)
            profile.append(a)
            if a > best_a:
                best_a, best_l = a, l
        near_peak = [l for l, acc in enumerate(profile) if best_a - acc < FLAT_WINDOW]
        is_flat = len(near_peak) > 3
        return best_l, best_a, profile, near_peak, is_flat


    def bootstrap_all(rev, fil, L):
        """
        Returns point estimates (averaged over draws) and percentile CIs for
        matched, null, and residual. Bootstrap resamples EVENTS with replacement
        and REFITS both probes each iteration; original event IDs are carried
        through the resample step so cv_accuracy's group-aware split
        can prevent the same event landing on both sides of a fold.
        """
        Xr = np.array([w[L, CENTER] for w in rev], dtype=np.float32)
        Xf = np.array([w[L, CENTER] for w in fil], dtype=np.float32)
        gid_r = np.arange(len(Xr))
        gid_f = np.arange(len(Xf)) + 10**7

        # ---- point estimates (averaged over N_PT_DRAWS balanced draws) ----
        matched_pt, matched_draws = averaged_point_estimate(Xr, Xf, gid_r, gid_f, SEED)

        permf = np.random.RandomState(SEED).permutation(len(Xf))
        hf    = len(Xf) // 2
        Xf_a, Xf_b_half = Xf[permf[:hf]], Xf[permf[hf:2*hf]]
        gf_a, gf_b_half = gid_f[permf[:hf]], gid_f[permf[hf:2*hf]]
        # null halves are disjoint filler subsets -> give them disjoint synthetic
        # group ids too, since they're different physical events by construction
        null_pt, null_draws = averaged_point_estimate(Xf_a, Xf_b_half, gf_a, gf_b_half, SEED)

        resid_pt = matched_pt - null_pt

        # ---- bootstrap ----
        rng = np.random.RandomState(SEED)
        matched_b, null_b, resid_b = [], [], []
        for b in range(N_BOOT):
            ir  = rng.choice(len(Xr), len(Xr), replace=True)
            iff = rng.choice(len(Xf), len(Xf), replace=True)
            Xr_b, gr_b = Xr[ir], gid_r[ir]
            Xf_b, gf_b = Xf[iff], gid_f[iff]

            m_b = cv_accuracy(Xr_b, Xf_b, gr_b, gf_b, SEED + b)              # refit, group-aware

            pf  = np.random.RandomState(SEED + b).permutation(len(Xf_b))
            h   = len(Xf_b) // 2
            n_b = cv_accuracy(Xf_b[pf[:h]], Xf_b[pf[h:2*h]],
                               gf_b[pf[:h]], gf_b[pf[h:2*h]], SEED + b)      # refit, group-aware

            matched_b.append(m_b)
            null_b.append(n_b)
            resid_b.append(m_b - n_b)

            if (b + 1) % 250 == 0:
                print(f"      boot {b+1}/{N_BOOT}")

        def ci(arr):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            return float(lo), float(hi)

        m_lo, m_hi = ci(matched_b)
        n_lo, n_hi = ci(null_b)
        r_lo, r_hi = ci(resid_b)

        return {
            "matched":      matched_pt, "matched_ci":  [m_lo, m_hi],
            "null_fil":     null_pt,    "null_ci":     [n_lo, n_hi],
            "residual":     resid_pt,   "residual_ci": [r_lo, r_hi],
            "excludes_zero":        bool(r_lo > 0),
            "null_brackets_chance": bool(n_lo <= 0.50 <= n_hi),
            "intervals_disjoint":   bool(m_lo > n_hi),
            # Acceptance-criterion helper: does the point estimate land
            # comfortably inside its own bootstrap interval, or hug an edge?
            "matched_pt_ok": bool(m_lo < matched_pt < m_hi),
            "null_pt_ok":    bool(n_lo < null_pt < n_hi),
            "n_boot": N_BOOT,
            "n_pt_draws": N_PT_DRAWS,
        }

    # ---- 06, cell 3 (verbatim except where marked) ----
    # ---- resume support -------------------------------------------------------
    results = {}
    if OUTFILE.exists():
        results = json.load(open(OUTFILE))
        print(f"Resuming -- already done: {list(results.keys())}")

    # ---- 06, cell 4 (verbatim except where marked) ----
    for m in MODELS:
        if m in results:
            print(f"  {m}: already done, skipping")
            continue

        print(f"\n{'='*60}\n{m}\n{'='*60}")
        rev = load_rev(m)
        fil = load_fil(m)
        nl  = rev[0].shape[0]
        print(f"  revisions={len(rev)}  fillers={len(fil)}")

        L, a, profile, near_peak, is_flat = peak_layer(rev, fil, nl)
        print(f"  peak layer = {L}/{nl}  (matched acc={a:.3f})")
        print(f"  layers within {FLAT_WINDOW} of peak: {near_peak}")
        print(f"  {'FLAT PROFILE -- argmax is likely not a real peak' if is_flat else 'looks like a genuine peak'}")

        print(f"  bootstrapping (N_BOOT={N_BOOT}, group-aware CV, "
              f"point estimate averaged over {N_PT_DRAWS} draws)...")
        r = bootstrap_all(rev, fil, L)

        print(f"    matched  = {r['matched']:.3f}  "
              f"[{r['matched_ci'][0]:.3f}, {r['matched_ci'][1]:.3f}]  pt_inside_ci={r['matched_pt_ok']}")
        print(f"    null-fil = {r['null_fil']:.3f}  "
              f"[{r['null_ci'][0]:.3f}, {r['null_ci'][1]:.3f}]"
              f"   brackets 0.50: {r['null_brackets_chance']}  pt_inside_ci={r['null_pt_ok']}")
        print(f"    RESIDUAL = {r['residual']:+.3f}  "
              f"[{r['residual_ci'][0]:+.3f}, {r['residual_ci'][1]:+.3f}]"
              f"   excludes 0: {r['excludes_zero']}")
        print(f"    matched/null intervals disjoint: {r['intervals_disjoint']}")

        results[m] = {"peak_layer": L, "layer_profile": profile,
                      "near_peak_layers": near_peak, "flat_profile": is_flat,
                      "n_rev": len(rev), "n_fil": len(fil), **r}
        json.dump(results, open(OUTFILE, "w"), indent=2)
        del rev, fil
        gc.collect()

    # ---- 06, cell 5 (verbatim except where marked) ----
    # ---- SUMMARY --------------------------------------------------------------
    print("\n" + "#"*78)
    print(f"#  MATCHED-MARKER PROBE (N_BOOT={N_BOOT}, group-aware CV, averaged point estimate)")
    print("#"*78)
    print(f"\n  {'Model':<16} {'matched [95% CI]':>24} {'null [95% CI]':>24} "
          f"{'residual [95% CI]':>26}")
    print("  " + "-"*92)
    for m in MODELS:
        if m not in results:
            continue
        r = results[m]
        short = m.replace("DeepSeek-R1-Distill-", "R1-")
        print(f"  {short:<16} "
              f"{r['matched']:.3f} [{r['matched_ci'][0]:.3f},{r['matched_ci'][1]:.3f}]".rjust(24)
              + "  " +
              f"{r['null_fil']:.3f} [{r['null_ci'][0]:.3f},{r['null_ci'][1]:.3f}]".rjust(22)
              + "  " +
              f"{r['residual']:+.3f} [{r['residual_ci'][0]:+.3f},{r['residual_ci'][1]:+.3f}]".rjust(24))

    print("\n  ACCEPTANCE CRITERIA")
    print("  " + "-"*76)
    for m in MODELS:
        if m not in results:
            continue
        r = results[m]
        short = m.replace("DeepSeek-R1-Distill-", "R1-")
        print(f"    {short:<16} pt-in-CI(matched/null): "
              f"{str(r['matched_pt_ok']):>5}/{str(r['null_pt_ok']):>5}   "
              f"residual excl. 0: {str(r['excludes_zero']):>5}   "
              f"null brackets .50: {str(r['null_brackets_chance']):>5}   "
              f"flat peak profile: {str(r['flat_profile']):>5}")

    n_pass    = sum(results[m]["excludes_zero"] for m in results)
    n_null_ok = sum(results[m]["null_brackets_chance"] for m in results)
    n_pt_ok   = sum(results[m]["matched_pt_ok"] and results[m]["null_pt_ok"] for m in results)
    print(f"\n  {n_pass}/{len(results)} residual CIs exclude zero.")
    print(f"  {n_null_ok}/{len(results)} null CIs bracket chance (0.50) as they should.")
    print(f"  {n_pt_ok}/{len(results)} models have BOTH point estimates comfortably inside their CIs.")
    if n_pass == len(results) and n_null_ok == len(results) and n_pt_ok == len(results):
        print("  >>> All acceptance criteria met (duplicate-leakage-free).")
    else:
        print("  >>> One or more acceptance criteria failed -- see per-model detail above "
              "before writing anything up.")

    # ---- 06, cell 6 (verbatim except where marked) ----
    # =============================================================================
    # 06 PART B -- PAIRED RESIDUAL POINT ESTIMATE (Table 1: residual column)
    #
    # Part A's residual point estimate is matched_pt - null_pt, two separately
    # averaged marginals, while its CI comes from within-replicate differences.
    # They describe different quantities, and for Qwen-7B the point estimate
    # fell outside its own CI. This recomputes the point estimate the way the
    # bootstrap pairs it: one balanced filler subsample per draw, used for both
    # the matched negative class and the null split, differenced within the draw,
    # averaged over N_PT_DRAWS = 100 draws, at Part A's saved peak layer.
    #
    # matched, null and all three CIs are carried over from Part A unchanged.
    # Writes results_v3/task1_residual_pointfix.json (the file Table 1 cites:
    # column `residual_paired`).
    #
    # CPU, minutes per model.
    # =============================================================================

    from pathlib import Path
    import json, gc
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    import warnings
    warnings.filterwarnings("ignore")

    GDRIVE_BASE = DATA_ROOT
    ACTIV_BASE  = GDRIVE_BASE / NEW_REV   # CHANGED
    FILLER_BASE = GDRIVE_BASE / NEW_FIL   # CHANGED

    # Part A's results file
    V3_RESULTS_PATH = GDRIVE_BASE / "results_v3" / "task1_residual_cis_nboot2000_v3_llama8b_reextract.json"   # CHANGED
    OUT_PATH        = GDRIVE_BASE / "results_v3" / "task1_residual_pointfix_llama8b_reextract.json"   # CHANGED

    MODELS = ["DeepSeek-R1-Distill-Llama-8B"]   # CHANGED: Llama-8B only

    CENTER     = 30
    SEED       = 42
    N_PCA      = 50     # must match Part A
    MAX_ITER   = 2000   # must match Part A
    N_PT_DRAWS = 100    # paired draws (Part A used 30 for its unpaired marginals)


    def load_rev(m):
        out = []
        for p in sorted((ACTIV_BASE / m).glob("*.npz")):
            with np.load(p) as z:
                rw = z.get("revision_windows")
                if rw is not None:
                    out.extend(rw[i] for i in range(rw.shape[0]))
        return out


    def load_fil(m):
        out = []
        for p in sorted((FILLER_BASE / m).glob("*.npz")):
            with np.load(p) as z:
                w = z.get("filler_windows")
                if w is not None:
                    out.extend(w[i] for i in range(w.shape[0]))
        return out


    def cv_accuracy(Xp, Xn, gp, gn, seed):
        """Identical to Part A's: balanced, group-aware, PCA fit inside each fold."""
        n = min(len(Xp), len(Xn))
        if n < 20:
            return 0.5
        rng = np.random.RandomState(seed)
        ip = rng.choice(len(Xp), n, False)
        inn = rng.choice(len(Xn), n, False)
        X = np.vstack([Xp[ip], Xn[inn]]).astype(np.float32)
        X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
        y = np.array([1]*n + [0]*n)
        g = np.concatenate([gp[ip], gn[inn]])
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n_components=min(N_PCA, len(y) - 1), random_state=seed)),
            ("clf", LogisticRegression(max_iter=MAX_ITER, random_state=seed)),
        ])
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in cv.split(X, y, groups=g):
            pipe.fit(X[tr], y[tr])
            accs.append((pipe.predict(X[te]) == y[te]).mean())
        return float(np.mean(accs))


    def paired_residual(Xr, Xf, gid_r, gid_f, base_seed, n_draws=N_PT_DRAWS):
        """One balanced filler subsample per draw, used for BOTH the matched
        negative class and the null split -- mirroring what each bootstrap
        replicate does, so the point estimate and the CI describe the same
        quantity."""
        vals = []
        for d in range(n_draws):
            seed = base_seed + 10_000 + d
            rng = np.random.RandomState(seed)
            n = min(len(Xr), len(Xf))
            ir = rng.choice(len(Xr), n, False)
            iff = rng.choice(len(Xf), n, False)
            Xr_d, gr_d = Xr[ir], gid_r[ir]
            Xf_d, gf_d = Xf[iff], gid_f[iff]

            m = cv_accuracy(Xr_d, Xf_d, gr_d, gf_d, seed)

            pf = np.random.RandomState(seed).permutation(len(Xf_d))
            h = len(Xf_d) // 2
            n_ = cv_accuracy(Xf_d[pf[:h]], Xf_d[pf[h:2*h]],
                              gf_d[pf[:h]], gf_d[pf[h:2*h]], seed)
            vals.append(m - n_)
        return float(np.mean(vals)), vals


    def main():
        assert V3_RESULTS_PATH.exists(), (
            f"{V3_RESULTS_PATH} not found -- point this at your v3 results JSON.")
        with open(V3_RESULTS_PATH) as f:
            v3 = json.load(f)
        print(f"Loaded v3 results for: {list(v3.keys())}\n")

        out = {}
        for m in MODELS:
            if m not in v3:
                print(f"{m}: not in v3 results, skipping")
                continue
            r = v3[m]
            layer = r["peak_layer"]
            old_resid = r["residual"]
            lo, hi = r["residual_ci"]

            print(f"{'='*66}\n{m}\n{'='*66}")
            print(f"  peak layer {layer} (reused from v3 -- no layer sweep)")
            print(f"  v3 residual {old_resid:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"inside CI: {lo < old_resid < hi}")

            rev, fil = load_rev(m), load_fil(m)
            Xr = np.array([w[layer, CENTER] for w in rev], dtype=np.float32)
            Xf = np.array([w[layer, CENTER] for w in fil], dtype=np.float32)
            gid_r = np.arange(len(Xr))
            gid_f = np.arange(len(Xf)) + 10**7
            print(f"  n_rev={len(Xr)}  n_fil={len(Xf)}  -- computing {N_PT_DRAWS} paired draws...")

            new_resid, draws = paired_residual(Xr, Xf, gid_r, gid_f, SEED)
            inside = lo < new_resid < hi
            print(f"  NEW residual {new_resid:+.4f}  (draw sd {np.std(draws):.4f})")
            print(f"  inside its own CI: {inside}   change {new_resid - old_resid:+.4f}")

            out[m] = {
                "peak_layer": layer,
                "residual_v3_unpaired": old_resid,
                "residual_paired": new_resid,
                "residual_ci": [lo, hi],
                "inside_ci_before": bool(lo < old_resid < hi),
                "inside_ci_after": bool(inside),
                "n_pt_draws": N_PT_DRAWS,
                "draw_sd": float(np.std(draws)),
                # unchanged, carried through for a complete table
                "matched": r["matched"], "matched_ci": r["matched_ci"],
                "null_fil": r["null_fil"], "null_ci": r["null_ci"],
            }
            del rev, fil, Xr, Xf
            gc.collect()

        print(f"\n{'#'*66}\n# CORRECTED RESIDUAL TABLE\n{'#'*66}")
        print(f"{'Model':<32} {'old':>8} {'new':>8}  {'CI':>22}  in CI")
        all_ok = True
        for m, r in out.items():
            lo, hi = r["residual_ci"]
            ok = r["inside_ci_after"]
            all_ok &= ok
            print(f"{m:<32} {r['residual_v3_unpaired']:+.4f} {r['residual_paired']:+.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]  {ok}")
        print(f"\nAll point estimates inside their own CIs: {all_ok}")
        if not all_ok:
            print("If any remain outside, that is genuine skew rather than the pairing")
            print("mismatch -- worth understanding before publishing, not patching over.")

        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {OUT_PATH}")
        print("\nNOTE: matched, null, and ALL THREE CIs are unchanged from v3. Only the")
        print("residual POINT ESTIMATE is recomputed. The CIs were already paired")
        print("(bootstrap replicates difference within-replicate), so they were always")
        print("correct -- it was the point estimate that was inconsistent with them.")


    if __name__ == "__main__":
        main()
