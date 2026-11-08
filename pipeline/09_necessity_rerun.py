# =============================================================================
# 09 — NECESSITY: LEXICAL SUPPRESSION AND ACTIVATION STEERING
#
# Run in Colab (GPU for the two generation stages). Set STAGE below, run the
# whole file, read the output, then move to the next stage:
#     select -> lexical -> steering -> analyze
#
# The paper's steering numbers are STEERING_CORRECTED in `analyze`: the three
# Qwen models from STAGE = "steering" (published directions), and Llama-8B from
# a second STAGE = "steering" run with LLAMA_DIRECTION = "reextract", whose
# direction is rebuilt from the re-extracted Llama-8B windows
# (pipeline/11_reextract_llama8b.py, stages revision, filler, direction).
# pipeline/RUN_llama8b_direction_all.py runs that second part in one go.
#
# SAMPLE = "original" (default): the 83 revision + 80 filler events of the
#   first version of the paper (counterfactual_pilot_events.json and
#   filler_control_pilot_events.json), re-run with clean prefixes.
# SAMPLE = "redraw": a new uniform draw of up to N_PAIRS matched pairs.
#
# Outputs go to hand_labeling_export/redraw_2026_10/[original_rerun_clean_prefix/];
# nothing published is overwritten. Stages resume where they stopped.
# =============================================================================
# =============================================================================
# NECESSITY RE-RUN / REDRAW
#
# Oct 27: SAMPLE = "original" (default) re-runs the PUBLISHED 83/80 events
# with byte-level-BPE-free prefixes (all 39 old Llama-8B prefixes were in Ġ/Ċ
# characters), the two never-generated lexical events, and event-keyed resume.
# Output: hand_labeling_export/redraw_2026_10/original_rerun_clean_prefix/
# SAMPLE = "redraw" is the 160+160 uniform redraw described below.
#
# Replaces the inherited 83/80 sample for the revision. The old results are
# NEVER read for writing and NEVER overwritten: everything goes to
#   hand_labeling_export/redraw_2026_10/
#
# Run one stage at a time, in order:
#   STAGE = "select"     CPU, minutes.  Builds the event files. Read its report.
#   STAGE = "lexical"    GPU.  Lexical suppression rollouts, both arms.
#   STAGE = "steering"   GPU.  Activation-steering rollouts, both arms.
#   STAGE = "analyze"    CPU, minutes.  All reported outputs, old beside new.
#
# What is different from the 83/80 pipeline, and why — every item is logged
# in manifest.json:
#   1. Eligibility uses the CORRECTED parser (extract_answer_new +
#      normalise_answer + load_gt(use_repaired)), not the saved `correct` flag.
#   2. The revision arm is drawn uniformly from all GPT-4o-verified revisions
#      in eligible traces — no human-label gate. The filler arm is drawn from
#      GPT-4o-confirmed fillers in eligible traces, 1:1 on (model, marker),
#      with the marker normalised exactly as the filler-arm selection did
#      (superseded/necessity_lexical_filler_arm.ipynb).
#   3. Every offset is VERIFIED: the marker text must sit at char_start in the
#      trace. Failures are dropped and counted, never guessed.
#   4. Resume is keyed on the EVENT — (arm, model, problem_id, char_start) —
#      not (problem_id, model). The old key silently dropped the second event
#      of any problem with two events on a resumed run; the steering generator
#      (superseded/necessity_steering_both_arms.ipynb) had the same key.
#   5. PREFIX RESTORATION. The old prefixes were full_response[:cs] passed to
#      the tokenizer raw. If a trace was stored in byte-level-BPE character
#      space (literal Ġ for space, Ċ for newline), the model was conditioned
#      on text it never produced. Here the slice is taken FIRST, then decoded,
#      so offsets stay valid. The path taken for each event is recorded, and
#      the select stage reports how many OLD prefixes were affected.
#   6. Seeds fixed and logged. Wall-clock per stage and per event, GPU name,
#      and package versions are written to manifest.json, so the compute
#      section can be measured rather than reconstructed.
#
# Production code is EMBEDDED verbatim (see "EMBEDDED PRODUCTION CODE"
# below): generators, leak filter, answer extractor, ground-truth loader,
# necessity scorer and marker normaliser. No code is read from Drive; Drive
# holds data only.
# =============================================================================

STAGE         = "select"          # select | lexical | steering | analyze
LLAMA_DIRECTION = "original"      # steering stage only:
                                  # "original":  directions from steering_directions.json
                                  #   (Llama-8B's was built from the original Llama-8B
                                  #   windows, which could not be reproduced)
                                  # "reextract": Llama-8B ONLY, direction rebuilt from the
                                  #   re-extracted windows (11_reextract_llama8b.py, stage
                                  #   "direction"); same seeds and strength multiplier;
                                  #   separate rollout files
SAMPLE        = "original"        # "original": re-run the published 83/80
                                  #   events with clean prefixes (Oct 27)
                                  # "redraw":   new 160+160 uniform draw
RERUN_MODELS  = None              # None = all four models. To re-run only
                                  # the model whose prefixes were corrupted:
                                  # ["DeepSeek-R1-Distill-Llama-8B"]
N_PAIRS       = 160               # revision/filler pairs to draw (redraw only)
INTERVENTIONS = ("lexical", "steering")
SEEDS = {"select": 20261020, "match": 20261021,
         # rollout seed = base + r, as in the published runs:
         #   lexical  1000 + r  (superseded/necessity_lexical_revision_arm.py,
         #                       necessity_lexical_filler_arm.ipynb)
         #   steering 4000 + r  (calling loop in necessity_steering_both_arms.ipynb)
         "rollout_base": {"lexical": 1000, "steering": 4000,
                          "steering_llama8b_reextract": 4000}}
MODELS = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-1.5B",
          "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-14B"]
N_BOOT = 2000

import ast, hashlib, inspect, json, math, os, random, re, sys, time, types
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

BASE    = Path(os.environ.get("REVISION_DATA_ROOT",
                              os.environ.get("P2_BASE", "/content/gdrive/MyDrive/llm_planning_study")))
if not BASE.exists():
    try:
        from google.colab import drive; drive.mount("/content/gdrive")
    except Exception as e:
        print(f"mount: {e}")
EXPORT  = BASE / "hand_labeling_export"
LABELS  = BASE / "labels"
REJECTS = BASE / "rejects"
TRACES  = BASE / "traces"
OUT_DIR = EXPORT / "redraw_2026_10"
if SAMPLE == "original":
    OUT_DIR = OUT_DIR / "original_rerun_clean_prefix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if RERUN_MODELS: MODELS = [m for m in MODELS if m in RERUN_MODELS]

EV_REV   = OUT_DIR / "redraw_revision_events.json"
EV_FIL   = OUT_DIR / "redraw_filler_events.json"
ROLL     = {("lexical", arm): OUT_DIR / f"redraw_lexical_{arm}_rollouts.json"
            for arm in ("revision", "filler")}
ROLL.update({("steering", arm): OUT_DIR / f"redraw_steering_{arm}_rollouts_seed4000.json"
             for arm in ("revision", "filler")})
# the first steering re-run (Nov 1) used 1000 + r by mistake; kept, and reported
# by `analyze` as a second, independent seed set
ROLL.update({("steering_seed1000", arm): OUT_DIR / f"redraw_steering_{arm}_rollouts.json"
             for arm in ("revision", "filler")})
# Llama-8B steering with the direction rebuilt from the re-extracted windows
ROLL.update({("steering_llama8b_reextract", arm):
             OUT_DIR / f"redraw_steering_{arm}_rollouts_seed4000_llama8b_reextract.json"
             for arm in ("revision", "filler")})
# what `analyze` scores as the corrected steering result: Qwen models from the
# seed-4000 files + Llama-8B from the re-extract files (merged copy, rebuilt
# by every `analyze`)
ROLL.update({("steering_corrected", arm):
             OUT_DIR / f"redraw_steering_{arm}_rollouts_seed4000_CORRECTED_merged.json"
             for arm in ("revision", "filler")})
LLAMA = "DeepSeek-R1-Distill-Llama-8B"
MANIFEST = OUT_DIR / "manifest.json"
RESULTS  = OUT_DIR / "redraw_results.json"

OLD = {"lexical":  {"n": "81/80", "rev": 0.194, "fil": 0.095, "diff": 0.099,
                    "p": 0.0036, "ci": (0.043, 0.159), "rep": 0.112},
       "steering": {"n": "83/80", "rev": 0.276, "fil": 0.175, "diff": 0.101,
                    "p": 0.022, "ci": (0.022, 0.190), "rep": 0.118}}

def hdr(t): print(f"\n{'='*78}\n{t}\n{'='*78}", flush=True)
def rjson(p, default=None):
    try: return json.load(open(p))
    except Exception: return default
def wjson(obj, p):
    tmp = Path(str(p) + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)
def now(): return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# manifest — one record per stage run
# ---------------------------------------------------------------------------
def env_info():
    info = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "numpy", "scipy", "sklearn"):
        try:
            m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "none")
        info["cuda"] = torch.version.cuda
    except Exception:
        info["gpu"] = None
    return info

def manifest_add(rec):
    m = rjson(MANIFEST, {"stages": []})
    m["stages"].append(rec)
    wjson(m, MANIFEST)


# ---------------------------------------------------------------------------
# EMBEDDED PRODUCTION CODE (Oct 29). Nothing used to GENERATE or SCORE
# rollouts is searched for on Drive any more: several debugging versions of
# the necessity notebook exist there, and the first lexical re-run picked the
# 27-phrase pilot. These are verbatim copies:
#   lexical  : superseded/necessity_lexical_revision_arm.py (38 phrases,
#              boundary prefixes; same list as the filler-arm notebook)
#   steering : the steering generator used for the published runs
#              (generate_with_steering, alpha table CHOSEN_ALPHA_MULTIPLIER)
#   leak     : is_leaked + EXPLICITLY_BANNED from
#              superseded/necessity_steering_both_arms.ipynb (the published
#              leak-filter sensitivity)
#   extractor: boxed_spans / normalise_answer / extract_answer_new / STOPWORDS
#              as used for the paper's corrected necessity numbers (identical
#              to superseded/necessity_verification_and_leak_audit.ipynb §1)
#   scorer   : load_gt / necessity_scores, verbatim from that notebook's §1
# ---------------------------------------------------------------------------
REFS = {'lexical': {'N_ROLLOUTS': 'N_ROLLOUTS = 8\n', 'ROLLOUT_MAX_NEW_TOKENS': 'ROLLOUT_MAX_NEW_TOKENS = 2048   # smaller than the original 8192 -- continuing\n', 'SAMPLE_TEMPERATURE': 'SAMPLE_TEMPERATURE = 0.6\n', 'SAMPLE_TOP_P': 'SAMPLE_TOP_P = 0.95\n', 'build_prompt': 'def build_prompt(problem: dict, tokenizer, model_name: str) -> str:\n    is_reasoning = any(tag in model_name for tag in ["R1", "QwQ"])\n\n    if problem["difficulty"] == "hard":\n        instruction = (\n            "Solve the following competition mathematics problem carefully. "\n            "Show your reasoning step by step and verify your answer."\n        )\n    else:\n        instruction = (\n            "Solve the following mathematics competition problem. "\n            "Show your working step by step."\n        )\n\n    if is_reasoning:\n        messages = [{"role": "user",\n                     "content": f"{instruction}\\n\\nProblem: {problem[\'problem\']}"}]\n    else:\n        messages = [\n            {"role": "system",\n             "content": "You are an expert mathematician. Solve problems with "\n                        "clear step-by-step reasoning."},\n            {"role": "user",\n             "content": f"{instruction}\\n\\nProblem: {problem[\'problem\']}"}\n        ]\n\n    return tokenizer.apply_chat_template(\n        messages, tokenize=False, add_generation_prompt=True\n    )\n', 'SUPPRESSED_PHRASES': 'SUPPRESSED_PHRASES = [\n    "wait", "Wait", "actually", "Actually", "let me reconsider",\n    "that\'s wrong", "that is wrong", "i made an error", "I made an error",\n    "let me try again", "this doesn\'t work", "this does not work",\n    "i need to reconsider", "I need to reconsider", "hmm", "Hmm",\n    "that\'s not right", "that is not right", "let me rethink",\n    "on second thought", "let me go back", "wait, let me", "Wait, let me",\n    "actually, let me", "Actually, let me", "let me recheck", "let me re-check",\n    # Added after auditing suppression leaks -- these are confirmed real\n    # matched_text values seen in this project\'s actual labels/*.json data\n    # (e.g. during the type_hint vs verdict investigation) that REVISION_RE\n    # from Task_1_Data_Pipeline.ipynb did NOT cover -- that regex is\n    # apparently incomplete relative to whatever actually generated the real\n    # marker detections. If more gaps turn up via audit_suppression_leaks.py,\n    # add them here too rather than treating this list as final.\n    "alternatively", "Alternatively", "I made a mistake", "i made a mistake",\n    "I think I made a mistake", "let me double check", "let me double-check",\n    "Let me double check", "Let me double-check",\n    # Added after reading real rollout transcripts (Step 2.5) -- "hold on"\n    # leaked repeatedly and was never on any version of this list.\n    "hold on", "Hold on",\n]\n', 'build_bad_words_ids': 'def build_bad_words_ids(tokenizer):\n    """\n    Builds token-id sequences to ban via generate()\'s bad_words_ids. Tries each\n    phrase with several boundary-context prefixes, since byte-level BPE\n    tokenizers can merge preceding whitespace/punctuation into the SAME token\n    as the following word -- e.g. "\\\\n\\\\nAlternatively" may be encoded as a\n    compound token distinct from both "Alternatively" and " Alternatively".\n    Missing a merge pattern means the ban silently never engages for that\n    specific boundary -- this cannot be assumed complete, hence the\n    verify-after-generation check in generate_suppressed_rollout below.\n    """\n    bad_words_ids = []\n    seen = set()\n    boundary_prefixes = ["", " ", "\\n", "\\n\\n", ". ", ", ", ": "]\n    for phrase in SUPPRESSED_PHRASES:\n        for boundary in boundary_prefixes:\n            variant = boundary + phrase\n            ids = tuple(tokenizer.encode(variant, add_special_tokens=False))\n            if ids and ids not in seen:\n                seen.add(ids)\n                bad_words_ids.append(list(ids))\n    return bad_words_ids\n', 'load_model_and_tokenizer': 'def load_model_and_tokenizer(model_name: str):\n    print(f"Loading tokenizer: {model_name}")\n    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n\n    print(f"Loading model (fp16)...")\n    model = AutoModelForCausalLM.from_pretrained(\n        model_name, torch_dtype=torch.float16, device_map="auto",\n        trust_remote_code=True,\n    )\n    model.eval()\n    print("Model loaded.")\n    return model, tokenizer\n', 'generate_suppressed_rollout': '@torch.no_grad()\ndef generate_suppressed_rollout(model, tokenizer, prefix_text: str, bad_words_ids, seed: int):\n    torch.manual_seed(seed)\n    inputs = tokenizer(prefix_text, return_tensors="pt", truncation=False).to(model.device)\n    output_ids = model.generate(\n        **inputs,\n        max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,\n        do_sample=True,\n        temperature=SAMPLE_TEMPERATURE,\n        top_p=SAMPLE_TOP_P,\n        bad_words_ids=bad_words_ids,\n        pad_token_id=tokenizer.pad_token_id,\n        eos_token_id=tokenizer.eos_token_id,\n    )\n    prompt_len = inputs["input_ids"].shape[1]\n    new_tokens = output_ids[0][prompt_len:]\n    full_response = tokenizer.decode(new_tokens, skip_special_tokens=False)\n    return full_response\n'}, 'steering': {'N_ROLLOUTS': 'N_ROLLOUTS = 8\n', 'ROLLOUT_MAX_NEW_TOKENS': 'ROLLOUT_MAX_NEW_TOKENS = 2048\n', 'CHOSEN_ALPHA_MULTIPLIER': 'CHOSEN_ALPHA_MULTIPLIER = {\n    # Llama-8B: "wait" stayed at 100% across the ENTIRE 0.25-6.0 sweep -- not\n    # moving this word is a known, accepted limitation here, not something\n    # calibration failed to find. 1.5 chosen because coherence is solid (75%,\n    # same as 2.0) and other words (actually, alternatively, verify) showed\n    # real, consistent reduction at this strength -- no benefit to pushing\n    # higher for a word that demonstrably won\'t move.\n    "DeepSeek-R1-Distill-Llama-8B": 1.5,\n\n    # Qwen-1.5B: coherence collapses hard at 4.0+ (extraction 25%, 25%, 0%).\n    # 3.0 is the highest usable point AND the only alpha showing any flagged-\n    # rate reduction (100%->75%) with full coherence (100% extraction) --\n    # the clear best point in the safe range.\n    "DeepSeek-R1-Distill-Qwen-1.5B": 3.0,\n\n    # Qwen-7B: "wait" stayed at 100% across the entire sweep, same accepted\n    # limitation as Llama-8B. 1.5 chosen for the same reason -- coherence\n    # (50% extraction) is stable across 1.0-3.0, no gain from pushing further.\n    "DeepSeek-R1-Distill-Qwen-7B": 1.5,\n\n    # Qwen-14B: the one clean, unconfounded result -- "wait" genuinely drops\n    # 100%->50% at alpha=5.0 with coherence FULLY intact (100% extraction,\n    # matching the alpha=0 baseline\'s own ceiling). No reason to prefer 6.0\n    # over 5.0 since both give identical extraction/flagged rates.\n    "DeepSeek-R1-Distill-Qwen-14B": 5.0,\n}\n', 'build_prompt': 'def build_prompt(problem: dict, tokenizer, model_name: str) -> str:\n    is_reasoning = any(tag in model_name for tag in ["R1", "QwQ"])\n    if problem["difficulty"] == "hard":\n        instruction = ("Solve the following competition mathematics problem carefully. "\n                        "Show your reasoning step by step and verify your answer.")\n    else:\n        instruction = ("Solve the following mathematics competition problem. "\n                        "Show your working step by step.")\n    if is_reasoning:\n        messages = [{"role": "user", "content": f"{instruction}\\n\\nProblem: {problem[\'problem\']}"}]\n    else:\n        messages = [\n            {"role": "system", "content": "You are an expert mathematician. Solve problems with "\n                                          "clear step-by-step reasoning."},\n            {"role": "user", "content": f"{instruction}\\n\\nProblem: {problem[\'problem\']}"}\n        ]\n    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n', 'get_decoder_layers': 'def get_decoder_layers(model):\n    assert hasattr(model, "model") and hasattr(model.model, "layers"), (\n        "Expected model.model.layers (standard HF Llama/Qwen2 layout) -- this model\'s "\n        "architecture doesn\'t match. Inspect model structure manually before proceeding."\n    )\n    return model.model.layers\n', 'make_steering_hook': 'def make_steering_hook(direction_tensor, alpha):\n    def hook(module, inputs, output):\n        if isinstance(output, tuple):\n            hidden_states = output[0] - alpha * direction_tensor\n            return (hidden_states,) + output[1:]\n        return output - alpha * direction_tensor\n    return hook\n', 'generate_with_steering': '@torch.no_grad()\ndef generate_with_steering(model, tokenizer, prefix_text, layer_idx, direction_tensor, alpha, seed):\n    torch.manual_seed(seed)\n    inputs = tokenizer(prefix_text, return_tensors="pt", truncation=False).to(model.device)\n\n    handle = None\n    if alpha != 0.0:\n        layers = get_decoder_layers(model)\n        handle = layers[layer_idx].register_forward_hook(make_steering_hook(direction_tensor, alpha))\n\n    try:\n        output_ids = model.generate(\n            **inputs, max_new_tokens=ROLLOUT_MAX_NEW_TOKENS, do_sample=True,\n            temperature=0.6, top_p=0.95,\n            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,\n        )\n    finally:\n        if handle is not None:\n            handle.remove()  # always remove, even if generation raises -- verified in testing\n\n    prompt_len = inputs["input_ids"].shape[1]\n    new_tokens = output_ids[0][prompt_len:]\n    return tokenizer.decode(new_tokens, skip_special_tokens=False)\n'}}
REFS["leak"] = {"EXPLICITLY_BANNED": 'EXPLICITLY_BANNED = [\n    "wait", "actually", "let me reconsider", "that\'s wrong", "that is wrong",\n    "i made an error", "let me try again", "this doesn\'t work",\n    "this does not work", "i need to reconsider", "hmm", "that\'s not right",\n    "that is not right", "let me rethink", "on second thought",\n    "let me go back", "wait, let me", "actually, let me", "let me recheck",\n    "let me re-check", "alternatively", "i made a mistake",\n    "i think i made a mistake", "let me double check", "let me double-check",\n    "hold on",\n]', "is_leaked": 'def is_leaked(text):\n    t = text.lower()\n    return any(p in t for p in EXPLICITLY_BANNED)'}
REFS["extractor"] = {'boxed_spans': 'def boxed_spans(text):\n    spans = []\n    for m in re.finditer(r\'\\\\boxed\\s*\\{\', text or ""):\n        i = m.end() - 1\n        depth = 0\n        for j in range(i, len(text)):\n            if text[j] == \'{\':\n                depth += 1\n            elif text[j] == \'}\':\n                depth -= 1\n                if depth == 0:\n                    spans.append(text[i + 1:j])\n                    break\n    return spans', 'normalise_answer': 'def normalise_answer(s):\n    if s is None:\n        return "UNKNOWN"\n    s = s.strip()\n    s = s.replace("\\u0120", " ").replace("\\u010a", " ")\n    s = re.sub(r\'^\\$+|\\$+$\', \'\', s)\n    s = re.sub(r\'\\\\(?:left|right|,|;|!|quad|qquad)\', \'\', s)\n    s = re.sub(r\'\\\\text\\s*\\{([^}]*)\\}\', r\'\\1\', s)\n    s = s.replace(r\'\\dfrac\', r\'\\frac\').replace(r\'\\tfrac\', r\'\\frac\')\n    s = re.sub(r\'\\\\frac\\s*\\{([^{}]*)\\}\\s*\\{([^{}]*)\\}\', r\'\\1/\\2\', s)\n    s = re.sub(r\'\\s+\', \'\', s)\n    s = s.rstrip(\'.,;:\')\n    s = re.sub(r\'^\\{|\\}$\', \'\', s)\n    return s or "UNKNOWN"', 'extract_answer_new': 'def extract_answer_new(text):\n    boxed = boxed_spans(text)\n    if boxed:\n        return normalise_answer(boxed[-1])\n    for pat in (r\'(?:final\\s+)?answer\\s*(?:is|=|:)\\s*(.{0,40})\',\n                r\'(?:therefore|thus|hence)\\s*,?\\s*(.{0,40})\'):\n        for m in re.finditer(pat, text, re.IGNORECASE):\n            for tok in re.findall(r\'[-+]?\\$?\\\\?[0-9A-Za-z\\\\{}\\^_/\\.,]+\', m.group(1)):\n                cand = normalise_answer(tok)\n                if cand == "UNKNOWN" or cand.lower() in STOPWORDS:\n                    continue\n                if re.search(r\'[0-9]|\\\\\', cand):\n                    return cand\n    return "UNKNOWN"', 'STOPWORDS': 'STOPWORDS = {"the", "a", "an", "is", "approximately", "about", "roughly",\n             "exactly", "correct", "incorrect", "therefore", "thus", "so",\n             "then", "that", "this", "it", "equal", "equals", "our", "final",\n             "answer", "value", "result", "less", "more", "different", "same",\n             "indeed"}'}
ORDER = {
    "lexical":  ["N_ROLLOUTS", "ROLLOUT_MAX_NEW_TOKENS", "SAMPLE_TEMPERATURE",
                 "SAMPLE_TOP_P", "SUPPRESSED_PHRASES", "build_prompt",
                 "build_bad_words_ids", "load_model_and_tokenizer",
                 "generate_suppressed_rollout"],
    "steering": ["N_ROLLOUTS", "ROLLOUT_MAX_NEW_TOKENS", "CHOSEN_ALPHA_MULTIPLIER",
                 "build_prompt", "get_decoder_layers", "make_steering_hook",
                 "generate_with_steering"],
    "leak":     ["EXPLICITLY_BANNED", "is_leaked"],
    "extractor": ["STOPWORDS", "boxed_spans", "normalise_answer", "extract_answer_new"],
}
HEADER = ("import re, json, time\nimport torch\n"
          "from transformers import AutoTokenizer, AutoModelForCausalLM\n")

def embedded_source(kind):
    return "\n\n".join(REFS[kind][k] for k in ORDER[kind])

def fingerprint(kind):
    return hashlib.sha256(embedded_source(kind).encode()).hexdigest()[:16]

def build_module(kind):
    src = (HEADER if kind in ("lexical", "steering") else "import re\n") + embedded_source(kind)
    m = types.ModuleType(f"embedded_{kind}")
    exec(compile(src, f"<embedded {kind}>", "exec"), m.__dict__)
    fp = fingerprint(kind)
    print(f"  {kind}: embedded production code (fingerprint {fp})")
    return m, {"label": kind, "file": "embedded in redraw_necessity.py", "sha256_16": fp}

def _top_nodes(txt):
    out = {}
    for n in ast.parse(txt).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n
    return out

REFS["scorer"] = {"load_gt": 'def load_gt(use_repaired):\n    """Ground truth. If use_repaired, prefer the dataset-derived answers from\n    the audit over the Stage-1 values, which were truncated for 71 problems."""\n    gt = {}\n    for p in (EVENTS_REV, EVENTS_FIL):\n        if p.exists():\n            for e in json.load(open(p)):\n                gt[(e["model"], e["problem_id"])] = e["answer_gt"]\n    n_repaired = 0\n    if use_repaired and REPAIRED_GT.exists():\n        rep = json.load(open(REPAIRED_GT))\n        for k, v in rep.items():\n            model, pid = k.split("|", 1)\n            if (model, pid) in gt and v.get("true_answer_gt"):\n                gt[(model, pid)] = v["true_answer_gt"]\n                n_repaired += 1\n    return gt, n_repaired', "necessity_scores": 'def necessity_scores(path, gt, parser, unknown_mode):\n    """unknown_mode: \'as_incorrect\' | \'dropped\' | \'as_correct\'."""\n    out = []\n    for r in json.load(open(path)):\n        g = gt.get((r["model"], r["problem_id"]))\n        texts = r.get("rollout_texts", [])\n        if g is None or not texts:\n            continue\n        vals = []\n        for t in texts:\n            a = parser(t)\n            unk = (a == "UNKNOWN")\n            if unk and unknown_mode == "dropped":\n                continue\n            if unk:\n                correct = (unknown_mode == "as_correct")\n            else:\n                correct = (normalise_answer(a) == normalise_answer(g))\n            vals.append(correct)\n        if vals:\n            out.append({"model": r["model"], "problem_id": r["problem_id"],\n                        "human_label": r.get("human_label"),\n                        "necessity": 1.0 - sum(vals) / len(vals)})\n    return out'}
ORDER["scorer"] = ["load_gt", "necessity_scores"]
EVENTS_REV_OLD = EXPORT / "counterfactual_pilot_events.json"
EVENTS_FIL_OLD = EXPORT / "filler_control_pilot_events.json"
REPAIRED_GT = EXPORT / "repaired_ground_truth.json"

def lift_scorer():
    """The corrected scorer, all embedded: extractor (see REFS['extractor'])
    plus load_gt / necessity_scores verbatim from
    superseded/necessity_verification_and_leak_audit.ipynb, section 1
    (= notebook 'Verification and audits', section 1). load_gt reads the
    answer_gt of the published 83/80 events and overrides it with the
    dataset-derived answer in repaired_ground_truth.json."""
    m, _ = build_module("extractor")
    m.json = json
    m.EVENTS_REV, m.EVENTS_FIL, m.REPAIRED_GT = EVENTS_REV_OLD, EVENTS_FIL_OLD, REPAIRED_GT
    exec(compile(embedded_source("scorer"), "<embedded scorer>", "exec"), m.__dict__)
    gt, n_rep = m.load_gt(use_repaired=True)
    print(f"  scorer: embedded (extractor {fingerprint('extractor')}, "
          f"load_gt/necessity_scores {fingerprint('scorer')})")
    print(f"    load_gt(use_repaired=True) -> {len(gt)} entries, "
          f"{len({k[1] for k in gt})} problems, {n_rep} repaired")
    return m, gt, {"label": "scorer", "file": "embedded",
                   "extractor_fingerprint": fingerprint("extractor"),
                   "scorer_fingerprint": fingerprint("scorer")}

def load_gt_all_traces():
    """Ground truth for EVERY trace (redraw sample only): the dataset-derived
    answer from repaired_ground_truth.json, falling back to the stored one."""
    rep = rjson(REPAIRED_GT, {}) or {}
    gt = {}
    for k, v in rep.items():
        model, pid = k.split("|", 1)
        a = v.get("true_answer_gt") or v.get("stored_answer_gt")
        if a: gt[(model, pid)] = a
    print(f"    all-trace ground truth: {len(gt)} (model, problem) entries")
    return gt


# ---------------------------------------------------------------------------
# text helpers (verbatim from the 'Check 8' repetition analysis,
# superseded/audits_during_revision.ipynb §16)
# ---------------------------------------------------------------------------
_TOK = re.compile(r"[a-z0-9]+")
def toks(s): return set(_TOK.findall(s.lower()))

def max_jaccard(before, after):
    bs = [toks(s) for s in before]; as_ = [toks(s) for s in after]
    best = 0.0
    for b in bs:
        if not b: continue
        for a in as_:
            if not a: continue
            i = len(b & a)
            if i:
                j = i / len(b | a)
                if j > best: best = j
    return float(best)

_SENT = re.compile(r'(?<=[.!?])\s+')
def split_sentences(t): return [s for s in _SENT.split(t) if s.strip()]

def take_by_budget(sents, budget, from_end):
    out, used = [], 0
    for s in (reversed(sents) if from_end else sents):
        if used + len(s) > budget and out: break
        out.append(s); used += len(s)
    return list(reversed(out)) if from_end else out

def extract_window(fr, cs, ce, BB=900, BA=700, RB=2500, RA=1200, NB=6, NA=6):
    rs = max(0, cs - RB); re_ = min(len(fr), ce + RA)
    w = fr[rs:re_]
    ls, le = cs - rs, ce - rs
    if not (0 <= ls <= le <= len(w)): return None, None
    bef = take_by_budget(split_sentences(w[:ls]), BB, True)
    aft = take_by_budget(split_sentences(w[le:]), BA, False)
    if aft and not re.search(r'[.!?]\s*$', aft[-1]): aft = aft[:-1]
    bef, aft = bef[-NB:], aft[:NA]
    if not bef or not aft: return None, None
    return bef, aft

# length-preserving; used ONLY for offset verification and windows, exactly as
# Check 8 did, so the repetition covariate is comparable with the old one
_LP = {"Ġ": " ", "Ċ": "\n", "Ģ": "\n", "▁": " "}
def clean_lp(t):
    s = t or ""
    for k, v in _LP.items(): s = s.replace(k, v)
    return s

# proper byte-level-BPE decoding (GPT-2 bytes_to_unicode inverse), used for
# the PREFIX the model is conditioned on. Not length-preserving, so it is
# applied to a slice that has already been cut at the verified offset.
def _byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
BYTE_DEC = _byte_decoder()

def restore_text(s):
    """-> (text, path). 'clean': no BPE markers. 'bpe_decoded': the string was
    entirely in byte-level character space (no real spaces) and was decoded
    exactly. 'mixed_minimal': markers alongside real spaces; only Ġ/Ċ/ĉ are
    mapped back, and the event is flagged."""
    if "Ġ" not in s and "Ċ" not in s and "ĉ" not in s:
        return s, "clean"
    if " " not in s:
        try:
            return (bytes(BYTE_DEC[ch] for ch in s)
                    .decode("utf-8", errors="replace"), "bpe_decoded")
        except KeyError:
            pass
    return (s.replace("Ġ", " ").replace("Ċ", "\n")
             .replace("ĉ", "\t"), "mixed_minimal")

def marker_at(full, cs, marker):
    """Check 8's verification: the marker must sit at cs in the trace."""
    mk = clean_lp(str(marker or "")).strip()
    txt = clean_lp(full or "")
    if cs is None or not mk: return False
    try: cs = int(cs)
    except Exception: return False
    return 0 <= cs < len(txt) and txt[cs:cs + len(mk)].lower() == mk.lower()

def rep_at(full, cs, ce, marker):
    txt = clean_lp(full or "")
    mk = clean_lp(str(marker or "")).strip()
    if ce is None: ce = int(cs) + max(len(mk), 1)
    b, a = extract_window(txt, int(cs), int(ce))
    return None if b is None else max_jaccard(b, a)


# ---------------------------------------------------------------------------
# estimators (verbatim from the same §16)
# ---------------------------------------------------------------------------
def ls_fit(Y, X):
    b, *_ = np.linalg.lstsq(X, Y, rcond=None); return b

def cluster_boot(Y, X, groups, nb=N_BOOT, seed=8080):
    beta = ls_fit(Y, X); rng = np.random.RandomState(seed)
    gi = defaultdict(list)
    for i, g in enumerate(groups): gi[g].append(i)
    arrs = [np.array(v) for v in gi.values()]; G = len(arrs)
    D = np.empty((nb, X.shape[1]))
    for k in range(nb):
        idx = np.concatenate([arrs[j] for j in rng.randint(0, G, G)])
        D[k] = ls_fit(Y[idx], X[idx])
    return beta, D, G

def ci_p(col):
    lo, hi = np.percentile(col, [2.5, 97.5])
    p = 2 * min((col <= 0).mean(), (col >= 0).mean())
    return float(lo), float(hi), float(min(1.0, max(p, 1.0 / len(col))))

def boot_mean_diff(a, b, ga, gb, nb=N_BOOT, seed=8080):
    rng = np.random.RandomState(seed)
    by = defaultdict(lambda: ([], []))
    for v, g in zip(a, ga): by[g][0].append(v)
    for v, g in zip(b, gb): by[g][1].append(v)
    keys = list(by); G = len(keys); out = []
    for _ in range(nb):
        pick = [keys[j] for j in rng.randint(0, G, G)]
        A = [v for k in pick for v in by[k][0]]
        B = [v for k in pick for v in by[k][1]]
        if A and B: out.append(np.mean(A) - np.mean(B))
    out = np.array(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), int(G)

def pfmt(p, nb=N_BOOT):
    return f"< {1/nb:.2g} (floor)" if p <= 1/nb + 1e-12 else f"{p:.4g}"


# ---------------------------------------------------------------------------
# helpers shared by the stages
# ---------------------------------------------------------------------------
def event_key(e):
    return (e["arm"], e["model"], str(e["problem_id"]), int(e["char_start"]))

_tr = {}
def trace(model, pid):
    k = (model, str(pid))
    if k not in _tr:
        _tr[k] = rjson(TRACES / model / f"{pid}.json")
    return _tr[k]


# #############################################################################
# STAGE: select
# #############################################################################
def stage_select():
    t0 = time.time()
    hdr("SELECT — building the redrawn event sets")
    prod, _gt_events, src_scorer = lift_scorer()
    gt = load_gt_all_traces()
    # the filler-arm notebook's normaliser, verbatim
    def normalize_marker(text):
        return re.sub(r"[^\w\s]", "", str(text).lower()).strip()
    src_norm = {"label": "marker normaliser", "file": "embedded"}
    print(f"    normalize_marker('Wait,') = {normalize_marker('Wait,')!r}")

    # ---- 1. eligible traces, CORRECTED parser ------------------------------
    hdr("1 — eligible traces (corrected parser, not truncated)")
    elig, why_t = {}, Counter()
    parse_disagree = 0
    for model in MODELS:
        d = TRACES / model
        if not d.is_dir(): continue
        for p in sorted(d.glob("*.json")):
            pid = p.stem
            tr = rjson(p)
            if tr is None: why_t["unreadable"] += 1; continue
            if tr.get("truncated", False):
                why_t["truncated"] += 1; elig[(model, pid)] = False; continue
            g = gt.get((model, pid))
            if g is None:
                why_t["no repaired ground truth"] += 1
                elig[(model, pid)] = False; continue
            raw = tr.get("full_response", "") or ""
            text, _ = restore_text(raw)
            a = prod.extract_answer_new(text)
            a_raw = prod.extract_answer_new(raw)
            if (a == "UNKNOWN") != (a_raw == "UNKNOWN") or (
                    a != "UNKNOWN" and a_raw != "UNKNOWN" and
                    prod.normalise_answer(a) != prod.normalise_answer(a_raw)):
                parse_disagree += 1
            if a == "UNKNOWN":
                why_t["corrected parser: UNKNOWN"] += 1
                elig[(model, pid)] = False; continue
            ok = prod.normalise_answer(a) == prod.normalise_answer(g)
            why_t["correct" if ok else "incorrect"] += 1
            elig[(model, pid)] = ok
    n_elig = sum(elig.values())
    print(f"  traces scanned: {len(elig)}   eligible: {n_elig}")
    for k, v in why_t.most_common(): print(f"      {v:>6}  {k}")
    print(f"  traces where parsing raw vs restored text disagrees: "
          f"{parse_disagree}  (expect ~0)")

    # ---- 2. pools, with verified offsets -----------------------------------
    hdr("2 — revision and filler pools (offset verified)")
    rev_pool, fil_pool, why_r, why_f = [], [], Counter(), Counter()
    for model in MODELS:
        d = LABELS / model
        if not d.is_dir(): continue
        for p in sorted(d.glob("*.json")):
            o = rjson(p) or {}
            pid = str(o.get("problem_id", p.stem))
            for rv in (o.get("revisions") or []):
                if not rv.get("verified", True): continue
                why_r["verified revision"] += 1
                if not elig.get((model, pid)):
                    why_r["trace not eligible"] += 1; continue
                tr = trace(model, pid)
                mk = str(rv.get("matched_text") or rv.get("marker_phrase") or "")
                cs = rv.get("char_start")
                if not marker_at(tr.get("full_response", ""), cs, mk):
                    why_r["marker not at offset"] += 1; continue
                why_r["eligible"] += 1
                rev_pool.append({"arm": "revision", "model": model,
                                 "problem_id": pid, "char_start": int(cs),
                                 "char_end": rv.get("char_end"),
                                 "marker_phrase": mk,
                                 "marker_norm": normalize_marker(mk)})
        rp = REJECTS / f"{model}_confirmed_filler.json"
        for fl in (rjson(rp) or []):
            pid = str(fl.get("problem_id"))
            why_f["confirmed filler"] += 1
            if not elig.get((model, pid)):
                why_f["trace not eligible"] += 1; continue
            tr = trace(model, pid)
            mk = str(fl.get("matched_text") or fl.get("marker") or "")
            cs = fl.get("char_start")
            if not marker_at(tr.get("full_response", ""), cs, mk):
                why_f["marker not at offset"] += 1; continue
            why_f["eligible"] += 1
            fil_pool.append({"arm": "filler", "model": model,
                             "problem_id": pid, "char_start": int(cs),
                             "char_end": (int(cs) + len(mk)),
                             "marker_phrase": mk,
                             "marker_norm": normalize_marker(mk)})
    for lab, w in (("revision", why_r), ("filler", why_f)):
        print(f"  {lab}:")
        for k, v in w.most_common(): print(f"      {v:>6}  {k}")
    print(f"  pool sizes: {len(rev_pool)} revision / {len(fil_pool)} filler")

    # ---- 3. old-sample prefix contamination, for the record ----------------
    hdr("3 — were the OLD 83/80 prefixes contaminated with BPE markers?")
    old_rep = {}
    for nm_, p in (("old revision", EXPORT / "counterfactual_pilot_events.json"),
                   ("old filler", EXPORT / "filler_control_pilot_events.json")):
        evs = rjson(p, []) or []
        by = defaultdict(Counter)
        for e in evs:
            s = e.get("prefix_response_text") or ""
            by[e["model"]][restore_text(s)[1]] += 1
        print(f"  {nm_} ({len(evs)} events):")
        for m, c in sorted(by.items()):
            print(f"      {m[-14:]:<16}{dict(c)}")
        old_rep[nm_] = {m: dict(c) for m, c in by.items()}
    print("""
  'bpe_decoded' or 'mixed_minimal' means that event's rollouts were generated
  from a prefix containing literal BPE marker characters. Matched across arms,
  so it does not bias the revision-filler DIFFERENCE by construction, but it is
  a fidelity problem for those events and belongs in the paper if it occurred.""")

    # ---- 4. uniform draw, then 1:1 matching --------------------------------
    hdr(f"4 — uniform draw of the revision arm, 1:1 filler matching "
        f"(target {N_PAIRS} pairs)")
    rng_s = random.Random(SEEDS["select"])
    rng_m = random.Random(SEEDS["match"])
    order = rev_pool[:]; rng_s.shuffle(order)
    by_cell = defaultdict(list)
    for f in fil_pool: by_cell[(f["model"], f["marker_norm"])].append(f)
    for v in by_cell.values(): rng_m.shuffle(v)
    used, pairs, skipped = defaultdict(int), [], Counter()
    for r in order:
        if len(pairs) >= N_PAIRS: break
        cell = (r["model"], r["marker_norm"])
        if used[cell] >= len(by_cell.get(cell, [])):
            skipped[cell] += 1; continue
        f = by_cell[cell][used[cell]]; used[cell] += 1
        pairs.append((r, f))
    n_seen = sum(skipped.values()) + len(pairs)
    print(f"  revisions examined in random order: {n_seen}")
    print(f"  pairs formed: {len(pairs)}")
    print(f"  revisions skipped, no filler left in their (model, marker) cell: "
          f"{sum(skipped.values())}")
    for (m, mk), c in skipped.most_common(8):
        print(f"      {m[-14:]:<16}{mk!r:<26}{c}")
    if len(pairs) < N_PAIRS:
        print(f"  WARNING: only {len(pairs)} pairs available — the matchable "
              f"pool is the binding constraint, not N_PAIRS.")
    print("""
  DISCLOSE: the revision arm is uniform among revisions whose (model, marker)
  cell still had an unused filler. Skipped revisions are concentrated in
  markers fillers rarely carry, so the matched arm over-represents common
  markers relative to the revision population. The composition table below
  is the evidence.""")

    old_keys = set()
    for p, arm in ((EXPORT / "counterfactual_pilot_events.json", "revision"),
                   (EXPORT / "filler_control_pilot_events.json", "filler")):
        for e in rjson(p, []) or []:
            if e.get("char_start") is not None:
                old_keys.add((arm, e["model"], str(e["problem_id"]),
                              int(e["char_start"])))

    def finish(e, k):
        tr = trace(e["model"], e["problem_id"])
        full = tr.get("full_response", "") or ""
        prefix, path = restore_text(full[:e["char_start"]])
        e.update({"eid": f"{e['arm'][0].upper()}{k:04d}",
                  "problem": tr.get("problem", ""),
                  "difficulty": tr.get("difficulty", "hard"),
                  "answer_gt": str(gt.get((e["model"], e["problem_id"]))),
                  "prefix_response_text": prefix,
                  "prefix_restore_path": path,
                  "in_old_sample": event_key(e) in old_keys,
                  "repetition": rep_at(full, e["char_start"], e.get("char_end"),
                                       e["marker_phrase"])})
        return e
    revs = [finish(dict(r), k) for k, (r, _) in enumerate(pairs)]
    fils = [finish(dict(f), k) for k, (_, f) in enumerate(pairs)]
    wjson(revs, EV_REV); wjson(fils, EV_FIL)

    hdr("5 — composition of the drawn sample")
    for lab, arr in (("revision", revs), ("filler", fils)):
        print(f"  {lab}: {len(arr)} events, "
              f"{len({e['problem_id'] for e in arr})} problems, "
              f"{len({(e['model'], e['problem_id']) for e in arr})} traces")
        print(f"      by model : {dict(Counter(e['model'][-12:] for e in arr))}")
        print(f"      by marker: {dict(Counter(e['marker_norm'] for e in arr).most_common(8))}")
        print(f"      prefix   : {dict(Counter(e['prefix_restore_path'] for e in arr))}")
        print(f"      in old sample: {sum(e['in_old_sample'] for e in arr)}")
    rr = [e["repetition"] for e in revs if e["repetition"] is not None]
    rf = [e["repetition"] for e in fils if e["repetition"] is not None]
    if rr and rf:
        print(f"  repetition: revision {np.mean(rr):.4f}  filler {np.mean(rf):.4f}"
              f"  (old sample: 0.4014 / 0.4826)")
    print(f"\n  wrote {EV_REV.name}, {EV_FIL.name}")
    est = len(pairs) * 2 * 0.146   # measured: 526 s per event on an A100 (Oct 29 run)
    print(f"  estimated GPU time per intervention: ~{est:.0f} A100-hours")

    manifest_add({"stage": "select", "started": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
                  "finished": now(), "wall_clock_s": round(time.time() - t0, 1),
                  "env": env_info(), "seeds": SEEDS, "n_pairs_target": N_PAIRS,
                  "n_pairs": len(pairs), "eligible_traces": n_elig,
                  "trace_reasons": dict(why_t), "revision_reasons": dict(why_r),
                  "filler_reasons": dict(why_f),
                  "pool": {"revision": len(rev_pool), "filler": len(fil_pool)},
                  "skipped_unmatchable": sum(skipped.values()),
                  "old_prefix_contamination": old_rep,
                  "parse_raw_vs_restored_disagree": parse_disagree,
                  "sources": [src_scorer, src_norm]})



# #############################################################################
# STAGE: select, SAMPLE = "original"
#   The published 83 revision + 80 filler events, unchanged in identity
#   (model, problem_id, char_start, marker, label), rebuilt with:
#     - prefix = restore_text(full_response[:char_start])  (no BPE characters)
#     - the corrected parser's repaired ground truth
#     - event-keyed resume (fixes the two lexical events never generated)
#   Nothing about WHICH events are used changes.
# #############################################################################
def _short(m): return str(m).split("/")[-1]

def stage_select_original():
    t0 = time.time()
    hdr("SELECT (original sample) — the published 83/80, clean prefixes")
    prod, gt, src_scorer = lift_scorer()
    out, why = {"revision": [], "filler": []}, Counter()
    for arm, path in (("revision", EXPORT / "counterfactual_pilot_events.json"),
                      ("filler",   EXPORT / "filler_control_pilot_events.json")):
        evs = rjson(path, []) or []
        print(f"\n  {arm}: {len(evs)} events from {path.name}")
        for k, e in enumerate(evs):
            model = _short(e["model"])
            if model not in MODELS:
                why[f"{arm}: model not in RERUN_MODELS"] += 1; continue
            pid, cs = str(e["problem_id"]), e.get("char_start")
            mk = str(e.get("marker_phrase") or e.get("matched_text")
                     or e.get("marker") or "")
            tr = trace(model, pid)
            if tr is None or cs is None:
                why[f"{arm}: no trace / no char_start"] += 1; continue
            full = tr.get("full_response", "") or ""
            cs = int(cs)
            old_prefix = e.get("prefix_response_text")
            same = (old_prefix == full[:cs]) if old_prefix is not None else None
            if mk:
                if not marker_at(full, cs, mk):
                    why[f"{arm}: marker not at offset"] += 1; continue
            elif not same:
                # no marker stored: the offset is verified by the stored
                # prefix being exactly the trace slice instead
                why[f"{arm}: no marker and stored prefix != trace slice"] += 1
                continue
            else:
                why[f"{arm}: verified by stored prefix (no marker field)"] += 1
            why[f"{arm}: stored prefix == full_response[:cs]: {same}"] += 1
            prefix, rpath = restore_text(full[:cs])
            why[f"{arm}: prefix {rpath}"] += 1
            g = gt.get((model, pid))
            if g is None:
                why[f"{arm}: no repaired ground truth"] += 1; continue
            ce = e.get("char_end") or (cs + len(mk))
            out[arm].append({
                "arm": arm, "eid": f"{arm[0].upper()}O{k:04d}",
                "old_eid": e.get("eid"), "round": e.get("round"),
                "human_label": e.get("human_label"),
                "model": model, "problem_id": pid, "char_start": cs,
                "char_end": ce, "marker_phrase": mk,
                "problem": tr.get("problem", ""),
                "difficulty": tr.get("difficulty", "hard"),
                "answer_gt": str(g),
                "prefix_response_text": prefix, "prefix_restore_path": rpath,
                "old_prefix_had_bpe": bool(old_prefix and ("Ġ" in old_prefix
                                                           or "Ċ" in old_prefix)),
                "in_old_sample": True,
                "repetition": rep_at(full, cs, ce, mk)})
    for k, v in sorted(why.items()): print(f"      {v:>5}  {k}")
    for arm in ("revision", "filler"):
        a = out[arm]
        print(f"\n  {arm}: {len(a)} events kept  by model "
              f"{dict(Counter(e['model'][-12:] for e in a))}")
        print(f"      prefix paths: {dict(Counter(e['prefix_restore_path'] for e in a))}"
              f"   old prefix had BPE: {sum(e['old_prefix_had_bpe'] for e in a)}")
    ex = next((e for e in out["revision"] if e["prefix_restore_path"] != "clean"), None)
    if ex:
        print(f"\n  example restored tail ({ex['model'][-12:]}): "
              f"{ex['prefix_response_text'][-80:]!r}")
    # same model, same input => the restored prefix must re-tokenize to the
    # ids the model originally generated; checked here on the cheap (tokenizer
    # only, no weights)
    try:
        from transformers import AutoTokenizer
        for model in MODELS:
            sub = [e for e in out["revision"] + out["filler"] if e["model"] == model]
            if not sub: continue
            tok = AutoTokenizer.from_pretrained(f"deepseek-ai/{model}")
            bad = sum(1 for e in sub if any(c in e["prefix_response_text"]
                                            for c in "ĠĊ"))
            n_tok = [len(tok(e["prefix_response_text"], add_special_tokens=False)
                         ["input_ids"]) for e in sub[:5]]
            print(f"  {model[-14:]:<16}restored prefixes still containing Ġ/Ċ: {bad}"
                  f"   token lengths of first 5 prefixes: {n_tok}")
    except Exception as ex_:
        print(f"  (tokenizer check skipped: {type(ex_).__name__}: {ex_})")
    wjson(out["revision"], EV_REV); wjson(out["filler"], EV_FIL)
    n = len(out["revision"]) + len(out["filler"])
    print(f"\n  wrote {EV_REV} and {EV_FIL.name}")
    print(f"  {n} events x 8 rollouts per intervention: ~{n*0.146:.0f} A100-h "
          f"per intervention at the measured 0.146 A100-h/event.")
    manifest_add({"stage": "select_original",
                  "started": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
                  "finished": now(), "wall_clock_s": round(time.time() - t0, 1),
                  "env": env_info(), "seeds": SEEDS, "models": MODELS,
                  "reasons": dict(why),
                  "n": {a: len(out[a]) for a in out}, "sources": [src_scorer]})

# #############################################################################
# STAGES: lexical / steering
# #############################################################################
def stage_generate(intervention):
    t0 = time.time()
    hdr(f"GENERATE — {intervention}")
    revs, fils = rjson(EV_REV), rjson(EV_FIL)
    if not revs or not fils:
        raise SystemExit("run STAGE = 'select' first")

    if intervention == "lexical":
        gen, src_gen = build_module("lexical")
        print(f"    SUPPRESSED_PHRASES: {len(gen.SUPPRESSED_PHRASES)}")
        loader = gen.load_model_and_tokenizer
        sources = [src_gen]
    else:
        gen, src_gen = build_module("steering")
        lex, src_lex = build_module("lexical")      # model loader only
        loader = lex.load_model_and_tokenizer
        dirs_file = (EXPORT / "steering_directions_llama8b_reextract.json"
                     if intervention == "steering_llama8b_reextract" else EXPORT / "steering_directions.json")
        if not dirs_file.exists():
            raise SystemExit(f"{dirs_file.name} not found (run 11_reextract_llama8b.py stage 'direction' first)")
        print(f"    directions: {dirs_file.name}")
        dirs = rjson(dirs_file) or {}
        dirs = dirs.get("models", dirs) if isinstance(dirs, dict) else {}
        alpha_k = gen.CHOSEN_ALPHA_MULTIPLIER
        print(f"    alpha multipliers: {alpha_k}")
        sources = [src_gen, src_lex]
    GEN_FP = src_gen["sha256_16"]
    n_roll = int(gen.N_ROLLOUTS)
    print(f"    N_ROLLOUTS={n_roll}  max_new_tokens={gen.ROLLOUT_MAX_NEW_TOKENS}")
    lk, src_lk = build_module("leak")
    leak = lk.is_leaked; sources.append(src_lk)

    results = {}
    for arm in ("revision", "filler"):
        recs_ = rjson(ROLL[(intervention, arm)], []) or []
        want = SEEDS["rollout_base"][intervention]
        stale = [r for r in recs_ if r.get("generator_fingerprint") != GEN_FP
                 or (r.get("seeds") or [want])[0] != want]
        if stale:
            arch = Path(str(ROLL[(intervention, arm)]).replace(
                ".json", f"_ARCHIVED_not_production_{datetime.now():%Y%m%d_%H%M%S}.json"))
            wjson(recs_, arch)
            print(f"  {arm}: {len(stale)} of {len(recs_)} existing records were made by a "
                  f"different generator or seed set -> whole file archived to {arch.name}; starting fresh")
            recs_ = []
        results[arm] = recs_
    done = {(r["arm"], r["model"], str(r["problem_id"]), int(r["char_start"]))
            for arm in results for r in results[arm]}
    print(f"  rollout seeds: {SEEDS['rollout_base'][intervention]} + r   -> {ROLL[(intervention, 'revision')].name}")
    print(f"  resuming: {len(done)} events already complete (keyed on the event)")

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    per_model_time = {}
    for model in ([LLAMA] if intervention == "steering_llama8b_reextract" else MODELS):
        todo = [e for e in revs + fils
                if e["model"] == model and event_key(e) not in done]
        if not todo:
            print(f"\n  {model}: nothing to do"); continue
        tm = time.time()
        full = model if "/" in model else f"deepseek-ai/{model}"
        print(f"\n  {model}: {len(todo)} events", flush=True)
        mdl, tok = loader(full)
        if intervention == "lexical":
            bad = gen.build_bad_words_ids(tok)
            print(f"    {len(bad)} banned token sequences (the pilot ban gave 54)")
        else:
            info = dirs.get(model)
            if not info:
                print(f"    no steering direction for {model}; skipped"); continue
            layer = int(info.get("layer", info.get("layer_idx")))
            alpha = float(alpha_k[model]) * float(info["raw_direction_norm"])
            direction = torch.tensor(info["unit_direction"], dtype=torch.float16,
                                     device=mdl.device)
            print(f"    layer {layer}  k={alpha_k[model]}  alpha={alpha:.3f}")
        for e in todo:
            te = time.time()
            prefix = gen.build_prompt({"problem": e["problem"],
                                       "difficulty": e["difficulty"]},
                                      tok, full) + e["prefix_response_text"]
            texts = []
            for r in range(n_roll):
                seed = SEEDS["rollout_base"][intervention] + r
                if intervention == "lexical":
                    t = gen.generate_suppressed_rollout(mdl, tok, prefix, bad, seed)
                else:
                    t = gen.generate_with_steering(mdl, tok, prefix, layer,
                                                   direction, alpha, seed)
                texts.append(t)
            rec = {"arm": e["arm"], "eid": e["eid"], "model": model,
                   "problem_id": e["problem_id"], "char_start": e["char_start"],
                   "marker_phrase": e["marker_phrase"],
                   "rollout_texts": texts,
                   "seeds": [SEEDS["rollout_base"][intervention] + r for r in range(n_roll)],
                   "leaked": ([bool(leak(t)) for t in texts] if leak else None),
                   "generator_fingerprint": GEN_FP,
                   "n_banned_sequences": (len(bad) if intervention == "lexical" else None),
                   "wall_clock_s": round(time.time() - te, 1), "gpu": gpu}
            results[e["arm"]].append(rec)
            wjson(results[e["arm"]], ROLL[(intervention, e["arm"])])
            print(f"    {e['eid']} {e['problem_id']:<12} "
                  f"{rec['wall_clock_s']:>6.0f}s", flush=True)
        per_model_time[model] = round(time.time() - tm, 1)
        del mdl
        torch.cuda.empty_cache()

    manifest_add({"stage": intervention,
                  "started": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
                  "finished": now(), "wall_clock_s": round(time.time() - t0, 1),
                  "per_model_wall_clock_s": per_model_time,
                  "env": env_info(), "seeds": SEEDS, "n_rollouts": n_roll,
                  "generator_fingerprint": GEN_FP, "sources": sources})
    print(f"\n  done. {intervention} wall clock {time.time()-t0:.0f}s on {gpu}")


# #############################################################################
# STAGE: analyze
# #############################################################################
def stage_analyze():
    t0 = time.time()
    hdr(f"ANALYZE — {SAMPLE} sample, beside the old 83/80")
    prod, gt, src_scorer = lift_scorer()
    try:
        print("\n  necessity_scores source (first 30 lines), for the record:")
        for ln in REFS["scorer"]["necessity_scores"].splitlines()[:30]:
            print(f"    | {ln}")
    except Exception as e:
        print(f"    (source not retrievable: {e})")

    from scipy.stats import mannwhitneyu
    evs = {e["eid"]: e for e in (rjson(EV_REV, []) + rjson(EV_FIL, []))}
    out = {}
    labels = list(INTERVENTIONS)
    if "steering" in labels and ROLL[("steering_llama8b_reextract", "revision")].exists():
        for arm in ("revision", "filler"):
            q = [r for r in (rjson(ROLL[("steering", arm)], []) or []) if r["model"] != LLAMA]
            l8 = [r for r in (rjson(ROLL[("steering_llama8b_reextract", arm)], []) or []) if r["model"] == LLAMA]
            wjson(q + l8, ROLL[("steering_corrected", arm)])
        labels.insert(labels.index("steering"), "steering_corrected")
    if "steering" in labels and ROLL[("steering_seed1000", "revision")].exists():
        labels.append("steering_seed1000")
    for label in labels:
        iv = "steering" if label.startswith("steering") else label
        note = {"steering_seed1000": "  (second seed set: 1000 + r; NOT the published procedure)",
                "steering_corrected": "  (seeds 4000 + r; Llama-8B direction rebuilt from the re-extracted windows)"}
        if label == "steering" and "steering_corrected" in labels:
            note["steering"] = "  (seeds 4000 + r; Llama-8B direction from the ORIGINAL windows, superseded)"
        hdr(f"{label.upper()}" + note.get(label, f"  (seeds {SEEDS['rollout_base'][iv]} + r)"))
        recs = {arm: rjson(ROLL[(label, arm)], []) or [] for arm in ("revision", "filler")}
        fp_ = fingerprint(iv)
        for arm in recs:
            bad_ = sum(1 for r in recs[arm] if r.get("generator_fingerprint") != fp_)
            if bad_:
                print(f"  {arm}: {bad_} records NOT made by the production generator "
                      f"-> excluded; re-run STAGE = '{iv}'")
                recs[arm] = [r for r in recs[arm] if r.get("generator_fingerprint") == fp_]
        if not recs["revision"] or not recs["filler"]:
            print("  no rollouts yet"); continue

        def score(rec, mode, drop_leaked=False):
            g = gt.get((_short(rec["model"]), str(rec["problem_id"])))
            texts = rec.get("rollout_texts") or []
            if drop_leaked and rec.get("leaked"):
                texts = [t for t, l in zip(texts, rec["leaked"]) if not l]
            if g is None or not texts: return None
            vals = []
            for t in texts:
                a = prod.extract_answer_new(t); unk = (a == "UNKNOWN")
                if unk and mode == "dropped": continue
                vals.append((mode == "as_correct") if unk else
                            (prod.normalise_answer(a) == prod.normalise_answer(g)))
            return None if not vals else 1.0 - sum(vals) / len(vals)

        res = {}
        variants = [("as_incorrect", False), ("dropped", False),
                    ("as_correct", False)]
        if iv == "lexical": variants.append(("as_incorrect", True))
        for mode, dl in variants:
            tag = mode + ("+leak_filter" if dl else "")
            A = [(score(r, mode, dl), r) for r in recs["revision"]]
            B = [(score(r, mode, dl), r) for r in recs["filler"]]
            A = [(s, r) for s, r in A if s is not None]
            B = [(s, r) for s, r in B if s is not None]
            a = np.array([s for s, _ in A]); b = np.array([s for s, _ in B])
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            lo1, hi1, G1 = boot_mean_diff(a, b, [str(r["problem_id"]) for _, r in A],
                                          [str(r["problem_id"]) for _, r in B])
            lo2, hi2, G2 = boot_mean_diff(a, b, [f"{r['model']}|{r['problem_id']}" for _, r in A],
                                          [f"{r['model']}|{r['problem_id']}" for _, r in B])
            res[tag] = {"n": f"{len(a)}/{len(b)}", "rev": float(a.mean()),
                        "fil": float(b.mean()), "diff": float(a.mean() - b.mean()),
                        "mw_p": float(p), "ci_problem": [lo1, hi1, G1],
                        "ci_model_problem": [lo2, hi2, G2]}
            print(f"  {tag:<26} n={len(a)}/{len(b)}  {a.mean():.4f} vs {b.mean():.4f}"
                  f"  diff {a.mean()-b.mean():+.4f}  MW p={p:.4g}"
                  f"  CI(problem, {G1}) [{lo1:+.4f}, {hi1:+.4f}]")
            if mode == "as_incorrect" and not dl:
                # repetition-controlled estimate, clustered by problem
                rows = []
                for s, r in A + B:
                    e = evs.get(r.get("eid"))
                    if e and e.get("repetition") is not None:
                        rows.append((s, 1.0 if r["arm"] == "revision" else 0.0,
                                     e["repetition"], str(r["problem_id"])))
                if len(rows) > 20:
                    Y = np.array([x[0] for x in rows]); X1 = np.array([x[1] for x in rows])
                    C = np.array([x[2] for x in rows]); C = C - C.mean()
                    beta, D, G = cluster_boot(Y, np.column_stack([np.ones_like(Y), X1, C]),
                                              [x[3] for x in rows])
                    lo, hi, pp = ci_p(D[:, 1])
                    res["repetition_controlled"] = {"arm": float(beta[1]),
                                                    "ci": [lo, hi], "p": pp,
                                                    "rep_slope": float(beta[2]),
                                                    "clusters": G}
                    print(f"  {'+ repetition (problem cl.)':<26} arm {beta[1]:+.4f} "
                          f"[{lo:+.4f}, {hi:+.4f}] p={pfmt(pp)}  slope {beta[2]:+.4f}")
        # per model, and pooled without Llama-8B, NEW vs OLD rollouts scored
        # by the same function (old files: published 83/80 rollouts)
        OLD_FILES = {"lexical": ("counterfactual_rollout_results.json",
                                 "filler_control_rollout_results.json"),
                     "steering": ("steering_revision_rollouts_powered.json",
                                  "steering_filler_rollouts_powered.json")}
        oldr = {arm: rjson(EXPORT / f, []) or []
                for arm, f in zip(("revision", "filler"), OLD_FILES[iv])}
        def arm_means(rr, keep):
            v = [score(r, "as_incorrect") for r in rr if keep(_short(r.get("model")))]
            v = [x for x in v if x is not None]
            return (float(np.mean(v)) if v else float("nan")), len(v)
        print(f"\n  per model, 'as_incorrect' (NEW clean-prefix  |  OLD published)")
        res["per_model"] = {}
        groups_ = [(m, (lambda mm: (lambda x: x == mm))(m)) for m in MODELS]
        if "DeepSeek-R1-Distill-Llama-8B" in MODELS and len(MODELS) > 1:
            groups_.append(("pooled, excluding Llama-8B",
                            lambda x: x != "DeepSeek-R1-Distill-Llama-8B"))
        groups_.append(("pooled, these models", lambda x: x in MODELS))
        for name, keep in groups_:
            nr, nnr = arm_means(recs["revision"], keep)
            nf, nnf = arm_means(recs["filler"], keep)
            orr, onr = arm_means(oldr["revision"], keep)
            of, onf = arm_means(oldr["filler"], keep)
            print(f"    {name[-30:]:<31}NEW {nr:.3f} vs {nf:.3f} diff {nr-nf:+.3f}"
                  f" (n {nnr}/{nnf})  |  OLD {orr:.3f} vs {of:.3f} diff {orr-of:+.3f}"
                  f" (n {onr}/{onf})")
            res["per_model"][name] = {"new": [nr, nf, nnr, nnf],
                                      "old": [orr, of, onr, onf]}
        # per model with CIs (appendix table): the pooled estimator applied to
        # each model's events -- MW two-sided, cluster bootstrap by problem_id
        # (B = N_BOOT, seed 8080, 2.5/97.5 percentiles)
        print(f"\n  per model, 'as_incorrect', with CIs (NEW rollouts)")
        res["per_model_ci"] = {}
        for name in MODELS:
            A = [(score(r, "as_incorrect"), r) for r in recs["revision"] if _short(r["model"]) == name]
            B = [(score(r, "as_incorrect"), r) for r in recs["filler"] if _short(r["model"]) == name]
            A = [(x, r) for x, r in A if x is not None]; B = [(x, r) for x, r in B if x is not None]
            if len(A) < 2 or len(B) < 2:
                continue
            a = np.array([x for x, _ in A]); b = np.array([x for x, _ in B])
            lo, hi, G = boot_mean_diff(a, b, [str(r["problem_id"]) for _, r in A],
                                       [str(r["problem_id"]) for _, r in B])
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            res["per_model_ci"][name] = {"n": f"{len(a)}/{len(b)}", "rev": float(a.mean()), "fil": float(b.mean()),
                                         "diff": float(a.mean() - b.mean()), "mw_p": float(p),
                                         "ci_problem": [lo, hi, G]}
            print(f"    {name[-30:]:<31}{a.mean():.3f} vs {b.mean():.3f}  diff {a.mean()-b.mean():+.3f}"
                  f"  MW p={p:.3g}  CI(problem, {G}) [{lo:+.3f}, {hi:+.3f}]  (n {len(a)}/{len(b)})")
        # pooled-without-Llama CI, problem-clustered, as_incorrect
        if "DeepSeek-R1-Distill-Llama-8B" in MODELS and len(MODELS) > 1:
            A = [(score(r, "as_incorrect"), r) for r in recs["revision"]
                 if _short(r["model"]) != "DeepSeek-R1-Distill-Llama-8B"]
            B = [(score(r, "as_incorrect"), r) for r in recs["filler"]
                 if _short(r["model"]) != "DeepSeek-R1-Distill-Llama-8B"]
            A = [(x, r) for x, r in A if x is not None]; B = [(x, r) for x, r in B if x is not None]
            a = np.array([x for x, _ in A]); b = np.array([x for x, _ in B])
            lo, hi, G = boot_mean_diff(a, b, [str(r["problem_id"]) for _, r in A],
                                       [str(r["problem_id"]) for _, r in B])
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            print(f"    excluding Llama-8B: diff {a.mean()-b.mean():+.4f}  MW p={p:.4g}"
                  f"  CI(problem, {G}) [{lo:+.4f}, {hi:+.4f}]")
            res["excluding_llama8b"] = {"diff": float(a.mean() - b.mean()), "mw_p": float(p),
                                        "ci_problem": [lo, hi, G]}
        # equivalence with production necessity_scores on the same file
        try:
            pr = prod.necessity_scores(ROLL[(label, "revision")], gt,
                                       prod.extract_answer_new, "as_incorrect")
            mine = sorted(s for s in (score(r, "as_incorrect") for r in recs["revision"])
                          if s is not None)
            ok = len(pr) == len(mine) and np.allclose(
                sorted(x["necessity"] for x in pr), mine)
            print(f"  equivalence vs production necessity_scores(): "
                  f"{'OK' if ok else 'MISMATCH'} ({len(mine)} vs {len(pr)})")
            res["equivalence_ok"] = bool(ok)
        except Exception as e:
            print(f"  equivalence check not run: {type(e).__name__}: {e}")
        if iv == "lexical":
            nl = sum(sum(r.get("leaked") or []) for r in recs["revision"] + recs["filler"])
            nt = sum(len(r.get("rollout_texts") or []) for r in recs["revision"] + recs["filler"])
            print(f"  leaked rollouts: {nl} of {nt}")
            res["leaked"] = [nl, nt]
        o = OLD[iv]
        print(f"\n  OLD ({o['n']}): {o['rev']:.3f} vs {o['fil']:.3f}  diff +{o['diff']:.3f}"
              f"  p={o['p']}  CI [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]"
              f"  + repetition +{o['rep']:.3f}")
        out[label] = res

    wjson(out, RESULTS)
    manifest_add({"stage": "analyze", "started": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
                  "finished": now(), "wall_clock_s": round(time.time() - t0, 1),
                  "env": env_info(), "sources": [src_scorer]})
    print(f"\n  wrote {RESULTS}")
    if SAMPLE == "original":
        print("""
  ORIGINAL SAMPLE, CLEAN PREFIXES. Same 83/80 events as published; the only
  changes are the restored prefix (Llama-8B), the two lexical events that were
  never generated, and event-keyed resume. Qwen prefixes were already clean
  and the seeds are the published ones (lexical 1000+r, steering 4000+r), so
  for the three Qwen models NEW vs OLD is a replication check. Llama-8B NEW
  vs OLD is the effect of the fix. STEERING_SEED1000 is the first steering
  re-run, made with 1000+r by mistake: an independent seed set, not the
  published procedure; a seed-robustness check only.""")
    print("""
  REPORT BOTH. The old 83/80 numbers stay as they were. 'as_incorrect' without the leak filter
  is the treatment the published necessity numbers used. The leak-filtered line is a sensitivity.""")


if __name__ == "__main__" or True:
    {"select": (stage_select_original if SAMPLE == "original" else stage_select),
     "lexical": lambda: stage_generate("lexical"),
     "steering": lambda: stage_generate("steering_llama8b_reextract" if LLAMA_DIRECTION == "reextract"
                                        else "steering"),
     "analyze": stage_analyze}[STAGE]()
