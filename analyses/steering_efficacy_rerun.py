# =============================================================================
# STEERING EFFICACY ON THE RE-RUN (clean prefixes)  +  RE-RUN COMPUTE TIME
#
# The paper's efficacy numbers (+19.3 / +31.0 / -2.0 / +50.5 %, "terms
# reduction") came from steering_efficacy_report.json, written by the efficacy
# diagnostic: an UNSTEERED (alpha = 0) baseline on the first 8 revision events
# per model (sorted by problem_id), 2 rollouts each (seeds 9000 + r, 2048 new
# tokens, temperature 0.6, top-p 0.95), compared with the steered rollouts on
# the same events. Metric: mean number of distinct backtracking terms per
# rollout (20 terms, observed, never banned), and
#     terms_reduction_pct = (terms_base - terms_steer) / terms_base.
#
# That baseline was generated from the OLD prefixes, which for Llama-8B were in
# byte-level characters. This file repeats the diagnostic on the re-run:
#   STAGE = "baseline": regenerate the alpha = 0 baseline with the CLEAN
#       prefixes, same subset rule, same seeds, same generation settings, same
#       generation code (the diagnostic's loop, verbatim). GPU; 64 generations.
#   STAGE = "report":  (CPU) recompute efficacy against the re-run's steered
#       rollouts on the same events, and print beside it
#         - the paper's values (old baseline vs old steered), and
#         - the old baseline vs the NEW steered rollouts (a cross-check; for
#           Llama-8B this mixes a corrupted-prefix baseline with clean-prefix
#           steering and is labelled as such).
#       Also sums the re-run's compute time from the per-event wall clocks
#       stored in the rollout files and the per-session manifest records.
#       If the Llama-8B steering re-run with the rebuilt direction exists
#       (09, LLAMA_DIRECTION = "reextract"), its efficacy is reported too; that
#       is the Llama-8B number the paper uses.
#
# Output: hand_labeling_export/redraw_2026_10/original_rerun_clean_prefix/
#         steering_efficacy_baseline_clean.json, steering_efficacy_rerun.json
# =============================================================================
STAGE = "baseline"        # baseline | report

import json, os, re, time
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE = Path(os.environ.get("REVISION_DATA_ROOT", "/content/gdrive/MyDrive/llm_planning_study"))
if not BASE.exists():
    from google.colab import drive; drive.mount("/content/gdrive")
EXPORT  = BASE / "hand_labeling_export"
RUN_DIR = EXPORT / "redraw_2026_10" / "original_rerun_clean_prefix"
EV_REV        = RUN_DIR / "redraw_revision_events.json"
STEER_NEW     = RUN_DIR / "redraw_steering_revision_rollouts_seed4000.json"   # published seeds, 4000 + r
STEER_LLAMA_RE = RUN_DIR / "redraw_steering_revision_rollouts_seed4000_llama8b_reextract.json"  # Llama-8B, rebuilt direction
ROLL_FILES    = {("lexical", arm): RUN_DIR / f"redraw_lexical_{arm}_rollouts.json" for arm in ("revision", "filler")}
ROLL_FILES.update({("steering", arm): RUN_DIR / f"redraw_steering_{arm}_rollouts_seed4000.json"
                   for arm in ("revision", "filler")})
ROLL_FILES.update({("steering_llama8b_reextract", arm):
                   RUN_DIR / f"redraw_steering_{arm}_rollouts_seed4000_llama8b_reextract.json" for arm in ("revision", "filler")})
ROLL_FILES.update({("steering_seed1000", arm): RUN_DIR / f"redraw_steering_{arm}_rollouts.json"
                   for arm in ("revision", "filler")})   # first steering re-run (1000 + r, not the procedure)
MANIFEST      = RUN_DIR / "manifest.json"
BASELINE_NEW  = RUN_DIR / "steering_efficacy_baseline_clean.json"
REPORT_NEW    = RUN_DIR / "steering_efficacy_rerun.json"
BASELINE_OLD  = EXPORT / "steering_efficacy_baseline.json"
REPORT_OLD    = EXPORT / "steering_efficacy_report.json"

# ---- from the efficacy diagnostic, verbatim ---------------------------------
N_EVENTS_PER_MODEL     = 8
N_ROLLOUTS             = 2
ROLLOUT_MAX_NEW_TOKENS = 2048
SAMPLE_TEMPERATURE     = 0.6
SAMPLE_TOP_P           = 0.95
CHOSEN_ALPHA_MULTIPLIER = {
    "DeepSeek-R1-Distill-Llama-8B":  1.5,
    "DeepSeek-R1-Distill-Qwen-1.5B": 3.0,
    "DeepSeek-R1-Distill-Qwen-7B":   1.5,
    "DeepSeek-R1-Distill-Qwen-14B":  5.0,
}
BACKTRACK_TERMS = [
    "wait", "actually", "alternatively", "hmm", "hold on",
    "let me reconsider", "let me rethink", "on second thought",
    "let me try again", "let me go back", "let me recheck",
    "let me double check", "let me double-check",
    "that's wrong", "that is wrong", "that's not right", "that is not right",
    "i made a mistake", "i made an error", "this doesn't work",
]

def extract_answer(text: str) -> str:
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r'(?:answer is|therefore)[:\s]*([0-9A-Za-z\-\.\/]+)',
                  text, re.IGNORECASE)
    return m.group(1).strip() if m else "UNKNOWN"

def build_prompt(problem, tokenizer, model_name):
    is_reasoning = any(t in model_name for t in ["R1", "QwQ"])
    if problem.get("difficulty") == "hard":
        instruction = ("Solve the following competition mathematics problem carefully. "
                        "Show your reasoning step by step and verify your answer.")
    else:
        instruction = ("Solve the following mathematics competition problem. "
                        "Show your working step by step.")
    if is_reasoning:
        msgs = [{"role": "user", "content": f"{instruction}\n\nProblem: {problem['problem']}"}]
    else:
        msgs = [{"role": "system", "content": "You are an expert mathematician. Solve problems "
                                              "with clear step-by-step reasoning."},
                {"role": "user", "content": f"{instruction}\n\nProblem: {problem['problem']}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def backtrack_rate(texts):
    if not texts:
        return None
    hits = sum(1 for t in texts if any(term in t.lower() for term in BACKTRACK_TERMS))
    return hits / len(texts)

def terms_per_rollout(texts):
    if not texts:
        return None
    return float(np.mean([sum(1 for term in BACKTRACK_TERMS if term in t.lower())
                          for t in texts]))
# -----------------------------------------------------------------------------

def rjson(p, d=None):
    try: return json.load(open(p))
    except Exception: return d

def wjson(o, p):
    tmp = Path(str(p) + ".tmp"); json.dump(o, open(tmp, "w"), indent=2); os.replace(tmp, p)

def key(r): return (r["model"], str(r["problem_id"]), int(r["char_start"]))

def subset():
    """The diagnostic's rule: first N revision events per model by problem_id
    (stable sort, so ties keep event-file order)."""
    by = defaultdict(list)
    for e in rjson(EV_REV, []) or []:
        by[e["model"]].append(e)
    return {m: sorted(v, key=lambda e: e["problem_id"])[:N_EVENTS_PER_MODEL] for m, v in by.items()}


def stage_baseline():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    out = rjson(BASELINE_NEW, []) or []
    done = {key(r) for r in out}
    if done:
        print(f"Resuming: {len(done)} baseline events already done.")
    for model_short, evs in subset().items():
        remaining = [e for e in evs if key(e) not in done]
        if not remaining:
            print(f"\n{model_short}: baseline complete -- skipping load."); continue
        model_full = f"deepseek-ai/{model_short}"
        print(f"\n{'='*62}\n{model_full} — UNSTEERED baseline, clean prefixes ({len(remaining)} events)\n{'='*62}")
        tok = AutoTokenizer.from_pretrained(model_full, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_full, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        model.eval()
        for ev in remaining:
            prompt = build_prompt({"problem": ev["problem"], "difficulty": ev["difficulty"]},
                                  tok, model_full)
            prefix = prompt + ev["prefix_response_text"]
            texts = []
            t0 = time.time()
            for r in range(N_ROLLOUTS):
                torch.manual_seed(9000 + r)
                inp = tok(prefix, return_tensors="pt", truncation=False).to(model.device)
                with torch.no_grad():
                    o = model.generate(**inp, max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
                                       do_sample=True, temperature=SAMPLE_TEMPERATURE,
                                       top_p=SAMPLE_TOP_P, pad_token_id=tok.pad_token_id,
                                       eos_token_id=tok.eos_token_id)
                texts.append(tok.decode(o[0][inp["input_ids"].shape[1]:],
                                        skip_special_tokens=False))
            print(f"  {ev['eid']} {ev['problem_id']}: backtrack rate "
                  f"{backtrack_rate(texts):.2f}  ({time.time()-t0:.0f}s)")
            out.append({"eid": ev["eid"], "problem_id": ev["problem_id"], "model": model_short,
                        "char_start": ev["char_start"], "alpha": 0.0,
                        "seeds": [9000 + r for r in range(N_ROLLOUTS)],
                        "wall_clock_s": round(time.time() - t0, 1), "rollout_texts": texts})
            wjson(out, BASELINE_NEW)
        del model
        torch.cuda.empty_cache()
    print(f"\nWrote {len(out)} baseline events -> {BASELINE_NEW}")


def rows_for(base_by, steer_by):
    rows = {}
    for m in sorted(set(base_by) & set(steer_by)):
        b, s = base_by[m], steer_by[m]
        bt, st = terms_per_rollout(b), terms_per_rollout(s)
        br, sr = backtrack_rate(b), backtrack_rate(s)
        rows[m] = {"alpha_mult": CHOSEN_ALPHA_MULTIPLIER.get(m), "n_base": len(b), "n_steer": len(s),
                   "backtrack_base": br, "backtrack_steer": sr, "backtrack_reduction": br - sr,
                   "terms_base": bt, "terms_steer": st,
                   "terms_reduction_pct": (bt - st) / bt if bt else None,
                   "extract_base": float(np.mean([extract_answer(t) != "UNKNOWN" for t in b])),
                   "extract_steer": float(np.mean([extract_answer(t) != "UNKNOWN" for t in s]))}
    return rows


def stage_report():
    sub = subset()
    keys = {key(e) for v in sub.values() for e in v}
    steer = [r for r in (rjson(STEER_NEW, []) or []) if key(r) in keys]
    base = rjson(BASELINE_NEW, []) or []
    s_by, b_by = defaultdict(list), defaultdict(list)
    for r in steer: s_by[r["model"]].extend(r.get("rollout_texts") or [])
    for r in base:  b_by[r["model"]].extend(r.get("rollout_texts") or [])
    new = rows_for(b_by, s_by)
    print(f"  inputs: clean-prefix baseline {len(base)} events "
          f"({sum(float(r.get('wall_clock_s') or 0) for r in base):.0f}s) in {BASELINE_NEW.name}; "
          f"steered (4000 + r) {len(steer)} of {len(keys)} subset events in {STEER_NEW.name}"
          + ("" if STEER_NEW.exists() else "  <- file not found: run 09 STAGE = 'steering' first"))

    # cross-check: the OLD baseline file against the NEW steered rollouts, on the
    # events both cover (matched on model + problem_id, as the diagnostic did)
    old_b = rjson(BASELINE_OLD, []) or []
    ob_keys = {(r["model"], str(r["problem_id"])) for r in old_b}
    s2 = defaultdict(list); ob = defaultdict(list)
    for r in rjson(STEER_NEW, []) or []:
        if (r["model"], str(r["problem_id"])) in ob_keys: s2[r["model"]].extend(r.get("rollout_texts") or [])
    for r in old_b: ob[r["model"]].extend(r.get("rollout_texts") or [])
    cross = rows_for(ob, s2)
    paper = rjson(REPORT_OLD, {}) or {}

    print(f"\n{'#'*78}\n# STEERING EFFICACY — terms reduction per model\n{'#'*78}")
    print(f"  {'model':<16}{'k':>5}  {'paper (old/old)':>16}  {'re-run (new/new)':>17}  "
          f"{'old base/new steer':>19}  {'extract base->steer (new)':>26}")
    for m in sorted(set(new) | set(paper) | set(cross)):
        f = lambda d: (f"{d[m]['terms_reduction_pct']:+.1%}" if m in d and d[m].get("terms_reduction_pct") is not None else "n/a")
        ex = (f"{new[m]['extract_base']:.0%} -> {new[m]['extract_steer']:.0%}" if m in new else "")
        flag = "  (old baseline had byte-level prefixes)" if "Llama-8B" in m else ""
        print(f"  {m[-15:]:<16}{CHOSEN_ALPHA_MULTIPLIER.get(m, ''):>5}  {f(paper):>16}  {f(new):>17}  "
              f"{f(cross):>19}  {ex:>26}{flag}")
    for m, r in new.items():
        print(f"    {m[-15:]}: terms/rollout {r['terms_base']:.2f} -> {r['terms_steer']:.2f}  "
              f"(n rollouts {r['n_base']} base / {r['n_steer']} steered)")
    llama_re = {}
    if STEER_LLAMA_RE.exists():
        s3 = defaultdict(list)
        for r in rjson(STEER_LLAMA_RE, []) or []:
            if key(r) in keys and "Llama-8B" in r["model"]:
                s3[r["model"]].extend(r.get("rollout_texts") or [])
        llama_re = rows_for(b_by, s3)
        for m, r in llama_re.items():
            print(f"\n  {m}, direction REBUILT from the re-extracted windows (k = {r['alpha_mult']}):")
            print(f"    terms reduction {r['terms_reduction_pct']:+.1%}  (terms/rollout {r['terms_base']:.2f} -> "
                  f"{r['terms_steer']:.2f}; n {r['n_base']} base / {r['n_steer']} steered)  "
                  f"extract {r['extract_base']:.0%} -> {r['extract_steer']:.0%}")
            if m in new:
                print(f"    (original-window direction: {new[m]['terms_reduction_pct']:+.1%})")

    # ---- compute time --------------------------------------------------------
    timing = {}
    for (iv, arm), p in ROLL_FILES.items():
        for r in rjson(p, []) or []:
            t = timing.setdefault(iv, {"events": 0, "sum_event_wall_clock_s": 0.0, "by_model": defaultdict(float),
                                       "gpus": set()})
            t["events"] += 1; t["sum_event_wall_clock_s"] += float(r.get("wall_clock_s") or 0)
            t["by_model"][r["model"]] += float(r.get("wall_clock_s") or 0); t["gpus"].add(r.get("gpu"))
    sessions = defaultdict(list)
    for rec in ((rjson(MANIFEST, {}) or {}).get("stages") or []):
        if isinstance(rec, dict) and rec.get("stage") in ("lexical", "steering", "steering_llama8b_reextract"):
            rb = (rec.get("seeds") or {}).get("rollout_base")
            lab = rec["stage"]
            if lab == "steering" and not isinstance(rb, dict):     # first re-run: one base (1000) for both
                lab = "steering_seed1000"
            sessions[lab].append({"started": rec.get("started"), "finished": rec.get("finished"),
                                           "wall_clock_s": rec.get("wall_clock_s"),
                                           "per_model_wall_clock_s": rec.get("per_model_wall_clock_s")})
    for iv, t in timing.items():
        t["by_model"] = dict(t["by_model"]); t["gpus"] = sorted(g for g in t["gpus"] if g)
        t["sessions"] = sessions.get(iv, [])
        t["sum_session_wall_clock_s"] = float(sum(s["wall_clock_s"] or 0 for s in t["sessions"]))
        print(f"\n  {iv}: {t['events']} events; sum of per-event wall clock {t['sum_event_wall_clock_s']:.0f}s "
              f"({t['sum_event_wall_clock_s']/3600:.1f} h); {len(t['sessions'])} session(s) in the manifest, "
              f"total {t['sum_session_wall_clock_s']:.0f}s (includes model loading); GPU {t['gpus']}")
        for m, s in sorted(t["by_model"].items()):
            print(f"      {m[-15:]:<16}{s:>9.0f}s")
    base_s = sum(float(r.get("wall_clock_s") or 0) for r in base)
    out = {"subset_rule": f"first {N_EVENTS_PER_MODEL} revision events per model by problem_id",
           "subset": {m: [e["eid"] for e in v] for m, v in sub.items()},
           "rerun_new_baseline_vs_new_steered": new,
           "llama8b_rebuilt_direction": llama_re,
           "old_baseline_vs_new_steered": cross,
           "paper_old_baseline_vs_old_steered": paper,
           "compute": timing, "efficacy_baseline_wall_clock_s": base_s}
    wjson(out, REPORT_NEW)
    print(f"\nWrote {REPORT_NEW}")


if STAGE == "baseline":
    stage_baseline()
elif STAGE == "report":
    stage_report()
else:
    raise SystemExit(f"unknown STAGE {STAGE!r}")
