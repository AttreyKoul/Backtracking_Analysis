# =============================================================================
# WHERE ARE THE LLAMA-8B PROBE WINDOWS?  (CPU, tokenizers only; ~10-20 min)
#
# The probe-robustness `align` stage recomputed each window's centre with the
# extractor's own procedure (notebooks 03 / 05) and compared it with the
# token_idx stored next to the activations:
#     Qwen-1.5B / 7B / 14B : 100% identical
#     Llama-8B             : 3 of 1,021 revision, 2 of 505 filler identical
# So for Llama-8B the stored windows were NOT made from (current traces +
# current labels + current tokenizer + the extractor's text cleaning). This
# script tests which ingredient differs, using only stored metadata, the
# traces, the label / reject files and the tokenizer. It changes nothing.
#
#   A. row mapping   stored word_idx (revision meta) vs the label rows, with
#                    and without the `verified` filter
#   B. file times    is the trace / label file newer than its activation file?
#   C. token index   recompute token_idx from the STORED word_idx under each
#                    text variant (source field x cleaning) and prompt variant,
#                    and count exact matches with the stored token_idx
# Qwen-7B runs alongside as the control (it should match under the current
# variant).  Output: hand_labeling_export/llama8b_window_alignment.json
# =============================================================================
import json, os, time
from pathlib import Path
from collections import Counter

MODELS = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-7B"]
MAX_PROBLEMS = None          # None = every problem with activations

DATA_ROOT = Path(os.environ.get("REVISION_DATA_ROOT", "/content/gdrive/MyDrive/llm_planning_study"))
if not DATA_ROOT.exists():
    from google.colab import drive; drive.mount("/content/gdrive")
ACT, TRACES, LABELS = DATA_ROOT / "activations", DATA_ROOT / "traces", DATA_ROOT / "labels"
OUT = DATA_ROOT / "hand_labeling_export" / "llama8b_window_alignment.json"
CENTER = 30

def rjson(p, d=None):
    try: return json.load(open(p))
    except Exception: return d

def clean_text(t):                    # notebooks 02-05
    return t.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")

def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
U2B = _b2u()
def bytes_decode(t):                  # full byte-level decode (as 09's restore_text)
    try: return bytes(U2B[c] for c in t).decode("utf-8", errors="replace")
    except KeyError: return None

CLEANINGS = {"clean_text (current)": clean_text, "none (raw)": lambda t: t, "byte-level decode": bytes_decode}
SOURCES = ["think->response_clean (current)", "response_clean", "full_response"]

def source_text(tr, src, cl):
    if src.startswith("think"):
        th = cl(tr.get("think_content", "") or "") or ""; rc = cl(tr.get("response_clean", "") or "") or ""
        return th if th.strip() else rc
    return cl(tr.get(src, "") or "")

def build_prompt(tr, tok):
    instr = ("Solve the following competition mathematics problem carefully. "
             "Show your reasoning step by step and verify your answer."
             if tr.get("difficulty") == "hard" else
             "Solve the following mathematics competition problem. Show your working step by step.")
    msgs = [{"role": "user", "content": f"{instr}\n\nProblem: {tr['problem']}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

from transformers import AutoTokenizer
report = {}
for model in MODELS:
    t0 = time.time()
    print(f"\n{'='*78}\n{model}\n{'='*78}")
    tok = AutoTokenizer.from_pretrained(f"deepseek-ai/{model}", trust_remote_code=True)
    metas = sorted((ACT / model).glob("*_meta.json"))
    if MAX_PROBLEMS: metas = metas[:MAX_PROBLEMS]
    rowmap = Counter(); times = Counter(); res = Counter(); deltas = {}
    for mp in metas:
        pid = mp.name[:-len("_meta.json")]
        meta = rjson(mp, {}) or {}; rows = meta.get("revision_meta") or []
        tr = rjson(TRACES / model / f"{pid}.json"); lab = rjson(LABELS / model / f"{pid}.json")
        if tr is None or not rows: continue
        # A. row mapping
        revs = (lab or {}).get("revisions") or []
        ver = [r for r in revs if r.get("verified", True)]
        for name, lst in (("all label rows", revs), ("verified rows only", ver)):
            for i, m in enumerate(rows):
                ok = i < len(lst) and lst[i].get("revision_word_idx") == m.get("word_idx")
                rowmap[f"{name}: stored word_idx {'==' if ok else '!='} label row"] += 1
        # B. file times
        npz = ACT / model / f"{pid}.npz"
        try:
            ta = npz.stat().st_mtime
            for nm, p in (("trace", TRACES / model / f"{pid}.json"), ("label", LABELS / model / f"{pid}.json")):
                times[f"{nm} file {'NEWER' if p.stat().st_mtime > ta else 'older'} than activations"] += 1
        except Exception:
            times["mtime not available"] += 1
        # C. token index under each variant, from the STORED word_idx
        base_prompt = build_prompt(tr, tok)
        prompts = {"template (current)": base_prompt}
        if base_prompt.endswith("<think>\n"):
            prompts["template without trailing <think>\\n"] = base_prompt[:-len("<think>\n")]
        else:
            prompts["template + <think>\\n"] = base_prompt + "<think>\n"
        for pn, prompt in prompts.items():
            plen = len(tok(prompt)["input_ids"])
            for src in SOURCES:
                for cn, cl in CLEANINGS.items():
                    text = source_text(tr, src, cl)
                    if not text: continue
                    seq = len(tok(prompt + text)["input_ids"]); words = text.split()
                    key = f"{pn} | {src} | {cn}"
                    for m in rows:
                        w = int(m.get("word_idx", 0))
                        t = plen + len(tok(" ".join(words[:w]), add_special_tokens=False)["input_ids"])
                        t = max(CENTER, min(t, seq - CENTER - 1))
                        d = int(m["token_idx"]) - t
                        res[(key, d == 0)] += 1
                        deltas.setdefault(key, Counter())[d] += 1
    keys = sorted({k for k, _ in res}, key=lambda k: -res[(k, True)])
    print("  A. row mapping:", dict(rowmap))
    print("  B. file times: ", dict(times))
    print("  C. stored token_idx reproduced, by variant (best first):")
    rows_out = []
    for k in keys:
        n_ok, n = res[(k, True)], res[(k, True)] + res[(k, False)]
        mc = deltas[k].most_common(3)
        print(f"     {n_ok:>5}/{n:<5} {n_ok/n:6.1%}  {k}   most common stored-recomputed {mc}")
        rows_out.append({"variant": k, "match": n_ok, "n": n, "most_common_delta": mc})
    report[model] = {"row_mapping": dict(rowmap), "file_times": dict(times), "token_idx_by_variant": rows_out,
                     "seconds": round(time.time() - t0)}
    json.dump(report, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
