# =============================================================================
# PROBE ROBUSTNESS CHECKS (the Table 1 / "Table 4" sample, CPU only)
#
#   2a  one estimator        matched, null and residual all from the same paired
#                            draws (PCA-50, Table 1's layer). With PCA-50 this
#                            must reproduce Table 1's residual_paired exactly.
#   1a  PCA sensitivity      PCA-50 (reference), PCA-100, PCA-200, no PCA
#   1b  layer selection      choose the layer on half of the problems by matched
#                            accuracy, evaluate on the other half, both ways;
#                            plus the residual at every layer (full sample)
#   1c  repetition control   arm ~ probe_score + repetition (out-of-fold scores,
#                            clustered by problem); repetition-matched subsample
#                            with the caliper rule used for Qwen-7B
#   1d  positions            the same probe at token positions relative to the
#                            TRUE first token of the marker (-5..-1, 0, +1..+3)
#                            and at the paragraph-break token before it
#
# Sample and pipeline are Table 1's: every revision window
# (activations/{model}/*.npz) and every filler window
# (activations_filler/{model}/*.npz); StandardScaler -> PCA fit inside each
# training fold -> logistic regression (L2, C = 1, max_iter 2000);
# StratifiedGroupKFold(5) grouped on event; balanced draws; the paired null
# (filler vs filler, a random split of the SAME filler subsample used as the
# matched negative class). Layers are Table 1's, read from
# results_v3/task1_residual_pointfix.json.
#
# WHY 1d NEEDS AN ALIGNMENT STEP. The extractors placed the window centre with
# an approximate word->token conversion (words re-joined with single spaces and
# tokenized on their own; notebook 03 says "approximate but sufficient for ±30
# token windows"). The centre is therefore near, not exactly at, the token
# before the marker. The `align` stage re-tokenizes the same prompt + text with
# character offsets, finds the marker's true first token, and records where it
# sits in each stored window. It reports how far the stored centre is from it.
#
# HOW TO RUN (Colab, CPU runtime; high RAM recommended for Qwen-14B):
#   Run the file. The `cache` stage reads every activation file once (I/O
#   bound, most of the time) and keeps what is needed on LOCAL disk; it resumes
#   if the Drive mount drops. Every result is saved per (part, model, setting)
#   as soon as it is computed, so an interrupted run resumes where it stopped.
#   `align` downloads each model's tokenizer (no weights).
#   Output: hand_labeling_export/probe_robustness_checks.json and
#           hand_labeling_export/probe_robustness_draws.json (per-draw values)
# =============================================================================
import gc, json, os, pickle, random, re, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

PARTS   = ["cache", "1a", "1b", "1c", "align", "1d"]
ACTIVATIONS = "original"     # "original": Table 1's activations, all four models
                             # "llama8b_reextract": Llama-8B only, on the Nov 2026 re-extraction
                             #   (pipeline/11_reextract_llama8b.py); separate output files
MODELS  = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-1.5B",
           "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-14B"]
CENTER  = 30
SEED    = 42                 # Table 1's seed (paired draw d uses SEED + 10000 + d)
MAX_ITER = 2000
N_DRAWS_MAIN    = 100        # paired draws for 1a and the 1b evaluation (Table 1 used 100)
N_DRAWS_NOPCA   = 20         # no-PCA fits are much slower; fewer draws, reported as such
N_DRAWS_PROFILE = 10         # paired draws per layer for the profile and the layer choice
N_DRAWS_OFFSET  = 30         # paired draws per position in 1d
N_DRAWS_JOINT   = 10         # balanced draws for the 1c joint model
N_BOOT          = 2000       # cluster bootstrap for 1c
SPLIT_SEED      = 20261031   # 1b problem split
PCA_SETTINGS    = [50, 100, 200, None]
OFFSETS         = [-5, -4, -3, -2, -1, 0, 1, 2, 3]

DATA_ROOT = Path(os.environ.get("REVISION_DATA_ROOT", "/content/gdrive/MyDrive/llm_planning_study"))
if not DATA_ROOT.exists():
    from google.colab import drive; drive.mount("/content/gdrive")
ACT_REV  = DATA_ROOT / "activations"
ACT_FIL  = DATA_ROOT / "activations_filler"
LABELS   = DATA_ROOT / "labels"
REJECTS  = DATA_ROOT / "rejects"
TRACES   = DATA_ROOT / "traces"
EXPORT   = DATA_ROOT / "hand_labeling_export"
POINTFIX = DATA_ROOT / "results_v3" / "task1_residual_pointfix.json"
CACHE    = Path(os.environ.get("PROBE_CACHE", "/content/probe_cache" if Path("/content").exists()
                               else str(DATA_ROOT.parent / "probe_cache")))
CACHE.mkdir(parents=True, exist_ok=True)
EXPORT.mkdir(parents=True, exist_ok=True)
OUT      = EXPORT / "probe_robustness_checks.json"
DRAWS    = EXPORT / "probe_robustness_draws.json"
if ACTIVATIONS == "llama8b_reextract":
    MODELS   = ["DeepSeek-R1-Distill-Llama-8B"]
    ACT_REV  = DATA_ROOT / "activations_reextract_2026_11"
    ACT_FIL  = DATA_ROOT / "activations_filler_reextract_2026_11"
    POINTFIX = DATA_ROOT / "results_v3" / "task1_residual_pointfix_llama8b_reextract.json"
    CACHE    = CACHE / "llama8b_reextract"; CACHE.mkdir(parents=True, exist_ok=True)
    OUT      = EXPORT / "probe_robustness_checks_llama8b_reextract.json"
    DRAWS    = EXPORT / "probe_robustness_draws_llama8b_reextract.json"
elif ACTIVATIONS != "original":
    raise SystemExit(f"unknown ACTIVATIONS {ACTIVATIONS!r}")
print(f"activations: {ACTIVATIONS}  ({ACT_REV.name}, {ACT_FIL.name}); results -> {OUT.name}")


def hdr(t): print(f"\n{'='*78}\n{t}\n{'='*78}", flush=True)

def _remount():
    try:
        from google.colab import drive; drive.mount("/content/gdrive", force_remount=True)
    except Exception:
        pass

def rjson(p, d=None):
    for _ in range(4):
        try:
            return json.load(open(p))
        except FileNotFoundError:
            return d
        except OSError as e:
            if e.errno in (107, 5, 103, 108):
                time.sleep(5); _remount(); continue
            return d
        except Exception:
            return d
    return d

def save(obj, p):
    tmp = Path(str(p) + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, p)

RES = rjson(OUT, {}) or {}
DRW = rjson(DRAWS, {}) or {}

def put(part, model, key, value, draws=None):
    RES.setdefault(part, {}).setdefault(model, {})[key] = value
    if draws is not None:
        DRW.setdefault(part, {}).setdefault(model, {})[key] = draws
    save(RES, OUT); save(DRW, DRAWS)

def have(part, model, key):
    return key in RES.get(part, {}).get(model, {})

pf = rjson(POINTFIX, {}) or {}
LAYER = {m: int(pf[m]["peak_layer"]) for m in MODELS if m in pf}
print("Table 1 layers (from the point-fix file):", LAYER)


# =============================================================================
# the probe — verbatim from pipeline/06 (Part A cv_accuracy, Part B pairing),
# with the PCA size a parameter (None = no PCA)
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
import warnings; warnings.filterwarnings("ignore")

def make_pipe(n, n_pca, seed, n_feat):
    steps = [("sc", StandardScaler())]
    if n_pca is not None:
        # Table 1 uses min(n_pca, n - 1). Also capped at the training-fold size
        # and the feature count so small subsets cannot fail; for every
        # Table-1-sized sample this is exactly n_pca.
        steps.append(("pca", PCA(n_components=min(n_pca, n - 1, n_feat, int(n * 0.8) - 1),
                                 random_state=seed)))
    steps.append(("clf", LogisticRegression(max_iter=MAX_ITER, random_state=seed)))
    return Pipeline(steps)

def cv_accuracy(Xp, Xn, gp, gn, seed, n_pca=50):
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
    pipe = make_pipe(len(y), n_pca, seed, X.shape[1])
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y, groups=g):
        pipe.fit(X[tr], y[tr])
        accs.append((pipe.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def paired_draws(Xr, Xf, gid_r, gid_f, n_draws, n_pca=50, base_seed=SEED, matched_only=False):
    """Part B's paired residual, keeping the matched and null terms of every
    draw: one balanced filler subsample per draw, used both as the matched
    negative class and, split at random, for the null."""
    out = []
    for d in range(n_draws):
        seed = base_seed + 10_000 + d
        rng = np.random.RandomState(seed)
        n = min(len(Xr), len(Xf))
        idx_r = rng.choice(len(Xr), n, False)
        idx_f = rng.choice(len(Xf), n, False)
        Xr_d, gr_d = Xr[idx_r], gid_r[idx_r]
        Xf_d, gf_d = Xf[idx_f], gid_f[idx_f]
        m = cv_accuracy(Xr_d, Xf_d, gr_d, gf_d, seed, n_pca)
        if matched_only:
            out.append((m, float("nan"))); continue
        p = np.random.RandomState(seed).permutation(len(Xf_d))
        h = len(Xf_d) // 2
        n_ = cv_accuracy(Xf_d[p[:h]], Xf_d[p[h:2*h]], gf_d[p[:h]], gf_d[p[h:2*h]], seed, n_pca)
        out.append((m, n_))
    return out

def summarize(draws):
    m = np.array([d[0] for d in draws]); n = np.array([d[1] for d in draws]); r = m - n
    q = lambda a: ([float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
                   if np.all(np.isfinite(a)) else None)
    has_null = bool(np.isfinite(n).any())
    return {"n_draws": len(draws),
            "matched": float(m.mean()), "matched_draw_range95": q(m),
            "null": float(np.nanmean(n)) if has_null else None, "null_draw_range95": q(n),
            "residual": float(np.nanmean(r)) if has_null else None, "residual_draw_range95": q(r)}

def fmt(s):
    if s["null"] is None:
        return f"matched {s['matched']:.4f}"
    lo, hi = s["residual_draw_range95"]
    return (f"matched {s['matched']:.4f}  null {s['null']:.4f}  residual {s['residual']:+.4f} "
            f"(draw range {lo:+.3f}..{hi:+.3f})")


# =============================================================================
# cache — one pass over the activation files, in Table 1's load order
# =============================================================================
def cache_path(model): return CACHE / f"{model}.pkl"

def build_cache(model):
    cp = cache_path(model)
    C = (pickle.load(open(cp, "rb")) if cp.exists()
         else {"done": {"revision": [], "filler": []}, "revision": [], "filler": []})
    L = LAYER[model]
    for arm, d, key, mkey in (("revision", ACT_REV / model, "revision_windows", "revision_meta"),
                              ("filler", ACT_FIL / model, "filler_windows", "filler_meta")):
        files = sorted(d.glob("*.npz"))                     # same order as Table 1's load_rev / load_fil
        done = set(C["done"][arm]); t0 = time.time()
        for k, p in enumerate(files):
            if p.name in done:
                continue
            W = None
            for _ in range(4):
                try:
                    with np.load(p) as z:
                        W = z[key] if key in z.files else None
                    break
                except OSError as e:
                    if e.errno in (107, 5, 103, 108):
                        print("    mount dropped, remounting", flush=True); time.sleep(5); _remount(); continue
                    raise
            meta = rjson(Path(str(p)[:-4] + "_meta.json"), {}) or {}
            rows = meta.get(mkey) or []
            if W is not None:
                for i in range(W.shape[0]):
                    C[arm].append({"pid": p.stem, "row": i,
                                   "token_idx": rows[i].get("token_idx") if i < len(rows) else None,
                                   "meta": rows[i] if i < len(rows) else {},
                                   "center": W[i, :, CENTER, :].astype(np.float16),   # every layer
                                   "window": W[i, L, :, :].astype(np.float16)})      # Table 1's layer
                del W
            C["done"][arm].append(p.name)
            if (k + 1) % 10 == 0 or k + 1 == len(files):
                pickle.dump(C, open(cp, "wb"), protocol=4)
                print(f"    {model[-10:]} {arm:<8} {k+1}/{len(files)} files, {len(C[arm])} events, "
                      f"{time.time()-t0:.0f}s", flush=True)
        gc.collect()
    pickle.dump(C, open(cp, "wb"), protocol=4)
    return C


# =============================================================================
# repetition per event — the B1 / B2 code, verbatim
# =============================================================================
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
_LP = {"Ġ": " ", "Ċ": "\n", "Ģ": "\n", "▁": " ", "ĉ": "\t"}
def clean_lp(t):
    s = t or ""
    for k, v in _LP.items(): s = s.replace(k, v)
    return s
_tr, _lb, _rj = {}, {}, {}
def trace(model, pid):
    k = (model, pid)
    if k not in _tr: _tr[k] = rjson(TRACES / model / f"{pid}.json")
    return _tr[k]
def labels_rows(model, pid):
    k = (model, pid)
    if k not in _lb:
        o = rjson(LABELS / model / f"{pid}.json")
        _lb[k] = [r for r in ((o or {}).get("revisions") or []) if r.get("verified", True)]
    return _lb[k]
def rejects_rows(model):
    if model not in _rj:
        by = defaultdict(list)
        for r in (rjson(REJECTS / f"{model}_confirmed_filler.json") or []):
            by[str(r.get("problem_id"))].append(r)
        _rj[model] = by
    return _rj[model]
def source_row(arm, model, pid, idx):
    rows = labels_rows(model, pid) if arm == "revision" else rejects_rows(model).get(str(pid), [])
    return rows[idx] if idx < len(rows) else None
def rep_at(txt, cs, ce, marker):
    if not txt or cs is None: return None
    cs = int(cs)
    if not (0 <= cs < len(txt)): return None
    mk = clean_lp(str(marker or "")).strip()
    if mk and txt[cs:cs + len(mk)].lower() != mk.lower(): return None
    if ce is None: ce = cs + max(len(mk), 1)
    b, a = extract_window(txt, cs, int(ce))
    return None if b is None else max_jaccard(b, a)
def analysis_text(tr):
    th = clean_lp((tr or {}).get("think_content") or "")
    cl = clean_lp((tr or {}).get("response_clean") or "")
    return th if th.strip() else cl
def repetition(arm, model, ev):
    e = source_row(arm, model, ev["pid"], ev["row"]); tr = trace(model, ev["pid"])
    if e is None or tr is None: return None
    mk = str(e.get("marker_phrase") or e.get("matched_text") or ev["meta"].get("matched_text") or "")
    texts = ((clean_lp(tr.get("full_response") or ""), analysis_text(tr)) if arm == "revision"
             else (analysis_text(tr), clean_lp(tr.get("full_response") or "")))
    for txt in texts:
        r = rep_at(txt, e.get("char_start"), e.get("char_end"), mk)
        if r is not None: return r
    return None


# =============================================================================
# align — where is the marker's first token inside each stored window?
# =============================================================================
def build_prompt(tr, tok):
    """The extractors' prompt (R1 models: no system prompt)."""
    instr = ("Solve the following competition mathematics problem carefully. "
             "Show your reasoning step by step and verify your answer."
             if tr.get("difficulty") == "hard" else
             "Solve the following mathematics competition problem. Show your working step by step.")
    msgs = [{"role": "user", "content": f"{instr}\n\nProblem: {tr['problem']}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def _clean_extractor(t):
    """clean_text from notebooks 02-05 (the text char_start and word_idx refer to)."""
    return t.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")

def extractor_text(tr):
    th = _clean_extractor(tr.get("think_content", "") or "")
    cl = _clean_extractor(tr.get("response_clean", "") or "")
    return th if th.strip() else cl

def align(model, C):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"deepseek-ai/{model}", trust_remote_code=True)
    enc_cache, stats, deltas = {}, Counter(), []
    for arm in ("revision", "filler"):
        for ev in C[arm]:
            ev["marker_win"] = None; ev["para_win"] = None
            e = source_row(arm, model, ev["pid"], ev["row"]); tr = trace(model, ev["pid"])
            if e is None or tr is None or ev["token_idx"] is None:
                stats[f"{arm}: no source row / trace / token_idx"] += 1; continue
            text = extractor_text(tr); cs = e.get("char_start")
            mk = clean_lp(str(e.get("matched_text") or e.get("marker_phrase") or "")).strip()
            if cs is None or not mk or text[int(cs):int(cs) + len(mk)].lower() != mk.lower():
                stats[f"{arm}: marker not at the stored offset"] += 1; continue
            if ev["pid"] not in enc_cache:
                prompt = build_prompt(tr, tok)
                enc = tok(prompt + text, return_offsets_mapping=True)
                enc_cache[ev["pid"]] = (len(prompt), list(enc["offset_mapping"]), list(enc["input_ids"]))
            plen_c, offs, ids = enc_cache[ev["pid"]]
            # check the cache row against the extractor: recompute its centre
            # (word_to_token in notebook 05 / word_idx_to_token_idx in 03)
            widx = e.get("revision_word_idx") if arm == "revision" else e.get("word_idx")
            if widx is not None:
                if ("plen", ev["pid"]) not in enc_cache:
                    enc_cache[("plen", ev["pid"])] = len(tok(build_prompt(tr, tok))["input_ids"])
                pref = " ".join(text.split()[:int(widx)])
                t_ex = enc_cache[("plen", ev["pid"])] + len(tok(pref, add_special_tokens=False)["input_ids"])
                t_ex = max(CENTER, min(t_ex, len(ids) - CENTER - 1))
                stats[f"{arm}: extractor centre recomputed {'==' if t_ex == int(ev['token_idx']) else '!='} stored"] += 1
            pos = plen_c + int(cs)
            t = next((i for i, (a, b) in enumerate(offs) if a <= pos < b and b > a), None)
            if t is None:
                stats[f"{arm}: no token covers the marker"] += 1; continue
            ev["marker_win"] = t - int(ev["token_idx"]) + CENTER
            deltas.append(int(ev["token_idx"]) - t)
            for k in range(1, 6):
                j = t - k
                if j >= 0 and "\n" in tok.decode([ids[j]]):
                    ev["para_win"] = j - int(ev["token_idx"]) + CENTER
                    break
            stats[f"{arm}: located"] += 1
    d = np.array(deltas)
    return {"counts": dict(stats),
            "stored_centre_minus_marker_first_token": None if not len(d) else {
                "n": int(len(d)), "median": float(np.median(d)),
                "share_exactly_-1": float(np.mean(d == -1)),
                "share_in_-3..+1": float(np.mean((d >= -3) & (d <= 1))),
                "min": int(d.min()), "max": int(d.max()),
                "most_common": {str(k): v for k, v in Counter(int(x) for x in d).most_common(12)}}}


# =============================================================================
# run
# =============================================================================
D_KEYS = (["stored_centre"] + [f"marker{o:+d}" for o in OFFSETS] + ["paragraph_break_before_marker"])
EXPECTED = {"1a": [f"pca_{n if n else 'none'}" for n in PCA_SETTINGS],
            "1b": ["profile_full", "select_on_A_evaluate_on_B", "select_on_B_evaluate_on_A"],
            "1c": ["joint"], "align": ["summary"], "1d": D_KEYS}

def missing(model):
    """(part, key) results not yet saved for this model, among the parts in PARTS."""
    return [(pt, k) for pt in ("1a", "1b", "1c", "align", "1d") if pt in PARTS
            for k in EXPECTED[pt] if not have(pt, model, k)]

for model in MODELS:
    if model not in LAYER:
        print(f"{model}: no Table 1 layer in {POINTFIX.name}; skipped"); continue
    L = LAYER[model]
    hdr(f"{model}   (Table 1 layer {L})")
    todo = missing(model)
    if not todo:
        print("  every result for this model is already saved; skipped (nothing is loaded or recomputed)")
        continue
    print(f"  {len(todo)} results still to compute: "
          + ", ".join(sorted({pt for pt, _ in todo})))
    C = build_cache(model) if ("cache" in PARTS or not cache_path(model).exists()) \
        else pickle.load(open(cache_path(model), "rb"))
    nr, nf = len(C["revision"]), len(C["filler"])
    print(f"  events: {nr} revision / {nf} filler")
    gid_r = np.arange(nr); gid_f = np.arange(nf) + 10**7       # Table 1's event ids

    # ---- 2a + 1a: PCA sensitivity ------------------------------------------
    if "1a" in PARTS:
        Xr = np.array([e["center"][L] for e in C["revision"]], np.float32)
        Xf = np.array([e["center"][L] for e in C["filler"]], np.float32)
        for n_pca in PCA_SETTINGS:
            key = f"pca_{n_pca if n_pca else 'none'}"
            if have("1a", model, key):
                print(f"  1a {key}: done"); continue
            t0 = time.time()
            nd = N_DRAWS_MAIN if n_pca else N_DRAWS_NOPCA
            if n_pca == 50 and model in pf:
                nd = int(pf[model].get("n_pt_draws", nd))     # same draws as Table 1 (100), so it must reproduce it
            dr = paired_draws(Xr, Xf, gid_r, gid_f, nd, n_pca)
            s = summarize(dr)
            if n_pca == 50 and model in pf:
                s["table1_residual_paired"] = pf[model].get("residual_paired")
                s["table1_minus_this"] = float(pf[model].get("residual_paired", float("nan")) - s["residual"])
                s["reproduces_table1_residual"] = bool(abs(s["table1_minus_this"]) < 1e-6)
                s["table1_matched_column_v3_unpaired"] = pf[model].get("matched")
                s["table1_null_column_v3_unpaired"] = pf[model].get("null_fil")
            put("1a", model, key, s, dr)
            print(f"  1a {key:<9} {fmt(s)}  [{time.time()-t0:.0f}s]", flush=True)
            if "reproduces_table1_residual" in s:
                print(f"       reproduces Table 1's residual ({s['table1_residual_paired']:+.6f}): "
                      f"{s['reproduces_table1_residual']}  (difference {s['table1_minus_this']:+.2e})")
        del Xr, Xf

    # ---- 1b: layer selection without the selection bonus -------------------
    if "1b" in PARTS:
        nl = C["revision"][0]["center"].shape[0]
        pids = sorted({e["pid"] for e in C["revision"]} | {e["pid"] for e in C["filler"]})
        sh = pids[:]; random.Random(SPLIT_SEED).shuffle(sh)
        half = {"A": set(sh[:len(sh)//2]), "B": set(sh[len(sh)//2:])}

        def XY(sel, layer):
            ir = [i for i, e in enumerate(C["revision"]) if sel is None or e["pid"] in sel]
            if_ = [i for i, e in enumerate(C["filler"]) if sel is None or e["pid"] in sel]
            Xr = np.array([C["revision"][i]["center"][layer] for i in ir], np.float32)
            Xf = np.array([C["filler"][i]["center"][layer] for i in if_], np.float32)
            return Xr, Xf, gid_r[ir], gid_f[if_]

        if not have("1b", model, "profile_full"):
            prof = [{"layer": l, **summarize(paired_draws(*XY(None, l), N_DRAWS_PROFILE))} for l in range(nl)]
            put("1b", model, "profile_full", prof)
            print(f"  1b residual by layer (full sample, {N_DRAWS_PROFILE} paired draws per layer):")
            print("     " + "  ".join(f"{p['layer']}:{p['residual']:+.3f}" for p in prof))
        for sel_h, eval_h in (("A", "B"), ("B", "A")):
            key = f"select_on_{sel_h}_evaluate_on_{eval_h}"
            if have("1b", model, key):
                print(f"  1b {key}: done"); continue
            accs = [summarize(paired_draws(*XY(half[sel_h], l), N_DRAWS_PROFILE, matched_only=True))["matched"]
                    for l in range(nl)]
            best = int(np.argmax(accs))
            dr = paired_draws(*XY(half[eval_h], best), N_DRAWS_MAIN)
            s = {"selected_layer": best, "table1_layer": L, "selection_matched_by_layer": accs,
                 "n_problems_select": len(half[sel_h]), "n_problems_evaluate": len(half[eval_h]),
                 **summarize(dr)}
            put("1b", model, key, s, dr)
            print(f"  1b choose on {sel_h} -> layer {best} (Table 1: {L}); on {eval_h}: {fmt(s)}", flush=True)

    # ---- 1c: repetition control on Table 1's sample -------------------------
    if "1c" in PARTS and not have("1c", model, "joint"):
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu
        rep_r = [repetition("revision", model, e) for e in C["revision"]]
        rep_f = [repetition("filler", model, e) for e in C["filler"]]
        okr = [i for i, v in enumerate(rep_r) if v is not None]
        okf = [i for i, v in enumerate(rep_f) if v is not None]
        print(f"  1c repetition located: {len(okr)}/{nr} revision, {len(okf)}/{nf} filler")
        Xr = np.array([C["revision"][i]["center"][L] for i in okr], np.float32)
        Xf = np.array([C["filler"][i]["center"][L] for i in okf], np.float32)
        Rr = np.array([rep_r[i] for i in okr]); Rf = np.array([rep_f[i] for i in okf])
        Pr = [C["revision"][i]["pid"] for i in okr]; Pf = [C["filler"][i]["pid"] for i in okf]

        def oof(XA, XB, gA, gB, seed):
            """One balanced draw, Table 1's pipeline, out-of-fold P(revision)."""
            n = min(len(XA), len(XB)); rng = np.random.RandomState(seed)
            ia = rng.choice(len(XA), n, False); ib = rng.choice(len(XB), n, False)
            X = np.nan_to_num(np.vstack([XA[ia], XB[ib]]).astype(np.float32))
            y = np.array([1]*n + [0]*n); g = np.concatenate([gA[ia], gB[ib]])
            sc = np.full(len(y), np.nan)
            pipe = make_pipe(len(y), 50, seed, X.shape[1])
            for tr_, te_ in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups=g):
                pipe.fit(X[tr_], y[tr_]); sc[te_] = pipe.predict_proba(X[te_])[:, 1]
            return ia, ib, y, sc

        def ls_fit(Y, X):
            b, *_ = np.linalg.lstsq(X, Y, rcond=None); return b

        def cluster_boot(Y, X, groups, nb=N_BOOT, seed=8080):
            beta = ls_fit(Y, X); rng = np.random.RandomState(seed); gi = defaultdict(list)
            for i, g in enumerate(groups): gi[g].append(i)
            arrs = [np.array(v) for v in gi.values()]; G = len(arrs)
            D = np.empty((nb, X.shape[1]))
            for k in range(nb):
                idx = np.concatenate([arrs[j] for j in rng.randint(0, G, G)])
                D[k] = ls_fit(Y[idx], X[idx])
            return beta, D, G

        def ci_p(col):
            lo, hi = np.percentile(col, [2.5, 97.5]); p = 2 * min((col <= 0).mean(), (col >= 0).mean())
            return float(lo), float(hi), float(min(1.0, max(p, 1.0 / len(col))))

        joint = []
        for d in range(N_DRAWS_JOINT):
            seed = SEED + 10_000 + d
            ia, ib, y, sc = oof(Xr, Xf, np.arange(len(Xr)), np.arange(len(Xf)) + 10**7, seed)
            rep = np.concatenate([Rr[ia], Rf[ib]]); grp = [Pr[i] for i in ia] + [Pf[i] for i in ib]
            X = np.column_stack([np.ones(len(y)), sc - sc.mean(), rep - rep.mean()])
            beta, D, G = cluster_boot(y.astype(float), X, grp)
            lo, hi, p = ci_p(D[:, 1]); lo2, hi2, p2 = ci_p(D[:, 2])
            joint.append({"draw": d, "n_per_arm": int(len(ia)), "clusters_problems": G,
                          "probe_beta": float(beta[1]), "probe_ci95": [lo, hi], "probe_p": p,
                          "rep_beta": float(beta[2]), "rep_ci95": [lo2, hi2], "rep_p": p2,
                          "probe_auc_oof": float(roc_auc_score(y, sc))})
        pb = np.array([j["probe_beta"] for j in joint])
        rep_auc = float(roc_auc_score(np.r_[np.ones(len(Rf)), np.zeros(len(Rr))], np.r_[Rf, Rr]))

        cal = {"found": False}      # B2's caliper rule: 0.5 SD, shrink x0.6, need >=100 pairs and balance p > 0.2
        tol = 0.5 * float(np.std(np.r_[Rr, Rf]) or 1.0)
        for _ in range(8):
            fs = np.argsort(Rf); fv = Rf[fs]; used = set(); pr_, pf_ = [], []
            for i in range(len(Rr)):
                j = int(np.searchsorted(fv, Rr[i])); best, bd = None, 1e18
                for k in range(max(0, j - 12), min(len(fs), j + 12)):
                    if fs[k] in used: continue
                    dd = abs(fv[k] - Rr[i])
                    if dd < bd: best, bd = fs[k], dd
                if best is not None and bd <= tol:
                    used.add(best); pr_.append(i); pf_.append(int(best))
            if len(pr_) >= 100:
                _, pbal = mannwhitneyu(Rr[pr_], Rf[pf_], alternative="two-sided")
                if pbal > 0.2:
                    _, _, y, sc = oof(Xr[pr_], Xf[pf_], np.arange(len(pr_)), np.arange(len(pf_)) + 10**7, SEED)
                    cal = {"found": True, "pairs": len(pr_), "balance_p": float(pbal), "caliper": float(tol),
                           "probe_auc_oof": float(roc_auc_score(y, sc))}
                    break
            tol *= 0.6
        s = {"layer": L, "n_rev_with_repetition": len(okr), "n_fil_with_repetition": len(okf),
             "repetition_auc_filler_vs_revision": rep_auc,
             "probe_beta_median_over_draws": float(np.median(pb)),
             "probe_beta_range_over_draws": [float(pb.min()), float(pb.max())],
             "draw0": joint[0], "caliper_matched": cal}
        put("1c", model, "joint", s, joint)
        j0 = joint[0]
        print(f"  1c probe_score {j0['probe_beta']:+.3f} [{j0['probe_ci95'][0]:+.3f}, {j0['probe_ci95'][1]:+.3f}] "
              f"p={j0['probe_p']:.3g} | repetition {j0['rep_beta']:+.3f} p={j0['rep_p']:.3g} | "
              f"{j0['clusters_problems']} problems | over {N_DRAWS_JOINT} draws: {pb.min():+.3f}..{pb.max():+.3f}")
        print(f"     repetition AUC (filler vs revision) {rep_auc:.3f}; caliper-matched: {cal}")

    # ---- align + 1d: positions relative to the marker -----------------------
    need_1d = "1d" in PARTS and any(pt == "1d" for pt, _ in todo)
    if "align" in PARTS and (not have("align", model, "summary")
                             or (need_1d and "marker_win" not in C["revision"][0])):
        # also re-run when the summary is saved but a rebuilt cache (new runtime)
        # lost the per-event marker positions that 1d needs
        summ = align(model, C)
        pickle.dump(C, open(cache_path(model), "wb"), protocol=4)
        if not have("align", model, "summary"):
            put("align", model, "summary", summ)
        print(f"  align: {summ['counts']}")
        print(f"  stored centre minus the marker's first token: {summ['stored_centre_minus_marker_first_token']}")
    if "1d" in PARTS:
        if "marker_win" not in C["revision"][0]:
            print("  1d needs the align stage first"); C = None; gc.collect(); continue
        positions = [("stored_centre", lambda e: CENTER)]
        positions += [(f"marker{o:+d}", (lambda o: lambda e: None if e.get("marker_win") is None
                                         else e["marker_win"] + o)(o)) for o in OFFSETS]
        positions += [("paragraph_break_before_marker", lambda e: e.get("para_win"))]
        for name, posf in positions:
            if have("1d", model, name):
                print(f"  1d {name}: done"); continue
            ok = lambda e: (posf(e) is not None) and 0 <= posf(e) < e["window"].shape[0]
            ir = [i for i, e in enumerate(C["revision"]) if ok(e)]
            if_ = [i for i, e in enumerate(C["filler"]) if ok(e)]
            if len(ir) < 20 or len(if_) < 20:
                put("1d", model, name, {"n_rev": len(ir), "n_fil": len(if_), "skipped": "fewer than 20 per arm"})
                print(f"  1d {name}: too few events ({len(ir)}/{len(if_)})"); continue
            Xr = np.array([C["revision"][i]["window"][posf(C["revision"][i])] for i in ir], np.float32)
            Xf = np.array([C["filler"][i]["window"][posf(C["filler"][i])] for i in if_], np.float32)
            dr = paired_draws(Xr, Xf, gid_r[ir], gid_f[if_], N_DRAWS_OFFSET)
            s = {"n_rev": len(ir), "n_fil": len(if_),
                 "label": "after the marker starts" if name.startswith("marker+") and name != "marker+0" else "",
                 **summarize(dr)}
            put("1d", model, name, s, dr)
            print(f"  1d {name:<30} n {len(ir)}/{len(if_)}  {fmt(s)}", flush=True)
    C = None; gc.collect()

print(f"\nResults: {OUT}\nPer-draw values: {DRAWS}")
