# =============================================================================
# PROBE: SURFACE FORM OF THE MARKER, AND THE PROBE AT THE MARKER TOKEN  (CPU)
#
# Windows: the three Qwen models use activations/ and activations_filler/;
# DeepSeek-R1-Distill-Llama-8B uses the re-extraction
# (activations_reextract_2026_11/, activations_filler_reextract_2026_11/),
# i.e. the windows behind the probe table. Layers: the probe table's
# (peak_layer in the point-fix files). Probe pipeline: notebook 06's
# (StandardScaler -> PCA-50 fit in each training fold -> logistic regression,
# StratifiedGroupKFold(5) grouped on event, balanced draws, paired null).
#
# Positions are relative to the marker's TRUE first token, found by
# re-tokenizing the extractor's exact prompt + text with character offsets
# ("align"). "marker+0" = the token containing the marker's first character;
# "marker-1" = the token before it. Position o is taken from the stored ±30
# window, so an event is used at o only if that token lies inside its window.
#
# PARTS (each result saved as soon as it is computed; re-running resumes and
# skips what is done):
#   check     stored centre, 100 paired draws: must reproduce the probe table's
#             residual exactly (confirms sample, pipeline and layer)
#   surface   per arm: the marker's first token (id, token string, decoded
#             text), what precedes it (paragraph break / line break / sentence
#             end / comma or other punctuation / word / start of text), whether
#             the first token also contains text before the marker, and how
#             many tokens the marker spans
#   matched   probe at stored centre, marker-1, marker+0 on a subsample matched
#             1:1 on (exact first-token id, preceding type) within the model;
#             100 paired draws; sizes and unmatched counts
#   initial   the same three positions, only events whose marker is
#             sentence-initial (preceded by paragraph break, line break or
#             sentence end), revisions vs fillers; 100 paired draws
#   table     the probe table at marker+0 and marker-1: matched / null /
#             residual from 100 paired draws, 95 % CIs from notebook 06's
#             bootstrap (B = 2000, events resampled with replacement, both
#             probes refitted per replicate, seeds SEED + b)
#   heldout   the held-out layer check at marker+0 (residual at every layer;
#             choose the layer on half the problems by matched accuracy,
#             evaluate on the other half with 100 paired draws; both ways)
#
# RUNNING IN PARALLEL. Results go to one file per model:
#   hand_labeling_export/probe_surface_position/{model}.json (+ _draws.json,
#   _events.json). Set MODELS to one model per Colab runtime to run them side
#   by side; nothing is shared between runs. Draws and bootstrap replicates
#   run on all CPU cores (N_JOBS); results do not depend on N_JOBS.
# =============================================================================
import gc, json, os, pickle, random, re, time, unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

MODELS  = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-1.5B",
           "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-14B"]
PARTS   = ["check", "surface", "matched", "initial", "table", "heldout"]
CENTER  = 30
SEED    = 42                 # the probe table's seed (paired draw d: SEED + 10000 + d; bootstrap b: SEED + b)
MAX_ITER = 2000
N_PCA   = 50
N_DRAWS = 100                # paired draws (the probe table's number)
N_BOOT  = 2000               # the probe table's bootstrap size (notebook 06)
N_DRAWS_LAYER = 10           # per layer, for the held-out profile and layer choice (as in the robustness checks)
SPLIT_SEED = 20261031        # held-out problem split (as in the robustness checks)
MATCH_SEED = 20261106        # which events are kept when a (token, preceding type) cell has more of one arm
N_JOBS  = int(os.environ.get("PROBE_N_JOBS", 0)) or max(1, os.cpu_count() or 1)
SENTENCE_INITIAL = ("paragraph break", "line break", "sentence end (. ? !)")

DATA_ROOT = Path(os.environ.get("REVISION_DATA_ROOT", "/content/gdrive/MyDrive/llm_planning_study"))
if not DATA_ROOT.exists():
    from google.colab import drive; drive.mount("/content/gdrive")
LABELS, REJECTS, TRACES = DATA_ROOT / "labels", DATA_ROOT / "rejects", DATA_ROOT / "traces"
EXPORT = DATA_ROOT / "hand_labeling_export"
OUTDIR = EXPORT / "probe_surface_position"; OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE  = Path(os.environ.get("PROBE_CACHE", "/content/probe_cache" if Path("/content").exists()
                             else str(DATA_ROOT.parent / "probe_cache"))) / "surface_position"
CACHE.mkdir(parents=True, exist_ok=True)
LLAMA = "DeepSeek-R1-Distill-Llama-8B"

def dirs_for(model):
    if model == LLAMA:
        return (DATA_ROOT / "activations_reextract_2026_11" / model,
                DATA_ROOT / "activations_filler_reextract_2026_11" / model,
                DATA_ROOT / "results_v3" / "task1_residual_pointfix_llama8b_reextract.json")
    return (DATA_ROOT / "activations" / model, DATA_ROOT / "activations_filler" / model,
            DATA_ROOT / "results_v3" / "task1_residual_pointfix.json")


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


# =============================================================================
# probe — notebook 06's cv_accuracy and Part B's pairing (as in
# analyses/probe_robustness_checks.py), draws run in parallel
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
import warnings; warnings.filterwarnings("ignore")

def make_pipe(n, seed, n_feat):
    return Pipeline([("sc", StandardScaler()),
                     # notebook 06: min(N_PCA, n - 1); also capped at the training-fold
                     # size and feature count so small subsets cannot fail (never binds
                     # at the probe table's sizes)
                     ("pca", PCA(n_components=min(N_PCA, n - 1, n_feat, int(n * 0.8) - 1), random_state=seed)),
                     ("clf", LogisticRegression(max_iter=MAX_ITER, random_state=seed))])

def cv_accuracy(Xp, Xn, gp, gn, seed):
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
    pipe = make_pipe(len(y), seed, X.shape[1])
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y, groups=g):
        pipe.fit(X[tr], y[tr])
        accs.append((pipe.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def _one_draw(Xr, Xf, gid_r, gid_f, d, matched_only):
    seed = SEED + 10_000 + d
    rng = np.random.RandomState(seed)
    n = min(len(Xr), len(Xf))
    idx_r = rng.choice(len(Xr), n, False); idx_f = rng.choice(len(Xf), n, False)
    Xr_d, gr_d, Xf_d, gf_d = Xr[idx_r], gid_r[idx_r], Xf[idx_f], gid_f[idx_f]
    m = cv_accuracy(Xr_d, Xf_d, gr_d, gf_d, seed)
    if matched_only:
        return (m, float("nan"))
    p = np.random.RandomState(seed).permutation(len(Xf_d)); h = len(Xf_d) // 2
    return (m, cv_accuracy(Xf_d[p[:h]], Xf_d[p[h:2*h]], gf_d[p[:h]], gf_d[p[h:2*h]], seed))

def paired_draws(Xr, Xf, gid_r, gid_f, n_draws=N_DRAWS, matched_only=False):
    return Parallel(n_jobs=N_JOBS)(delayed(_one_draw)(Xr, Xf, gid_r, gid_f, d, matched_only)
                                   for d in range(n_draws))

def _one_boot(Xr, Xf, gid_r, gid_f, b, ir, iff):
    """One replicate of notebook 06's bootstrap (Part A, bootstrap_all): events
    resampled with replacement (indices from boot_indices, the same stream as
    06), both probes refitted with seed SEED + b."""
    Xr_b, gr_b = Xr[ir], gid_r[ir]
    Xf_b, gf_b = Xf[iff], gid_f[iff]
    m_b = cv_accuracy(Xr_b, Xf_b, gr_b, gf_b, SEED + b)
    pf = np.random.RandomState(SEED + b).permutation(len(Xf_b)); h = len(Xf_b) // 2
    n_b = cv_accuracy(Xf_b[pf[:h]], Xf_b[pf[h:2*h]], gf_b[pf[:h]], gf_b[pf[h:2*h]], SEED + b)
    return (b, m_b, n_b)

def boot_indices(nr, nf, nb):
    """Exactly the index stream of notebook 06: one RandomState(SEED), and per
    replicate ir = choice(nr, nr, True) then iff = choice(nf, nf, True)."""
    rng = np.random.RandomState(SEED)
    return [(rng.choice(nr, nr, replace=True), rng.choice(nf, nf, replace=True)) for _ in range(nb)]

def summarize(draws):
    m = np.array([d[0] for d in draws]); n = np.array([d[1] for d in draws]); r = m - n
    q = lambda a: ([float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if np.all(np.isfinite(a)) else None)
    has_null = bool(np.isfinite(n).any())
    return {"n_draws": len(draws), "matched": float(m.mean()), "matched_draw_range95": q(m),
            "null": float(np.nanmean(n)) if has_null else None, "null_draw_range95": q(n),
            "residual": float(np.nanmean(r)) if has_null else None, "residual_draw_range95": q(r)}

def fmt(s):
    if s.get("skipped"): return f"skipped: {s['skipped']}"
    out = f"matched {s['matched']:.4f}"
    if s.get("null") is not None:
        out += f"  null {s['null']:.4f}  residual {s['residual']:+.4f}"
    if s.get("matched_ci"):
        out += (f"   95% CI matched [{s['matched_ci'][0]:.3f}, {s['matched_ci'][1]:.3f}]"
                f" null [{s['null_ci'][0]:.3f}, {s['null_ci'][1]:.3f}]"
                f" residual [{s['residual_ci'][0]:+.3f}, {s['residual_ci'][1]:+.3f}]")
    return out


# =============================================================================
# text side — the extractor's prompt and text, the label / reject rows
# =============================================================================
def _clean_extractor(t):
    """clean_text from notebooks 02-05 (the text char_start and word_idx refer to)."""
    return t.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")

def extractor_text(tr):
    th = _clean_extractor(tr.get("think_content", "") or "")
    cl = _clean_extractor(tr.get("response_clean", "") or "")
    return th if th.strip() else cl

def build_prompt(tr, tok):
    """The extractors' prompt (R1 models: no system prompt)."""
    instr = ("Solve the following competition mathematics problem carefully. "
             "Show your reasoning step by step and verify your answer."
             if tr.get("difficulty") == "hard" else
             "Solve the following mathematics competition problem. Show your working step by step.")
    msgs = [{"role": "user", "content": f"{instr}\n\nProblem: {tr['problem']}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

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

def preceding_type(text, cs):
    """What comes right before the marker, ignoring spaces and tabs."""
    m = text[:cs].rstrip(" \t")
    if not m:
        return "start of text"
    if re.search(r"\n[ \t]*\n$", m):
        return "paragraph break"
    if m.endswith("\n"):
        return "line break"
    c = m[-1]
    if c in ".?!":
        return "sentence end (. ? !)"
    if unicodedata.category(c)[0] in "PS":
        return "comma or other punctuation"
    return "word"

def extra_kind(s):
    if s == "": return "none"
    if s.strip(" ") == "": return "space(s) only"
    if s.strip(" \n\t") == "": return "whitespace incl. newline"
    return "punctuation or other text"


# =============================================================================
# events: align from metadata + traces (no activations), then one pass over
# the .npz files keeping the probe-table layer window and all layers at marker+0
# =============================================================================
def build_events(model, tok, rev_dir, fil_dir):
    ev = []
    stats = Counter()
    enc_cache = {}
    for arm, d, key in (("revision", rev_dir, "revision_meta"), ("filler", fil_dir, "filler_meta")):
        for p in sorted(d.glob("*.npz")):                       # the probe table's load order
            meta = rjson(Path(str(p)[:-4] + "_meta.json"), {}) or {}
            for i, row in enumerate(meta.get(key) or []):
                e = {"arm": arm, "pid": p.stem, "file": p.name, "row": i,
                     "token_idx": row.get("token_idx"), "marker_win": None}
                ev.append(e)
                src = source_row(arm, model, p.stem, i); tr = trace(model, p.stem)
                if src is None or tr is None or e["token_idx"] is None:
                    stats[f"{arm}: no source row / trace / token_idx"] += 1; continue
                text = extractor_text(tr); cs = src.get("char_start")
                mk = clean_lp(str(src.get("matched_text") or src.get("marker_phrase") or "")).strip()
                if cs is None or not mk or text[int(cs):int(cs) + len(mk)].lower() != mk.lower():
                    stats[f"{arm}: marker not at the stored offset"] += 1; continue
                if p.stem not in enc_cache:
                    prompt = build_prompt(tr, tok)
                    enc = tok(prompt + text, return_offsets_mapping=True)
                    enc_cache = {p.stem: (len(prompt), list(enc["offset_mapping"]), list(enc["input_ids"]),
                                          len(tok(prompt)["input_ids"]), prompt + text)}
                plen_c, offs, ids, plen, full_s = enc_cache[p.stem]
                pos = plen_c + int(cs)
                t = next((k for k, (a, b) in enumerate(offs) if a <= pos < b and b > a), None)
                if t is None:
                    stats[f"{arm}: no token covers the marker"] += 1; continue
                # the extractor's own centre must be the stored one (alignment check)
                widx = src.get("revision_word_idx") if arm == "revision" else src.get("word_idx")
                if widx is not None:
                    pref = " ".join(text.split()[:int(widx)])
                    t_ex = plen + len(tok(pref, add_special_tokens=False)["input_ids"])
                    t_ex = max(CENTER, min(t_ex, len(ids) - CENTER - 1))
                    stats[f"{arm}: extractor centre recomputed {'==' if t_ex == int(e['token_idx']) else '!='} stored"] += 1
                end = pos + len(mk)
                e.update({
                    "marker_win": t - int(e["token_idx"]) + CENTER,
                    "marker": mk,
                    "first_tok_id": int(ids[t]),
                    "first_tok_str": tok.convert_ids_to_tokens(int(ids[t])),
                    "first_tok_text": tok.decode([int(ids[t])]),
                    "first_tok_starts_before_marker": bool(offs[t][0] < pos),
                    "first_tok_extra_text": full_s[offs[t][0]:pos] if offs[t][0] < pos else "",
                    "n_marker_tokens": int(sum(1 for (a, b) in offs if b > a and a < end and b > pos)),
                    "preceding_type": preceding_type(text, int(cs)),
                    "preceding_char": repr(text[:int(cs)].rstrip(" \t")[-1:]) if text[:int(cs)].rstrip(" \t") else "''",
                })
                stats[f"{arm}: located"] += 1
    return ev, dict(stats)

_LP = {"Ġ": " ", "Ċ": "\n", "Ģ": "\n", "▁": " ", "ĉ": "\t"}
def clean_lp(t):
    s = t or ""
    for k, v in _LP.items(): s = s.replace(k, v)
    return s

def load_npz_key(path, key):
    for attempt in range(5):
        try:
            with np.load(path) as z:
                return z[key] if key in z.files else None
        except OSError as e:
            if getattr(e, "errno", None) in (107, 5, 103, 108) or "zip" in str(e).lower():
                print(f"    Drive read failed on {path.name}, retry {attempt+1}/5", flush=True)
                time.sleep(10 * (attempt + 1)); _remount(); continue
            raise
        except Exception as e:
            if "zip" in str(e).lower():
                print(f"    Drive read failed on {path.name}, retry {attempt+1}/5", flush=True)
                time.sleep(10 * (attempt + 1)); _remount(); continue
            raise
    raise RuntimeError(f"could not read {path} after 5 tries")

def build_cache(model, L, rev_dir, fil_dir, tok):
    cp = CACHE / f"{model}.pkl"
    C = pickle.load(open(cp, "rb")) if cp.exists() else None
    if C is None:
        t0 = time.time()
        ev, astats = build_events(model, tok, rev_dir, fil_dir)
        C = {"events": ev, "align_stats": astats, "done_files": [], "L": L}
        print(f"  align: {astats}  [{time.time()-t0:.0f}s]", flush=True)
        pickle.dump(C, open(cp, "wb"), protocol=4)
    by_file = defaultdict(list)
    for e in C["events"]:
        by_file[(e["arm"], e["file"])].append(e)
    done = set(map(tuple, C["done_files"])); t0 = time.time(); k = 0
    for (arm, fname), evs in by_file.items():
        if (arm, fname) in done: continue
        d, key = (rev_dir, "revision_windows") if arm == "revision" else (fil_dir, "filler_windows")
        W = load_npz_key(d / fname, key)
        for e in evs:
            if W is None or e["row"] >= W.shape[0]:
                e["missing_window"] = True; continue
            e["window"] = W[e["row"], L, :, :].astype(np.float16)
            mw = e.get("marker_win")
            e["pos0_all"] = (W[e["row"], :, mw, :].astype(np.float16)
                             if mw is not None and 0 <= mw < W.shape[2] else None)
        del W
        C["done_files"].append([arm, fname]); k += 1
        if k % 10 == 0:
            pickle.dump(C, open(cp, "wb"), protocol=4)
            print(f"    {model[-10:]} {k}/{len(by_file) - len(done)} files, {time.time()-t0:.0f}s", flush=True)
    pickle.dump(C, open(cp, "wb"), protocol=4)
    gc.collect()
    return C


# =============================================================================
# run
# =============================================================================
from transformers import AutoTokenizer

for model in MODELS:
    rev_dir, fil_dir, pf_path = dirs_for(model)
    pf = rjson(pf_path, {}) or {}
    if model not in pf:
        print(f"\n{model}: no probe-table layer in {pf_path.name}; skipped"); continue
    L = int(pf[model]["peak_layer"])
    OUT = OUTDIR / f"{model}.json"; DRW_P = OUTDIR / f"{model}_draws.json"; EV_P = OUTDIR / f"{model}_events.json"
    RES = rjson(OUT, {}) or {}; DRW = rjson(DRW_P, {}) or {}
    def put(part, key, value, draws=None):
        RES.setdefault(part, {})[key] = value
        if draws is not None: DRW.setdefault(part, {})[key] = draws
        save(RES, OUT); save(DRW, DRW_P)
    def have(part, key): return key in RES.get(part, {})
    EXPECT = {"check": ["stored_centre"], "surface": ["summary"],
              "matched": ["stored_centre", "marker-1", "marker+0"],
              "initial": ["stored_centre", "marker-1", "marker+0"],
              "table": ["marker+0", "marker-1"],
              "heldout": ["profile_marker+0", "select_on_A_evaluate_on_B", "select_on_B_evaluate_on_A"]}
    todo = [(pt, k) for pt in PARTS for k in EXPECT[pt] if not have(pt, k)]
    hdr(f"{model}   layer {L}   windows: {rev_dir.parent.name}, {fil_dir.parent.name}")
    if not todo:
        print("  every result for this model is already saved; skipped (nothing is loaded or recomputed)")
        continue
    print(f"  {len(todo)} results still to compute: {', '.join(sorted({p for p, _ in todo}))};  N_JOBS = {N_JOBS}")
    tok = AutoTokenizer.from_pretrained(f"deepseek-ai/{model}", trust_remote_code=True)
    C = build_cache(model, L, rev_dir, fil_dir, tok)
    E = [e for e in C["events"] if not e.get("missing_window")]
    R_ = [e for e in E if e["arm"] == "revision"]; F_ = [e for e in E if e["arm"] == "filler"]
    gid = {id(e): (i if e["arm"] == "revision" else i + 10**7)
           for i, e in enumerate(R_)} | {id(e): i + 10**7 for i, e in enumerate(F_)}
    print(f"  events: {len(R_)} revision / {len(F_)} filler;  align: {C['align_stats']}")
    save([{k: v for k, v in e.items() if k not in ("window", "pos0_all")} for e in E], EV_P)

    def feats(evs, posname):
        if posname == "stored_centre":
            return np.array([e["window"][CENTER] for e in evs], np.float32)
        o = int(posname.replace("marker", ""))
        return np.array([e["window"][e["marker_win"] + o] for e in evs], np.float32)
    def ok_at(e, posname):
        if posname == "stored_centre": return True
        o = int(posname.replace("marker", ""))
        return e.get("marker_win") is not None and 0 <= e["marker_win"] + o < e["window"].shape[0]
    def run_probe(revs, fils, posname, n_draws=N_DRAWS):
        Xr, Xf = feats(revs, posname), feats(fils, posname)
        gr = np.array([gid[id(e)] for e in revs]); gf = np.array([gid[id(e)] for e in fils])
        return paired_draws(Xr, Xf, gr, gf, n_draws), Xr, Xf, gr, gf

    # ---- check: the stored centre must reproduce the probe table -------------
    if "check" in PARTS and not have("check", "stored_centre"):
        t0 = time.time()
        dr, *_ = run_probe(R_, F_, "stored_centre", int(pf[model].get("n_pt_draws", N_DRAWS)))
        s = summarize(dr); s["probe_table_residual"] = pf[model].get("residual_paired")
        s["difference"] = float(s["probe_table_residual"] - s["residual"])
        s["reproduces"] = bool(abs(s["difference"]) < 1e-4)
        put("check", "stored_centre", s, dr)
        print(f"  check  stored centre {fmt(s)}  [{time.time()-t0:.0f}s]")
        print(f"         probe table residual {s['probe_table_residual']:+.6f}: reproduces {s['reproduces']} "
              f"(difference {s['difference']:+.2e})", flush=True)

    # ---- surface form ----------------------------------------------------------
    if "surface" in PARTS and not have("surface", "summary"):
        summ = {}
        for arm, evs in (("revision", R_), ("filler", F_)):
            loc = [e for e in evs if e.get("first_tok_id") is not None]
            n = len(loc)
            tokc = Counter((e["first_tok_id"], e["first_tok_str"], e["first_tok_text"]) for e in loc)
            prev = Counter(e["preceding_type"] for e in loc)
            summ[arm] = {
                "n_events": len(evs), "n_located": n,
                "n_distinct_first_tokens": len(tokc),
                "first_tokens_top20": [{"id": i, "token": s, "text": repr(t), "n": c, "share": c / n}
                                       for (i, s, t), c in tokc.most_common(20)],
                "preceding_type": {k: {"n": v, "share": v / n} for k, v in prev.most_common()},
                "preceding_char_top12": Counter(e["preceding_char"] for e in loc).most_common(12),
                "first_token_starts_before_marker": {k: {"n": v, "share": v / n} for k, v in
                                                     Counter(extra_kind(e["first_tok_extra_text"]) for e in loc).most_common()},
                "first_token_extra_text_top10": Counter(repr(e["first_tok_extra_text"]) for e in loc).most_common(10),
                "n_marker_tokens": dict(sorted(Counter(e["n_marker_tokens"] for e in loc).items())),
                "marker_in_window_at_0": sum(1 for e in loc if ok_at(e, "marker+0")),
                "marker_in_window_at_-1": sum(1 for e in loc if ok_at(e, "marker-1")),
            }
        put("surface", "summary", summ)
        for arm in ("revision", "filler"):
            s = summ[arm]
            print(f"\n  surface, {arm}: {s['n_located']}/{s['n_events']} located, "
                  f"{s['n_distinct_first_tokens']} distinct first tokens")
            print("    first token (id, token, text, share): " + "; ".join(
                f"{t['id']} {t['token']} {t['text']} {t['share']:.1%}" for t in s["first_tokens_top20"][:10]))
            print("    preceded by: " + "; ".join(f"{k} {v['share']:.1%}" for k, v in s["preceding_type"].items()))
            print("    first token also contains text before the marker: " + "; ".join(
                f"{k} {v['share']:.1%}" for k, v in s["first_token_starts_before_marker"].items()))
            print(f"    tokens per marker: {s['n_marker_tokens']}")

    # ---- matched on surface form -----------------------------------------------
    POS3 = ["stored_centre", "marker-1", "marker+0"]
    if "matched" in PARTS and any(not have("matched", k) for k in POS3):
        usable = [e for e in E if e.get("first_tok_id") is not None and all(ok_at(e, p) for p in POS3)]
        cells = defaultdict(lambda: ([], []))
        for e in usable:
            cells[(e["first_tok_id"], e["preceding_type"])][0 if e["arm"] == "revision" else 1].append(e)
        rng = np.random.RandomState(MATCH_SEED); mr, mf = [], []
        for k in sorted(cells, key=str):
            A, B = cells[k]; n = min(len(A), len(B))
            if n == 0: continue
            mr += [A[i] for i in rng.choice(len(A), n, replace=False)]
            mf += [B[i] for i in rng.choice(len(B), n, replace=False)]
        info = {"pairs": len(mr), "cells_matched": sum(1 for k in cells if min(map(len, cells[k])) > 0),
                "revision_usable": sum(1 for e in usable if e["arm"] == "revision"),
                "filler_usable": sum(1 for e in usable if e["arm"] == "filler"),
                "revision_unmatched": sum(1 for e in usable if e["arm"] == "revision") - len(mr),
                "filler_unmatched": sum(1 for e in usable if e["arm"] == "filler") - len(mf),
                "revision_not_usable": len(R_) - sum(1 for e in usable if e["arm"] == "revision"),
                "filler_not_usable": len(F_) - sum(1 for e in usable if e["arm"] == "filler")}
        print(f"\n  matched on (first-token id, preceding type): {info}")
        for pn in POS3:
            if have("matched", pn): continue
            t0 = time.time(); dr, *_ = run_probe(mr, mf, pn)
            s = {**info, **summarize(dr)}; put("matched", pn, s, dr)
            print(f"    {pn:<14} {fmt(s)}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- sentence-initial markers only -----------------------------------------
    if "initial" in PARTS and any(not have("initial", k) for k in POS3):
        ir = [e for e in R_ if e.get("preceding_type") in SENTENCE_INITIAL and all(ok_at(e, p) for p in POS3)]
        if_ = [e for e in F_ if e.get("preceding_type") in SENTENCE_INITIAL and all(ok_at(e, p) for p in POS3)]
        info = {"n_rev": len(ir), "n_fil": len(if_), "types": list(SENTENCE_INITIAL)}
        print(f"\n  sentence-initial markers only: {len(ir)} revision / {len(if_)} filler")
        for pn in POS3:
            if have("initial", pn): continue
            if min(len(ir), len(if_)) < 20:
                put("initial", pn, {**info, "skipped": "fewer than 20 events in an arm"}); continue
            t0 = time.time(); dr, *_ = run_probe(ir, if_, pn)
            s = {**info, **summarize(dr)}; put("initial", pn, s, dr)
            print(f"    {pn:<14} {fmt(s)}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- the probe table at marker+0 and marker-1 ------------------------------
    for pn in ("marker+0", "marker-1"):
        if "table" not in PARTS or have("table", pn): continue
        ir = [e for e in R_ if ok_at(e, pn)]; if_ = [e for e in F_ if ok_at(e, pn)]
        t0 = time.time()
        dr, Xr, Xf, gr, gf = run_probe(ir, if_, pn)
        point = summarize(dr)
        print(f"\n  table {pn}: n {len(ir)}/{len(if_)} (excluded: marker token outside the window or not "
              f"located: {len(R_)-len(ir)} revision, {len(F_)-len(if_)} filler)")
        print(f"    point estimates (100 paired draws): {fmt(point)}  [{time.time()-t0:.0f}s]", flush=True)
        prog_p = OUTDIR / f"{model}_bootstrap_{pn}.json"
        prog = rjson(prog_p, {}) or {}
        done_b = {int(k): v for k, v in prog.items()}
        BOOT_IDX = boot_indices(len(Xr), len(Xf), N_BOOT)
        todo_b = [b for b in range(N_BOOT) if b not in done_b]
        t0 = time.time()
        for i in range(0, len(todo_b), 100):
            chunk = todo_b[i:i + 100]
            for b, m_b, n_b in Parallel(n_jobs=N_JOBS)(delayed(_one_boot)(Xr, Xf, gr, gf, b, *BOOT_IDX[b])
                                                       for b in chunk):
                done_b[b] = [m_b, n_b]
            save({str(k): v for k, v in done_b.items()}, prog_p)
            print(f"      bootstrap {len(done_b)}/{N_BOOT}  [{time.time()-t0:.0f}s]", flush=True)
        mb = np.array([done_b[b][0] for b in range(N_BOOT)]); nb = np.array([done_b[b][1] for b in range(N_BOOT)])
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        s = {"n_rev": len(ir), "n_fil": len(if_), "excluded_rev": len(R_) - len(ir), "excluded_fil": len(F_) - len(if_),
             "layer": L, **point, "matched_ci": ci(mb), "null_ci": ci(nb), "residual_ci": ci(mb - nb),
             "n_boot": N_BOOT, "residual_ci_excludes_zero": bool(np.percentile(mb - nb, 2.5) > 0)}
        put("table", pn, s, {"paired_draws": dr})
        print(f"    {pn}: {fmt(s)}", flush=True)
        BOOT_IDX = None

    # ---- held-out layer check at marker+0 -----------------------------------------
    if "heldout" in PARTS and any(not have("heldout", k) for k in EXPECT["heldout"]):
        H = [e for e in E if e.get("pos0_all") is not None]
        Hr = [e for e in H if e["arm"] == "revision"]; Hf = [e for e in H if e["arm"] == "filler"]
        nl = Hr[0]["pos0_all"].shape[0]
        def XY(sel, layer):
            rr = [e for e in Hr if sel is None or e["pid"] in sel]; ff = [e for e in Hf if sel is None or e["pid"] in sel]
            return (np.array([e["pos0_all"][layer] for e in rr], np.float32),
                    np.array([e["pos0_all"][layer] for e in ff], np.float32),
                    np.array([gid[id(e)] for e in rr]), np.array([gid[id(e)] for e in ff]))
        if not have("heldout", "profile_marker+0"):
            t0 = time.time()
            prof = [{"layer": l, **summarize(paired_draws(*XY(None, l), N_DRAWS_LAYER))} for l in range(nl)]
            put("heldout", "profile_marker+0", {"n_rev": len(Hr), "n_fil": len(Hf), "profile": prof})
            print(f"\n  held-out, residual by layer at marker+0 ({N_DRAWS_LAYER} draws/layer, n {len(Hr)}/{len(Hf)}) "
                  f"[{time.time()-t0:.0f}s]:\n     " + "  ".join(f"{p['layer']}:{p['residual']:+.3f}" for p in prof))
        pids = sorted({e["pid"] for e in H}); sh = pids[:]; random.Random(SPLIT_SEED).shuffle(sh)
        half = {"A": set(sh[:len(sh)//2]), "B": set(sh[len(sh)//2:])}
        for a, b in (("A", "B"), ("B", "A")):
            key = f"select_on_{a}_evaluate_on_{b}"
            if have("heldout", key): continue
            t0 = time.time()
            accs = [summarize(paired_draws(*XY(half[a], l), N_DRAWS_LAYER, matched_only=True))["matched"] for l in range(nl)]
            best = int(np.argmax(accs)); dr = paired_draws(*XY(half[b], best), N_DRAWS)
            s = {"selected_layer": best, "probe_table_layer": L, "selection_matched_by_layer": accs, **summarize(dr)}
            put("heldout", key, s, dr)
            print(f"  held-out at marker+0: choose on {a} -> layer {best} (table: {L}); on {b}: {fmt(s)}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    C = None; E = R_ = F_ = None; gc.collect()

print(f"\nResults: {OUTDIR}/<model>.json  (per-draw values in <model>_draws.json, per-event surface data in <model>_events.json)")
