# =============================================================================
# SMOKE TEST — runs every notebook and script in this repository end to end on
# a tiny synthetic dataset, so you can see that the code executes before you
# spend GPU hours.
#
# What it does
#   1. Static checks: every notebook/script parses; no API keys, no personal
#      paths, no leftover code from the retired local/global analysis.
#   2. Builds a throwaway data folder (a temp directory — never your Drive) and
#      runs the pipeline IN ORDER on it, with the expensive or external parts
#      replaced by small local stand-ins:
#         models     -> a 4-layer, 16-dimensional random PyTorch model with a
#                       character-level tokenizer, generating canned reasoning
#                       text that contains revision markers and a \boxed answer
#         GPT-4o     -> deterministic fake verdicts
#         MATH       -> 105 synthetic problems
#         embeddings -> hashed bag-of-words vectors
#      Sizes are shrunk (bootstrap draws, problems, rollouts) by editing the
#      constants in a copy of each cell before it runs; the notebooks on disk
#      are never modified.
#   3. Prints PASS / WARN / FAIL per notebook, with the error for each failure.
#
# What it does NOT do
#   It does not check any number. With a random model and fake labels every
#   result is meaningless; the test only shows that each notebook runs from
#   its inputs to its outputs. It also cannot catch problems that only appear
#   with the real models (GPU memory, tokenizer specifics, real API errors).
#
# How to run
#   Colab (CPU runtime is enough):   !python smoke_test/run_smoke_test.py
#   Locally:                         python smoke_test/run_smoke_test.py
#   Needs: torch, numpy, scipy, scikit-learn, openpyxl. No GPU, no internet,
#   no Drive, no API key. About 5-15 minutes.
# =============================================================================
import ast, builtins, contextlib, hashlib, io, json, math, os, random, re, sys
import shutil, tempfile, time, traceback, types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
VERBOSE = "--verbose" in sys.argv
KEEP = "--keep" in sys.argv          # keep the temp data folder for inspection

# packages the repository's code needs to run here; the model, API and Colab
# libraries (transformers, datasets, sentence-transformers, openai, google.colab)
# are replaced by fakes, so they are not needed for the smoke test
_NEEDED = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "torch": "torch",
           "joblib": "joblib", "openpyxl": "openpyxl"}
_missing = []
for _mod, _pkg in _NEEDED.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print("The smoke test needs these packages, which are not installed in this Python:\n"
          f"    {' '.join(_missing)}\n"
          f"Install them, then run the smoke test again:\n"
          f"    {sys.executable} -m pip install {' '.join(_missing)}\n"
          "(or: pip install -r requirements.txt)")
    sys.exit(2)

import numpy as np
import torch
import torch.nn as nn

R1 = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-1.5B",
      "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-14B"]
INSTRUCT = "Llama-3.1-8B-Instruct"
FULL = {m: f"deepseek-ai/{m}" for m in R1}
FULL[INSTRUCT] = f"meta-llama/{INSTRUCT}"

RESULTS = []          # (name, status, detail, seconds)


# =============================================================================
# 1. stand-ins for external services
# =============================================================================
def h(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)


def answer_for(n):
    """Ground truth for synthetic problem n. Every 4th answer has nested
    braces, which the original answer regex truncates."""
    return f"\\frac{{1}}{{{n % 7 + 2}}}" if n % 4 == 0 else str(n % 97)


SENTENCES = [
    "We set up the equation for the quantity in question.",
    "Substituting the values gives an expression we can simplify.",
    "The sum of the first terms is easy to compute.",
    "We factor the polynomial and look at its roots.",
    "This gives a clean expression for the remaining part.",
    "Now we combine the two cases into a single count.",
    "The constraint fixes the last free parameter.",
    "We check the boundary case separately to be safe.",
]
MARKERS = [
    "Wait, let me check that again.", "Hmm, so the total should be smaller.",
    "Actually, that's wrong; I need a different approach.",
    "Alternatively, we can use symmetry here.", "Wait, actually the sign flips.",
    "Let me double-check the arithmetic.", "No, wait, the bound is strict.",
    "Actually, the same method still works.", "Hmm, let me think about this again.",
    "On second thought, the first case is empty.",
]


def reasoning(n, r1, long_, many):
    rng = random.Random(n * 7 + (3 if r1 else 1))
    parts = []
    target = 9000 if long_ else (5200 if many else 3000)
    k = 0
    while sum(len(p) + 1 for p in parts) < target:
        k += 1
        if r1 and (rng.random() < (0.75 if many else 0.25)):
            parts.append(rng.choice(MARKERS))
        parts.append(rng.choice(SENTENCES).replace("the ", f"the {k % 9 + 1} ", 1))
        if k % 6 == 0:
            parts.append("\n")
    return " ".join(parts)


def completion(model_name, prompt, r):
    """Fresh trace (prompt ends with the generation prompt) or a continuation."""
    m = re.search(r"item (\d+)", prompt)
    n = int(m.group(1)) if m else 0
    r1 = "R1" in model_name
    gt = answer_for(n)
    if prompt.endswith("<|assistant|><think>\n") or prompt.endswith("<|assistant|>"):
        long_ = (h(model_name + str(n)) % 9 == 0)           # some traces truncate
        many = (h(str(n)) % 7 == 0)                          # some exceed the 50 cap
        ok = (h(model_name + "c" + str(n)) % 10) < 7
        ans = gt if ok else str(n % 97 + 1)
        body = reasoning(n, r1, long_, many)
        tail = f"\n</think>\nThe answer is \\boxed{{{ans}}}." if r1 else f"\nTherefore the answer is \\boxed{{{ans}}}."
        return body + tail
    ok = (r if r is not None else 0.5) < 0.55
    ans = gt if ok else str(n % 97 + 3)
    return f" so the value follows from the last step.\n</think>\nThe answer is \\boxed{{{ans}}}."


class Enc(dict):
    def to(self, *a, **k):
        return self


class FakeTokenizer:
    def __init__(self, name):
        self.name = name
        self.bpe = "Llama-8B" in name and "R1" in name     # stored in byte-level characters, like the real data
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            s += "<|assistant|>" + ("<think>\n" if "R1" in self.name else "")
        return s

    def encode(self, text, add_special_tokens=False, **kw):
        return [ord(c) for c in text]

    def __call__(self, text, return_tensors=None, truncation=False, add_special_tokens=True, **kw):
        ids = [ord(c) for c in text]
        if kw.get("return_offsets_mapping") and return_tensors is None:
            return Enc(input_ids=ids, attention_mask=[1] * len(ids),
                       offset_mapping=[(i, i + 1) for i in range(len(ids))])
        if kw.get("return_offsets_mapping") and return_tensors == "pt":
            return Enc(input_ids=torch.tensor([ids], dtype=torch.long),
                       attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                       offset_mapping=torch.tensor([[(i, i + 1) for i in range(len(ids))]]))
        if return_tensors == "pt":
            return Enc(input_ids=torch.tensor([ids], dtype=torch.long),
                       attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        return Enc(input_ids=ids, attention_mask=[1] * len(ids))

    def convert_ids_to_tokens(self, ids):
        one = isinstance(ids, int)
        out = [chr(int(i)) if 0 <= int(i) < 0x110000 else "<unk>" for i in ([ids] if one else ids)]
        return out[0] if one else out

    def decode(self, ids, skip_special_tokens=False, **kw):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        out = "".join(("" if skip_special_tokens else "<eos>") if i == 0 else chr(i) for i in ids)
        return out.replace(" ", "Ġ").replace("\n", "Ċ") if self.bpe else out


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, *a, **k):
        return (x + torch.tanh(self.lin(x)),)


class FakeLM(nn.Module):
    def __init__(self, name, d=64, n_layers=4):
        super().__init__()
        torch.manual_seed(h(name) % 1000)
        self.name = name
        self.config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=d)
        self.emb = nn.Embedding(512, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = self.emb(input_ids % 512)
        for blk in self.model.layers:
            x = blk(x)[0]
        return SimpleNamespace(logits=x)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=128,
                 do_sample=False, **kw):
        self.forward(input_ids[:, -64:])                # runs any steering hook
        prompt = "".join(chr(i) for i in input_ids[0].tolist())
        r = float(torch.rand(1).item()) if do_sample else None
        new = [ord(c) for c in completion(self.name, prompt, r)][:max_new_tokens]
        if len(new) < max_new_tokens:
            new.append(0)
        return torch.tensor([input_ids[0].tolist() + new], dtype=torch.long)


def install_stubs():
    # transformers
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name, *a, **k: FakeTokenizer(name))
    tf.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda name, *a, **k: FakeLM(name))
    tf.BitsAndBytesConfig = lambda *a, **k: None
    sys.modules["transformers"] = tf

    # datasets
    ds = types.ModuleType("datasets")
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]

    def load_dataset(name, config=None, split="train", **kw):
        if split != "train":
            return []
        base = subjects.index(config) * 15
        rows = []
        for j in range(15):
            n = base + j
            rows.append({"problem": f"Compute the exact value for item {n} in the {config} set.",
                         "solution": f"Work through the steps. The result is $\\boxed{{{answer_for(n)}}}$.",
                         "level": f"Level {3 + n % 3}", "type": config.replace("_", " ").title()})
        return rows
    ds.load_dataset = load_dataset
    ds.concatenate_datasets = lambda parts: [r for p in parts for r in p]
    sys.modules["datasets"] = ds

    # google.colab
    g = types.ModuleType("google"); gc_ = types.ModuleType("google.colab")
    gc_.drive = SimpleNamespace(mount=lambda *a, **k: None)
    gc_.userdata = SimpleNamespace(get=lambda k: "smoke-test-key")
    gc_.runtime = SimpleNamespace(unassign=lambda: None)
    g.colab = gc_
    sys.modules["google"] = g; sys.modules["google.colab"] = gc_
    for sub in ("drive", "userdata", "runtime"):
        sys.modules[f"google.colab.{sub}"] = getattr(gc_, sub)

    # openai
    oa = types.ModuleType("openai")

    def create(model=None, messages=None, **kw):
        sysmsg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = messages[-1]["content"]
        k = h(user) % 10
        if "Reply with: OK" in user:
            content = "OK"
        elif "NOT_A_REVISION" in sysmsg:
            v = ["LOCAL", "GLOBAL", "LOCAL", "GLOBAL", "LOCAL", "LOCAL", "NOT_A_REVISION",
                 "NOT_A_REVISION", "NOT_A_REVISION", "NOT_A_REVISION"][k]
            content = json.dumps({"verdict": v, "confidence": 0.7 if k == 5 else 0.9, "reason": "smoke"})
        elif "DISCOURSE FILLER" in sysmsg:
            content = json.dumps({"verdict": "FILLER" if k < 7 else "REVISION", "confidence": 0.9})
        else:
            content = json.dumps({"verdict": "FILLER", "confidence": 0.5})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class OpenAI:
        def __init__(self, *a, **k):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    oa.OpenAI = OpenAI
    sys.modules["openai"] = oa

    # sentence-transformers
    st = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *a, **k): pass

        def encode(self, sentences, normalize_embeddings=True, **kw):
            single = isinstance(sentences, str)
            out = []
            for s in ([sentences] if single else sentences):
                v = np.zeros(32)
                for w in re.findall(r"[a-z0-9]+", s.lower()):
                    v[h(w) % 32] += 1.0
                v = v + 1e-3
                out.append(v / np.linalg.norm(v) if normalize_embeddings else v)
            out = np.array(out, dtype=np.float32)
            return out[0] if single else out
    st.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = st

    time.sleep = lambda *a, **k: None                   # no API back-off waits


# =============================================================================
# 2. running notebook code
# =============================================================================
def code_cells(path):
    if path.suffix == ".py":
        return [path.read_text(encoding="utf-8")]
    nb = json.load(open(path, encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        s = "\n".join("" if l.lstrip().startswith(("!", "%")) else l for l in s.splitlines())
        out.append(s)
    return out


def run(name, path, patches=(), ns_extra=None, expect_exit=False, cells=None):
    """Execute a notebook's code cells in order in one namespace. `patches` are
    (regex, replacement) pairs applied to the code; each must match at least
    once, so a renamed constant fails loudly instead of being silently ignored."""
    t0 = time.time()
    if cells is None and not Path(path).exists():
        RESULTS.append((name, "FAIL", f"file not found: {path}", 0))
        print(f"  FAIL  {name}  (file not found)", flush=True)
        return None
    srcs = cells if cells is not None else code_cells(path)
    hit = [0] * len(patches)
    out = []
    for s in srcs:
        for k, (pat, rep_) in enumerate(patches):
            s, n = re.subn(pat, rep_, s, flags=re.M)
            hit[k] += n
        out.append(s)
    missing = [patches[k][0] for k in range(len(patches)) if hit[k] == 0]
    if missing:
        RESULTS.append((name, "FAIL", f"smoke patch did not match (constant renamed?): {missing}", 0))
        return None
    ns = {"__name__": "__main__", "__file__": str(path)}
    ns.update(ns_extra or {})
    buf = io.StringIO()
    status, detail = "PASS", ""
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for i, s in enumerate(out):
                exec(compile(s, f"{path.name}[cell {i}]", "exec"), ns)
    except SystemExit as e:
        status = "WARN" if expect_exit else "FAIL"
        detail = f"SystemExit: {e}"
    except Exception as e:
        status = "FAIL"
        tb = traceback.format_exc().strip().splitlines()
        where = next((l.strip() for l in reversed(tb) if l.strip().startswith('File "') and path.name in l), "")
        detail = f"{type(e).__name__}: {e}  [{where}]"
    caught = [l.strip() for l in buf.getvalue().splitlines()
              if re.search(r"Traceback \(most recent call last\)|FAILED: |\[FAIL \d", l)]
    if status == "PASS" and caught:
        errs = [l.strip() for l in buf.getvalue().splitlines() if re.match(r"\s*\w+(Error|Exception):", l)]
        status = "FAIL"
        detail = (f"{len(caught)} error(s) caught and printed inside the notebook; first: "
                  f"{(errs or caught)[0][:200]}")
    if VERBOSE or status == "FAIL":
        tail = buf.getvalue().strip().splitlines()[-15:]
        detail += ("\n        last output:\n        " + "\n        ".join(tail)) if tail else ""
    RESULTS.append((name, status, detail, time.time() - t0))
    print(f"  {status:<5} {name}  ({time.time() - t0:.0f}s)", flush=True)
    return ns


# =============================================================================
# 3. static checks
# =============================================================================
def static_checks():
    print("\n[1] static checks")
    files = sorted(list(REPO.rglob("*.ipynb")) + list(REPO.rglob("*.py")))
    # only the repository's own code: skip virtual environments, installed
    # packages, git internals and any other hidden folder
    _skip = {"site-packages", "dist-packages", "node_modules", "__pycache__", "venv", "env"}
    files = [f for f in files
             if not any(part in _skip or part.startswith(".") for part in f.relative_to(REPO).parts[:-1])
             and "smoke_test" not in f.relative_to(REPO).parts]
    bad_parse, bad_terms, keys, paths = [], [], [], []
    term = re.compile("hier" + r"arch|scope_flag|SCOPE_ANALYSIS|kill_screen|Untitled10", re.I)
    for f in files:
        try:
            cells_ = code_cells(f) if f.suffix in (".ipynb", ".py") else []
        except Exception as e:                  # not UTF-8, not valid JSON, ...
            bad_parse.append(f"{f.relative_to(REPO)}: cannot be read ({type(e).__name__}: {str(e)[:80]})")
            cells_ = []
        for i, s in enumerate(cells_):
            try:
                ast.parse(s)
            except SyntaxError as e:
                bad_parse.append(f"{f.relative_to(REPO)}[{i}]: {e}")
        text = f.read_text(errors="replace")
        if term.search(text): bad_terms.append(str(f.relative_to(REPO)))
        if re.search(r"sk-[A-Za-z0-9_-]{20,}", text): keys.append(str(f.relative_to(REPO)))
        if re.search(r"/Users/[A-Za-z]|C:\\\\Users|/home/[a-z]+/", text): paths.append(str(f.relative_to(REPO)))
    ev_dir = REPO / "data" / "necessity_events"
    missing = [f for f in ("counterfactual_pilot_events.json", "filler_control_pilot_events.json",
                           "steering_pilot_events.json") if not (ev_dir / f).exists()]
    st = "PASS" if not missing else "WARN"
    RESULTS.append(("static: necessity event files in data/", st,
                    ("missing: " + ", ".join(missing)) if missing else "", 0))
    print(f"  {st:<5} necessity event files in data/" + (f"  -> missing {missing}" if missing else ""))
    for label, lst in (("all code parses", bad_parse), ("no retired local/global code", bad_terms),
                       ("no API keys", keys), ("no personal paths", paths)):
        st = "PASS" if not lst else "FAIL"
        RESULTS.append((f"static: {label}", st, "; ".join(lst[:6]), 0))
        print(f"  {st:<5} {label}" + ("" if not lst else f"  -> {lst[:6]}"))


# =============================================================================
# 4. the pipeline, in order
# =============================================================================
def model_line(full):
    return (r'^MODEL_NAME\s*=\s*"[^"]*"', f'MODEL_NAME = "{full}"')


def main():
    static_checks()
    install_stubs()
    tmp = Path(tempfile.mkdtemp(prefix="revision_smoke_"))
    data = tmp / "llm_planning_study"
    (data / "hand_labeling_export").mkdir(parents=True)
    os.environ["REVISION_DATA_ROOT"] = str(data)
    os.environ["P2_BASE"] = str(data)
    os.chdir(tmp)
    print(f"\n[2] pipeline on synthetic data in {tmp}")
    P = REPO / "pipeline"; A = REPO / "analyses"
    try:
        for m in R1 + [INSTRUCT]:
            run(f"01 generate traces [{m}]", P / "01_generate_traces.ipynb",
                [model_line(FULL[m]), (r"^N_HARD\s*=\s*200", "N_HARD = 14"), (r"^N_MEDIUM\s*=\s*100", "N_MEDIUM = 7")])
        for m in R1 + [INSTRUCT]:
            run(f"02 detect + verify revisions [{m}]", P / "02_detect_and_verify_revisions.ipynb", [model_line(m)])
        for m in R1:
            run(f"03 revision activations [{m}]", P / "03_extract_revision_activations.ipynb", [model_line(FULL[m])])
        run(f"03 baseline activations [{INSTRUCT}]", P / "03_extract_revision_activations.ipynb",
            [model_line(FULL[INSTRUCT]), (r"^IS_BASELINE_MODEL\s*=\s*False", "IS_BASELINE_MODEL = True"),
             (r"^N_BASELINE_WINDOWS\s*=\s*500", "N_BASELINE_WINDOWS = 10")])
        run("04 recover + confirm fillers", P / "04_recover_and_confirm_fillers.ipynb")
        for m in R1:
            run(f"05 filler activations [{m}]", P / "05_extract_filler_activations.ipynb", [model_line(FULL[m])])
        run("06 matched-marker probe (A+B)", P / "06_matched_marker_probe.ipynb",
            [(r"^N_BOOT\s*=\s*2000", "N_BOOT = 3"), (r"^N_PT_DRAWS\s*=\s*30", "N_PT_DRAWS = 2"),
             (r"^N_PT_DRAWS\s*=\s*100", "N_PT_DRAWS = 2")])
        os.environ["PROBE_CACHE"] = str(tmp / "probe_cache")
        probe_small = [(r"^N_DRAWS_MAIN\s*=\s*100", "N_DRAWS_MAIN = 3"), (r"^N_DRAWS_NOPCA\s*=\s*20", "N_DRAWS_NOPCA = 2"),
                       (r"^N_DRAWS_PROFILE\s*=\s*10", "N_DRAWS_PROFILE = 2"), (r"^N_DRAWS_OFFSET\s*=\s*30", "N_DRAWS_OFFSET = 2"),
                       (r"^N_DRAWS_JOINT\s*=\s*10", "N_DRAWS_JOINT = 2"), (r"^N_BOOT\s*=\s*2000", "N_BOOT = 50")]
        run("analyses: probe robustness checks (1a-1d)", A / "probe_robustness_checks.py", probe_small)
        ns = run("analyses: probe robustness checks (finished: re-run does nothing)", A / "probe_robustness_checks.py",
                 probe_small)
        stopped = "the probe script stopped before finishing; its own error is the FAIL line just above"
        if ns is None or "missing" not in ns:
            RESULTS.append(("probe checks: nothing left after a full run", "FAIL", stopped, 0))
        else:
            left = {m: ns["missing"](m) for m in ns["MODELS"]}
            RESULTS.append(("probe checks: nothing left after a full run",
                            "PASS" if not any(left.values()) else "FAIL", str({m: v for m, v in left.items() if v}), 0))
        # a new runtime mid-way: local cache gone, part of 1d not yet saved
        out_j = data / "hand_labeling_export" / "probe_robustness_checks.json"
        m0 = "DeepSeek-R1-Distill-Qwen-7B"
        try:
            res = json.load(open(out_j))
            for k in ["marker-1", "marker+0", "paragraph_break_before_marker"]:
                res["1d"][m0].pop(k, None)
            json.dump(res, open(out_j, "w"))
            shutil.rmtree(tmp / "probe_cache", ignore_errors=True)
            ns = run("analyses: probe robustness checks (resume on a new runtime)", A / "probe_robustness_checks.py",
                     probe_small)
            if ns is None or "missing" not in ns:
                RESULTS.append(("probe checks: resume on a new runtime completes 1d", "FAIL", stopped, 0))
            else:
                left = ns["missing"](m0)
                RESULTS.append(("probe checks: resume on a new runtime completes 1d",
                                "PASS" if not left else "FAIL", str(left), 0))
        except Exception as e:
            RESULTS.append(("probe checks: resume on a new runtime completes 1d", "FAIL",
                            f"no saved probe results to resume from ({type(e).__name__}: {e}); {stopped}", 0))
        run("analyses: Llama-8B window alignment diagnostic", A / "diagnose_llama8b_window_alignment.py")
        st = r'^STAGES = \["revision", "filler"\]'
        run("11 re-extract Llama-8B [revision, filler]", P / "11_reextract_llama8b.py")
        run("11 re-extract Llama-8B [verify]", P / "11_reextract_llama8b.py", [(st, 'STAGES = ["verify"]')])
        run("11 re-extract Llama-8B [probe]", P / "11_reextract_llama8b.py",
            [(st, 'STAGES = ["probe"]'), (r"^(\s*)N_BOOT\s*=\s*2000", r"\1N_BOOT = 3"),
             (r"^(\s*)N_PT_DRAWS\s*=\s*30", r"\1N_PT_DRAWS = 2"), (r"^(\s*)N_PT_DRAWS\s*=\s*100", r"\1N_PT_DRAWS = 2")])
        run("11 re-extract Llama-8B [revision, filler] again (resume: nothing to do)", P / "11_reextract_llama8b.py")
        run("analyses: probe robustness checks on the Llama-8B re-extraction", A / "probe_robustness_checks.py",
            probe_small + [(r'^ACTIVATIONS = "original"', 'ACTIVATIONS = "llama8b_reextract"')])
        run("07 repair ground truth", P / "07_repair_ground_truth.ipynb")

        n_ev = make_standin_events(data)
        print(f"        (stand-in necessity event files written: {n_ev})")

        run("08 steering directions + strength", P / "08_steering_directions_and_strength.ipynb",
            [(r"^ALPHA_MULTIPLIERS\s*=\s*\[[^\]]*\]", "ALPHA_MULTIPLIERS = [0.0, 1.0]")])

        # 09's original-sample select reads the published filler-arm event file (on
        # Drive in a real run); write a synthetic stand-in from the confirmed fillers.
        ex_ = data / "hand_labeling_export"
        if not (ex_ / "filler_control_pilot_events.json").exists():
            fevs = []
            for m in R1:
                for fl in json.load(open(data / "rejects" / f"{m}_confirmed_filler.json"))[:6]:
                    tr = json.load(open(data / "traces" / m / f"{fl['problem_id']}.json"))
                    if tr.get("truncated") or not tr.get("correct"): continue
                    fevs.append({"eid": len(fevs), "model": m, "problem_id": fl["problem_id"],
                                 "char_start": fl["char_start"], "marker_phrase": fl.get("matched_text"),
                                 "problem": tr["problem"], "difficulty": tr["difficulty"],
                                 "answer_gt": tr["answer_gt"], "round": 1 + len(fevs) % 2,
                                 "prefix_response_text": tr["full_response"][:fl["char_start"]]})
            json.dump(fevs, open(ex_ / "filler_control_pilot_events.json", "w"), indent=1)
            print(f"  synthetic filler events for 09: {len(fevs)}", flush=True)

        for stage in ("select", "lexical", "steering", "analyze"):
            if stage == "analyze":      # stand-in for the first steering re-run (seed 1000 + r)
                rd = data / "hand_labeling_export" / "redraw_2026_10" / "original_rerun_clean_prefix"
                for arm in ("revision", "filler"):
                    src = rd / f"redraw_steering_{arm}_rollouts_seed4000.json"
                    if src.exists():
                        shutil.copy(src, rd / f"redraw_steering_{arm}_rollouts.json")
            run(f"09 necessity re-run [{stage}]", P / "09_necessity_rerun.py",
                [(r'^STAGE\s*=\s*"select"', f'STAGE = "{stage}"'), (r"^N_BOOT\s*=\s*2000", "N_BOOT = 50")])
        for stage in ("baseline", "report"):
            run(f"analyses: steering efficacy on the re-run [{stage}]", A / "steering_efficacy_rerun.py",
                [(r'^STAGE\s*=\s*"baseline"', f'STAGE = "{stage}"')])
        run("11 re-extract Llama-8B [direction]", P / "11_reextract_llama8b.py", [(st, 'STAGES = ["direction"]')])
        for stage in ("steering", "analyze"):
            run(f"09 necessity re-run, Llama-8B rebuilt direction [{stage}]", P / "09_necessity_rerun.py",
                [(r'^STAGE\s*=\s*"select"', f'STAGE = "{stage}"'),
                 (r'^LLAMA_DIRECTION = "original"', 'LLAMA_DIRECTION = "reextract"')])
        run("analyses: steering efficacy with the rebuilt Llama-8B direction [report]",
            A / "steering_efficacy_rerun.py", [(r'^STAGE\s*=\s*"baseline"', 'STAGE = "report"')])
        os.environ["PROBE_N_JOBS"] = "2"
        sp_small = [(r"^N_DRAWS\s*=\s*100", "N_DRAWS = 3"), (r"^N_BOOT\s*=\s*2000", "N_BOOT = 6"),
                    (r"^N_DRAWS_LAYER\s*=\s*10", "N_DRAWS_LAYER = 2")]
        run("analyses: probe surface form and marker-token table", A / "probe_surface_and_position.py", sp_small)
        run("analyses: probe surface form and marker-token table (finished: re-run does nothing)",
            A / "probe_surface_and_position.py", sp_small)
        # the one-cell runner for the Nov 6 checks: embedded copies must be current
        if (P / "RUN_nov6_checks_all.py").exists():
            _src = (P / "RUN_nov6_checks_all.py").read_text()
            _ns = {}; exec(_src[_src.index("SRC = {}"):_src.index("\ndef patched(")], _ns)
            _nbsrc = "\n\n".join("".join(c["source"]) for c in json.load(open(A / "embedding_divergence_final.ipynb"))["cells"]
                                  if c["cell_type"] == "code")
            _same = {"necessity": _ns["SRC"]["necessity"] == (P / "09_necessity_rerun.py").read_text(),
                     "divergence": _ns["SRC"]["divergence"] == _nbsrc,
                     "probe": _ns["SRC"]["probe"] == (A / "probe_surface_and_position.py").read_text()}
            RESULTS.append(("RUN_nov6_checks_all embeds the current 09, divergence and probe code",
                            "PASS" if all(_same.values()) else "FAIL", str(_same), 0))
            for _f in ("divergence_final.json", "divergence_final_repaired_flag.json"):
                (data / "hand_labeling_export" / _f).unlink(missing_ok=True)
            run("RUN_nov6_checks_all (4 steps in one cell)", P / "RUN_nov6_checks_all.py")
            run("RUN_nov6_checks_all again (everything skips)", P / "RUN_nov6_checks_all.py")
        (data / "hand_labeling_export" / "steering_directions_llama8b_reextract.json").unlink(missing_ok=True)
        if (P / "RUN_llama8b_direction_all.py").exists():
            _src = (P / "RUN_llama8b_direction_all.py").read_text()
            _ns = {}; exec(_src[_src.index("SRC = {}"):_src.index("\ndef patched(")], _ns)
            _same = {k: _ns["SRC"][k] == f.read_text() for k, f in
                     [("reextract", P / "11_reextract_llama8b.py"), ("necessity", P / "09_necessity_rerun.py"),
                      ("efficacy", A / "steering_efficacy_rerun.py")]}
            RESULTS.append(("RUN_llama8b_direction_all embeds the current 11, 09 and efficacy scripts",
                            "PASS" if all(_same.values()) else "FAIL", str(_same), 0))
        if (P / "RUN_llama8b_direction_all.py").exists():
            run("RUN_llama8b_direction_all (4 steps in one file)", P / "RUN_llama8b_direction_all.py")
            run("RUN_llama8b_direction_all again (everything resumes / skips)", P / "RUN_llama8b_direction_all.py")
        for stage in ("select", "analyze"):
            run(f"09 necessity re-run, redraw sample [{stage}]", P / "09_necessity_rerun.py",
                [(r'^STAGE\s*=\s*"select"', f'STAGE = "{stage}"'), (r'^SAMPLE\s*=\s*"original"', 'SAMPLE = "redraw"'),
                 (r"^N_BOOT\s*=\s*2000", "N_BOOT = 50"), (r"^N_PAIRS\s*=\s*160", "N_PAIRS = 6")])

        cells = code_cells(P / "10_validate_gpt4o_labels.ipynb")
        split = next(i for i, s in enumerate(cells) if "SCORE THE GPT-4o" in s)
        pat = [(r"^N_PER_PILE\s*=\s*60", "N_PER_PILE = 8")]
        run("10 GPT-4o validation [part 1: export]", P / "10_validate_gpt4o_labels.ipynb", pat, cells=cells[:split])
        fill_workbook(data / "hand_labeling_export" / "gpt4o_validation_labeling.xlsx")
        run("10 GPT-4o validation [part 2: score]", P / "10_validate_gpt4o_labels.ipynb",
            [(r"^N_BOOT\s*=\s*10000", "N_BOOT = 200")], cells=cells[:1] + cells[split:])

        run("analyses: embedding divergence (4 parts)", A / "embedding_divergence.ipynb")
        run("analyses: divergence, final (paper numbers)", A / "embedding_divergence_final.ipynb",
            [(r"^N_BOOT\s*=\s*2000", "N_BOOT = 30")])
        run("analyses: divergence, final (re-run from saved per-event values)", A / "embedding_divergence_final.ipynb",
            [(r"^N_BOOT\s*=\s*2000", "N_BOOT = 30")])
        run("analyses: divergence, final, repaired correctness flag", A / "embedding_divergence_final.ipynb",
            [(r"^N_BOOT\s*=\s*2000", "N_BOOT = 30"), (r'^CORRECTNESS = "saved"', 'CORRECTNESS = "repaired"')])
        run("analyses: divergence regression + draw stability", A / "embedding_divergence_regression_and_draw_stability.ipynb",
            [(r"^N_BOOT\s*=\s*2000", "N_BOOT = 20"), (r"^N_BOOT\s*=\s*1500", "N_BOOT = 20"),
             (r"^N_DRAWS\s*=\s*40", "N_DRAWS = 3")])

    finally:
        os.chdir(REPO)
        if KEEP:
            print(f"\n  synthetic data kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    report()


def make_standin_events(data):
    """The necessity stage starts from two event files that are inputs to this
    repository (see README). Build small stand-ins with the same fields from the
    synthetic labels: verified revisions in correct, untruncated traces."""
    ex = data / "hand_labeling_export"
    gt_rep = json.load(open(ex / "repaired_ground_truth.json")) if (ex / "repaired_ground_truth.json").exists() else {}
    events, steer = [], []
    for m in R1:
        k = 0
        for lf in sorted((data / "labels" / m).glob("*.json")):
            if lf.name in ("labeling_log.json", "label_summary.json"): continue
            L = json.load(open(lf))
            if L.get("skipped"): continue
            tr = json.load(open(data / "traces" / m / f"{L['problem_id']}.json"))
            if tr.get("truncated") or not tr.get("correct"): continue
            for rv in L["revisions"][:2]:
                cs = rv["char_start"]
                ev = {"eid": len(events), "round": 1 + len(events) % 2, "model": m,
                      "problem_id": L["problem_id"], "human_label": rv["verdict"].lower(),
                      "char_start": cs, "char_end": rv["char_end"], "marker_phrase": rv["matched_text"],
                      "problem": tr["problem"], "difficulty": tr["difficulty"],
                      "answer_gt": tr["answer_gt"], "prefix_response_text": tr["full_response"][:cs],
                      "disambiguated_from_multiple": False, "disambiguation_unresolved": False}
                events.append(ev); k += 1
            if k >= 6: break
        steer += [e for e in events if e["model"] == m][:2]
    json.dump(events, open(ex / "counterfactual_pilot_events.json", "w"), indent=1)
    json.dump(steer, open(ex / "steering_pilot_events.json", "w"), indent=1)
    return len(events)


def fill_workbook(path):
    from openpyxl import load_workbook
    if not path.exists():
        return
    wb = load_workbook(path)
    ws = wb["Items"]
    for row in range(2, ws.max_row + 1):
        if ws.cell(row, 1).value is None: continue
        ws.cell(row, 3, ["REVISION", "FILLER", "REVISION", "FILLER", "UNCLEAR"][row % 5])
    wb.save(path)


def report():
    print("\n" + "=" * 78 + "\nSMOKE TEST SUMMARY\n" + "=" * 78)
    c = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for name, st, detail, sec in RESULTS:
        c[st] += 1
        if st != "PASS":
            print(f"  {st:<5} {name}\n        {detail}")
    print(f"\n  {c['PASS']} passed, {c['WARN']} warnings, {c['FAIL']} failed "
          f"(of {len(RESULTS)} checks)")
    print("  WARN = expected: the necessity event files not yet copied into data/.\n"
          "  FAIL = fix before a real run.")
    out = REPO / "smoke_test" / "last_report.txt"
    out.write_text("\n".join(f"{st}\t{name}\t{detail}" for name, st, detail, _ in RESULTS))
    print(f"  full report: {out}")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()