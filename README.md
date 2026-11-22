# Is It the Revision or the Word? Separating Revision Signals from Marker Words in Reasoning Traces

Code for the paper. Four R1-distilled models (DeepSeek-R1-Distill-Llama-8B, -Qwen-1.5B, -Qwen-7B, -Qwen-14B) and one non-reasoning control (Llama-3.1-8B-Instruct) solve 300 MATH problems. We then ask whether self-revision markers ("wait", "actually", "alternatively", …) carry a revision signal beyond the marker word itself. Three tests do this: a matched-marker probe, counterfactual necessity (lexical suppression and activation steering), and embedding divergence.

Everything runs in Google Colab. GPU stages were run on one A100 80 GB.

## Layout

```
pipeline/     the analysis, in run order (01 → 11)
analyses/     embedding divergence, probe robustness, steering efficacy, Llama-8B window diagnostic
smoke_test/   runs every notebook end to end on synthetic data (no GPU, no API, no Drive)
data/         the three necessity event files
requirements.txt
```

## Data folder

All notebooks read and write one folder on Google Drive, `DATA_ROOT`, set in the first cell of each notebook. The default is `/content/gdrive/MyDrive/llm_planning_study`; change it there, or set the environment variable `REVISION_DATA_ROOT`.

```
DATA_ROOT/
  dataset_summary.json               the 300 sampled problems (01)
  traces/{model}/{problem_id}.json   one greedy trace per problem (01)
  labels/{model}/{problem_id}.json   verified revisions (02)
  activations/{model}/               revision windows (03)
  rejects/{model}_*.json             rejected candidates and confirmed filler (04)
  activations_filler/{model}/        filler windows (05)
  activations_reextract_2026_11/     re-extracted Llama-8B revision windows (11)
  activations_filler_reextract_2026_11/  re-extracted Llama-8B filler windows (11)
  results_v3/                        probe results (06; *_llama8b_reextract.json from 11)
  hand_labeling_export/              everything else: ground truth, steering directions,
                                     necessity events and rollouts, validation workbook
```

## Run order

| # | Notebook | Compute | Writes | Used for |
|---|---|---|---|---|
| 01 | `pipeline/01_generate_traces.ipynb` | GPU, per model | `traces/`, `dataset_summary.json` | everything |
| 02 | `pipeline/02_detect_and_verify_revisions.ipynb` | OpenAI API, per model | `labels/` | revision counts and attrition; all revision events |
| 03 | `pipeline/03_extract_revision_activations.ipynb` | GPU, per model | `activations/` | probe and steering directions (Qwen models; Llama-8B comes from 11) |
| 04 | `pipeline/04_recover_and_confirm_fillers.ipynb` | CPU + OpenAI API | `rejects/` | filler counts; filler arm of every test |
| 05 | `pipeline/05_extract_filler_activations.ipynb` | GPU, per model | `activations_filler/` | probe, steering directions |
| 06 | `pipeline/06_matched_marker_probe.ipynb` | CPU, hours | `results_v3/task1_residual_*.json` | **probe table** (Part A: layer, CIs; Part B: residual), Qwen models |
| 07 | `pipeline/07_repair_ground_truth.ipynb` | CPU + internet | `repaired_ground_truth.json` | correctness for necessity |
| 08 | `pipeline/08_steering_directions_and_strength.ipynb` | CPU / GPU | `steering_directions.json`, calibration table | steering layer, direction and strength (Llama-8B direction: 11) |
| 09 | `pipeline/09_necessity_rerun.py` | GPU; measured 41.7 A100-h lexical, 19.3 A100-h steering (all four models) | `redraw_2026_10/original_rerun_clean_prefix/` | **necessity results** |
| 10 | `pipeline/10_validate_gpt4o_labels.ipynb` | CPU + one human labeling pass | validation workbook and results | GPT-4o precision and recall |
| 11 | `pipeline/11_reextract_llama8b.py` | GPU (`revision`, `filler`), then CPU (`verify`, `probe`, `direction`) | `activations_reextract_2026_11/`, `activations_filler_reextract_2026_11/`, `results_v3/*_llama8b_reextract.json`, `steering_directions_llama8b_reextract.json` | **Llama-8B probe row and steering direction** (the original Llama-8B windows could not be reproduced; see caveats) |
| — | `pipeline/RUN_llama8b_direction_all.py` | GPU, ~5 h | as 11 `direction`, 09 `steering` (`LLAMA_DIRECTION = "reextract"`), 09 `analyze`, efficacy `report` | runs those four steps in sequence, unattended |
| — | `analyses/embedding_divergence_final.ipynb` | CPU, ~40 min first run | `divergence_final.json` (`CORRECTNESS = "repaired"`: `divergence_final_repaired_flag.json`) | **divergence results reported in the paper**; the repaired-flag variant prints both side by side |
| — | `analyses/embedding_divergence.ipynb` | CPU | `divergence_*.json` | divergence three-arm control and audits |
| — | `analyses/embedding_divergence_regression_and_draw_stability.ipynb` | CPU | `divergence_*.json` | divergence regressions, draw stability |
| — | `analyses/probe_robustness_checks.py` | CPU, hours (first pass reads every activation file) | `probe_robustness_checks*.json`, `probe_robustness_draws*.json` | probe robustness: PCA size, held-out layer choice, repetition control, token position (`ACTIVATIONS = "llama8b_reextract"` for Llama-8B) |
| — | `pipeline/RUN_nov6_checks_all.py` | CPU | as the three rows it runs | one cell: 09 `analyze` (per-model CIs), divergence with the saved and the repaired flag, `probe_surface_and_position.py` |
| — | `analyses/probe_surface_and_position.py` | CPU, hours per model (one Colab runtime per model can run side by side) | `probe_surface_position/{model}.json` | marker surface form (first token, what precedes it); probe matched on surface form; sentence-initial markers only; the probe table at the marker's first token and the token before it, with CIs; held-out layer check there |
| — | `analyses/diagnose_llama8b_window_alignment.py` | CPU, tokenizers only | `llama8b_window_alignment.json` | why the original Llama-8B window positions could not be reproduced |
| — | `analyses/steering_efficacy_rerun.py` | GPU (`baseline`, 64 generations), then CPU (`report`) | `steering_efficacy_rerun.json` | steering efficacy on the re-run (Llama-8B with the rebuilt direction); compute time |

`09_necessity_rerun.py` is a script. Paste it into a Colab cell, or run it with `%run`. Set `STAGE` at the top and run the stages in this order: `select`, `lexical`, `steering`, `analyze`. Each stage resumes where it stopped.

**Full order for the reported numbers.** 01 → 08 as listed; 09 `select`, `lexical`, `steering`; 11 `revision`, `filler`, `verify`, `probe`, `direction`; 09 `steering` again with `LLAMA_DIRECTION = "reextract"` (Llama-8B only); 09 `analyze` (the paper's steering result is its STEERING_CORRECTED section); `analyses/steering_efficacy_rerun.py` `baseline`, then `report`; `analyses/probe_robustness_checks.py` with `ACTIVATIONS = "original"` and then `"llama8b_reextract"`. `pipeline/RUN_llama8b_direction_all.py` runs the four steps from 11 `direction` through the efficacy `report` in one go.

## Necessity event files

The necessity stage starts from three event files, included in `data/necessity_events/`. Copy them into `DATA_ROOT/hand_labeling_export/` before running notebook 08 or 09:

- `counterfactual_pilot_events.json`: **the 83 revision events.** 
- `filler_control_pilot_events.json`: the 80 filler events, matched 1:1 on (model, marker word) to the revision events. They were built by an earlier notebook that is not included in this repository; the resulting file is.
- `steering_pilot_events.json`: a small event set used only to calibrate the steering strength (notebook 08, Part B).


## Necessity: published vs re-run


`pipeline/09_necessity_rerun.py` re-runs **83 + 80 events**. The prefixes are cut from the trace and then decoded back to plain text. It also uses event-keyed resume and the production generator code embedded verbatim: 38 suppressed phrases with boundary variants, and the published per-model steering strengths. Rollout seeds are the published ones: `1000 + r` for lexical and `4000 + r` for steering (set in the calling loops of the earlier notebooks), so for the Qwen models old vs new is a replication check. A first steering re-run used `1000 + r` by mistake; its rollouts are kept (`redraw_steering_*_rollouts.json`) and `analyze` reports them separately as a second seed set, not as the procedure. Its `analyze` stage prints the new results beside the published ones: per model, pooled, and pooled without Llama-8B.

Results of the re-run (as_incorrect treatment; Mann-Whitney p; 95 % CI from a cluster bootstrap by problem):

| | Lexical | Steering (STEERING_CORRECTED) | Published lexical / steering |
|---|---|---|---|
| Pooled, 83 / 80 | +0.116, p = 0.0044, [+0.054, +0.182] | +0.107, p = 0.015, [+0.021, +0.205] | +0.099 / +0.101 |
| With repetition control | +0.126 [+0.058, +0.198] | +0.124 [+0.032, +0.226] | +0.112 / +0.118 |
| Excluding Llama-8B | +0.082, p = 0.056, [+0.011, +0.161] | +0.067, p = 0.13, [−0.033, +0.176] | +0.069 / +0.067 |

For the three Qwen models the re-run reproduces the published per-model numbers exactly (lexical: except Qwen-14B, which gains its previously missing event). STEERING_CORRECTED uses the published directions for the Qwen models and, for Llama-8B, the direction rebuilt from the re-extracted windows (see caveats). `analyze` also prints the earlier steering result with the original Llama-8B direction (+0.123) and the first steering re-run, made with the wrong seeds (+0.118), as a seed-robustness check.

## Reproducing each result

| Result | Run | Read |
|---|---|---|
| Probe table, matched-marker probe | Qwen: 06 Part A, then Part B. Llama-8B: 11 `probe` | `results_v3/task1_residual_pointfix.json` (Qwen) and `task1_residual_pointfix_llama8b_reextract.json` (Llama-8B): `residual_paired`, CIs. Matched and null from the same paired draws: `probe_robustness_checks*.json`, part `1a`, `pca_50` |
| Attrition and counts | 01, 02, 04 | `reason` field of `labels/{model}/*.json`; revisions per label file; `rejects/{model}_confirmed_filler.json` |
| Trace lengths, truncation | 01 | `response_tokens`, `truncated` in `traces/` |
| Steering layer, direction, strength | 08 (Llama-8B: 11 `direction`) | `steering_directions.json`, `steering_directions_llama8b_reextract.json`; k per model is in `CHOSEN_ALPHA_MULTIPLIER` in 09 |
| Necessity (lexical and steering), unparseable-answer treatments, leak-filter sensitivity, repetition-controlled estimate, per-model results with CIs | 07, then 09 (and 11 for the Llama-8B direction) | output of 09 `analyze`; `redraw_results.json` (`lexical`, `steering_corrected`; per model: `per_model_ci`) |
| GPT-4o precision and recall | 10 | `gpt4o_validation_results.json` |
| Embedding divergence: three arms on the 4× random pool, count-matched robustness, decomposition, repetition-controlled estimate (clustered by problem) | `analyses/embedding_divergence_final.ipynb` | `divergence_final.json` |
| Divergence audits, regressions, draw stability | the other two notebooks in `analyses/` | `divergence_*.json` |
| Probe robustness (one paired estimator for matched / null / residual; PCA 50, 100, 200, none; layer chosen on half the problems and tested on the other half; residual at every layer; `arm ~ probe score + repetition`; repetition-matched subsample; positions −5…+3 around the marker's first token and the paragraph break) | 06, then `analyses/probe_robustness_checks.py` | `probe_robustness_checks.json`; per-draw matched and null values in `probe_robustness_draws.json` |
| Steering efficacy (terms reduction) and compute time of the re-run | 09 `steering`, then `analyses/steering_efficacy_rerun.py` `baseline` and `report` | `steering_efficacy_rerun.json` (`llama8b_rebuilt_direction` for Llama-8B) |


## Known deviations and caveats

- **Candidate cap.** Notebook 02 caps candidates at 50 per trace. When a trace has more, it keeps the first 20, the last 10 and an *unseeded* random 20 from the middle; 22 traces were affected. Notebook 04 re-derives candidates with a plain first-50 cap. As a result, 104 of the 2,156 confirmed fillers, and 1 of the 80 necessity filler events, sit at positions notebook 02 may not have shown to GPT-4o. All of them were confirmed as filler by notebook 04's separate GPT-4o pass.
- **What text was labeled.** The R1 chat template already ends with `<think>`, so traces never contain the opening tag and `think_content` is empty. Detection and labeling therefore ran on `response_clean`, which is the reasoning plus the final answer. All necessity events, and all but 8 of 4,134 verified revisions, lie before `</think>`.
- **Byte-level characters in Llama-8B traces.** Every notebook that analyses text cleans them first (Ġ→space, Ċ→newline, ĉ→tab, length-preserving, so offsets stay valid). The re-run decodes necessity prefixes exactly.
- **Ground truth.** `answer_gt` in the traces was extracted with a regex that truncates nested braces (71 of 300 problems). Notebook 07 repairs it, and the necessity scoring uses the repaired values. The `correct` and saved `rollout_correct` flags were written before this fix. They are kept for the record but not used for any current number.
- **Answer normaliser.** It strips one leading `{` or one trailing `}`. This is load-bearing for answers the brace counter leaves wrapped; do not "simplify" it.
- **Where the probe window is centred.** Notebooks 03 and 05 convert the word index to a token index approximately: the words before it are re-joined with single spaces and tokenized on their own ("approximate but sufficient for ±30 token windows"). The word index is the word before the marker, so the window centre is near the first token of that word, not exactly the token before the marker. Measured with character offsets (`probe_robustness_checks.py`, stage `align`), the centre is a median 3–5 tokens before the marker's first token in all four models (exactly one token before it in 6–8 % of events). The same script runs the probe at fixed offsets from the true marker token.
- **Llama-8B window positions.** For the three Qwen models, recomputing the window centre from the traces, labels and tokenizer reproduces every stored index. For DeepSeek-R1-Distill-Llama-8B's original windows (`activations/`, `activations_filler/`) it reproduces 3 of 1,021 revision and 2 of 505 filler windows, under every text, cleaning and prompt variant tested (`analyses/diagnose_llama8b_window_alignment.py`). The original indices sit a median ~140 tokens before the recomputed ones. No original centre vector matches any position of its re-extracted ±30-token window (best cosine 0.95; median cosine with the re-extracted centre 0.31); for events whose original index falls inside that window, this means the original activations came from an input other than the current trace. Which input is unknown. The paper's Llama-8B probe row and steering direction therefore come from the re-extraction, whose windows reproduce exactly. `pipeline/11_reextract_llama8b.py` re-extracts Llama-8B with the current code into new folders (nothing is overwritten), checks the new windows, and re-runs the matched-marker probe on them; `analyses/probe_robustness_checks.py` runs on them with `ACTIVATIONS = "llama8b_reextract"`. Its `direction` stage rebuilds the Llama-8B steering direction from the new windows (notebook 08's code, new file `steering_directions_llama8b_reextract.json`); `09_necessity_rerun.py` with `LLAMA_DIRECTION = "reextract"` re-runs Llama-8B steering with it (published seeds and strength multiplier, separate files), and `analyze` then reports STEERING_CORRECTED (Qwen as before, Llama-8B with the rebuilt direction) beside the original.
- **Steering strength for Llama-8B.** The rebuilt direction keeps the published multiplier (1.5 × the direction's norm); the multiplier was not re-calibrated. On clean prefixes, steering reduces backtracking terms by +3.5 % for Llama-8B and +0.0 % for Qwen-7B (+31.0 % Qwen-1.5B, +50.5 % Qwen-14B), so for those two models the steering necessity numbers are not evidence that suppressing revision language mattered.
- **Unmatched layer sweep vs the matched probe.** Layer profiles in `analysis_results_all_models.json` come from an earlier unmatched probe (revision vs arbitrary control windows). The probe table uses the matched probe in notebook 06, whose per-layer profile is saved in Part A's output.
- **GPT-4o validation export.** Contexts for accepted items are rendered from `full_response` without cleaning byte-level characters, so 15 accepted-pile items were unreadable and labeled UNCLEAR. The export is unchanged so that it reproduces what was labeled.
- **Fixed during cleaning.**
  - Notebook 05 re-assigned `MODEL_NAME` inside the extraction cell, which overrode the setting cell.
  - Notebook 03's baseline path (Llama-3.1-8B-Instruct only) indexed the previous layer with an extra `[0]`. That made `baseline_norms` wrong and crashed on long traces. Those norms are not used in the paper.
  - Notebook 02 dropped a non-trace file with `traces.pop(300)`; it now filters by name, which gives the same 300 traces.
  - See `CHANGES.md`.
- **Software versions.** The versions used for the original runs were not recorded. The re-run (09) records its environment, GPU and wall-clock time in `manifest.json`.

## Smoke test

```
python smoke_test/run_smoke_test.py          # add --verbose for full output, --keep to keep the data
```

The smoke test runs every notebook and script in order (including the Llama-8B re-extraction and the one-file runner, and checks that the runner's embedded copies match the current scripts) on a throwaway synthetic dataset. Models are replaced by a small random PyTorch model, GPT-4o by deterministic fake verdicts, MATH by 105 synthetic problems and the sentence embedder by hashed vectors.  It reports PASS / WARN / FAIL for each notebook. It shows that the code runs, not that any number is right. If a required package is missing it stops first and prints the install command (`pip install -r requirements.txt`).

## Requirements

Python 3.10+. `pip install -r requirements.txt` installs `torch`, `transformers`, `datasets`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `sentence-transformers`, `openai` and `openpyxl`; the notebooks also use `accelerate` (preinstalled on Colab).
- **Llama-3.1-8B-Instruct** is gated on Hugging Face.
- **Notebooks 02 and 04** need `OPENAI_API_KEY` in Colab secrets.

## License

Code: MIT (see `LICENSE`). The MATH dataset and the DeepSeek-R1-Distill and Llama models are used under their own licenses.
