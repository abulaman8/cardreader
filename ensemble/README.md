# ensemble: late fusion + dual calibration

Model-agnostic post-processing over SoccerNet-format predictions. Like `benchmark/`, it reads only
`results_spotting.json` files and `Labels-v2.json`. It imports the frozen benchmark's GT loader and matcher
(`benchmark.benchmark.load_gt / load_preds / match`) so labels follow exactly the benchmark rules. It never
edits `benchmark/`.

Status: **code written, not run.**

## Why
Diagnostics on the existing val/test predictions (2026-10-14, see `changelog.md`):
- `v2_r2` and `tdeed_r2` each find ~138 of 228 test cards at 0.70, but different ones: union 161.
- Both models fire within ±5 s of 94-97% of cards. Recall is lost at the threshold, not at detection.
- The current calibration maps a score to P(this detection is real). On that scale the F1-optimal cut is
  roughly F1*/2 ≈ 0.37, below the benchmark grid floor of 0.70. Every run so far selected 0.70.

## What it does
1. **Platt per input run** (fitted on val): logistic on logit(raw score). Common probability scale, ranking kept.
2. **Fusion per half:** greedy clustering within ±5 s across runs. Score = Σ w_m · p_m (missing run = 0).
   Time = score-weighted mean. Weights: grid on the simplex (step 0.1), chosen by **val** AP@±5 s.
   Then NMS ±5 s.
3. **Two calibrators, both fitted on val only:**

| Variant | Calibrated score means | Benchmark run name |
|---|---|---|
| `marg` | P(this detection is a true card). Same meaning as all earlier rows. | `<name>_marg` |
| `cum` | Val precision of all detections scoring ≥ this one. Grid 0.70 = "≥70% of what you keep is real". | `<name>_cum` |

The benchmark is unchanged. Both variants are scored with the same frozen `benchmark.py` and logged as
separate rows.

## Usage
```bash
# ensemble of two runs
venv/bin/python -m ensemble.fuse --name ens_r2 --bench runs/v2_r2/bench runs/tdeed_r2/bench
venv/bin/python benchmark/benchmark.py --run ens_r2_marg \
    --val-pred runs/ens_r2/bench/calibrated_marg/val --test-pred runs/ens_r2/bench/calibrated_marg/test
venv/bin/python benchmark/benchmark.py --run ens_r2_cum \
    --val-pred runs/ens_r2/bench/calibrated_cum/val --test-pred runs/ens_r2/bench/calibrated_cum/test

# single run: recalibration only (isolates the calibration effect from the fusion effect)
venv/bin/python -m ensemble.fuse --name tdeed_r2_cal --bench runs/tdeed_r2/bench
```
Options: `--weights 0.6 0.4` (fixed weights, skip val search), `--tol 5`, `--nms-s 10`, `--out <dir>`.

Sanity check: for a single run, `<name>_marg` should reproduce that run's existing benchmark row. Isotonic
regression is invariant to the monotone Platt transform, and the input is already NMS'd at ±5 s.

## Output
`runs/<name>/bench/`:
- `fused/{val,test}/`: fused score before final calibration.
- `calibrated_marg/{val,test}/`, `calibrated_cum/{val,test}/`: benchmark inputs.
- `fuse.json`: inputs, Platt params, weight search, chosen weights, val AP, both calibrators.

## Leakage rules
- Test labels are never read. Platt, weights and both calibrators use val only.
- The benchmark threshold is still selected on val by `benchmark.py`.
