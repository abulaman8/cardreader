# Card-spotting benchmark (frozen)

This directory holds the fixed evaluation used to track progress across all model iterations.

| File | Role |
|---|---|
| `make_split.py` | Creates `split.json` **once**: 400 train / 50 val / 50 test. It refuses to overwrite an existing split. |
| `split.json` | The frozen game lists. Val and test are supersets of the v1 20/20 games. |
| `benchmark.py` | The frozen metric script. It's model-agnostic and imports no project code. |
| `results.csv` | One row appended per benchmark run. |

## Rules
1. **Never edit `benchmark.py` or `split.json`.** Both sha256 hashes are printed on every run and stored in `results.csv`. A row whose hashes differ from the reference hashes below is **not comparable**.
2. Models must write SoccerNet-format predictions to `<dir>/<game>/results_spotting.json`, for every val and every test game.
3. Include low-confidence peaks too (e.g. down to 0.01). The threshold-free AP and a-mAP need them; the thresholded metrics only use scores ≥ 0.70.
4. **Training and hard-negative mining must never use val or test games.**

## Metrics
- **Ground truth:** visible cards (Yellow, Red and Yellow->red merged). "Not shown" cards are ignored.
- **Primary tolerance:** ±5 s. ±1 s and ±2 s are reported for information only.
- **Fixed thresholds:** 0.70, 0.75, 0.80, 0.85, 0.90, 0.95.
- **Headline:**
  - Test F1 / P / R at ±5 s, at the threshold with the best val F1 at ±5 s.
  - Test AP at ±5 s.
  - Test loose a-mAP.

## Reference hashes
Fill these in from the first run's printout, and copy them into `changelog.md`.
```
benchmark.py sha256: 33a2a0f46327fbbd49de130f112f5219396170996f7644f63fb08cd99e49e09e
split.json   sha256: 7232ea9dc91dd1699208e93d56fe1271f823e48d44dd90f63ef01c690047c74f
```
