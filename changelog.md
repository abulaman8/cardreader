# Changelog

Every change and decision is recorded here, newest first. Each entry says **what** changed and **why**.

---

## 2026-10-08 — Initial pipeline (code only, not yet run)

Code is written but has **not been executed**. The GPU is busy with the `ballsaction` job. Run order is at the bottom.

### Task scope
- **Binary: card vs background.** Yellow, Red and Yellow->red are merged into one positive class.
  - *Why:* Red + Yellow->red is only ~100 samples (54 + 46) out of 1966 visible cards, which is too few for a reliable color classifier. The color distinction isn't needed for now.
- **Positives:** only cards with `visibility == "visible"` (1966 across 500 games).
  - *Why:* the 182 "not shown" cards have no visual evidence and would be label noise.
- **Displacement head: dropped** for now. It was discussed and validated as a plausible addition, but deferred. Revisit after the baseline transformer results.

### Data splits (held-out games)
- **Train:** official SoccerNet `train` split (300 games). Training clips (positives and negatives) are mined **only** from these games.
- **Val:** 20 games sampled (seed 0) from the official `valid` split. **Test:** 20 games sampled (seed 0) from the official `test` split.
- `src/index.py` asserts there's no overlap between train, val and test. The splits are written to `data/splits.json`.
  - *Why:* keeping eval to 40 games keeps evaluation time low. Game-level separation prevents leakage between clips of the same match.

### Clip construction
- **Sampling rate:** 10 fps (the source is 25 fps; the nearest frame is picked by timestamp).
- **Model window:** T = 50 frames (5 s). The card frame is jittered uniformly in [15, 34], which guarantees ≥1.5 s of context on each side.
  - *Why:* this is tight enough to cover the referee pulling out the card, without pulling in a foul that happened 10+ s earlier. The jitter stops the model from learning "card is always at the center", which would break sliding-window inference.
  - Planned ablations: 4 s (`--set data.window=40 data.margin=12`) and 3 s (`--set data.window=30 data.margin=10`).
- **Cache length:** each training clip stores 90 frames (9 s), and windows are cropped from it on the fly.
  - *Why:* one extraction supports jitter and every window-length ablation up to 9 s.
- **Negatives:** a pool of 15,000 train negatives, with a fresh 4:1 neg:pos mix drawn each epoch. The pool's composition:
  - 50% random live play.
  - 30% "foul": centered 0–10 s *after* a Foul / Direct free-kick / Indirect free-kick with no card nearby. *Why:* this teaches the model that a foul alone is not a card.
  - 20% "stoppage": Substitution / Offside / Kick-off / Throw-in ±2 s. *Why:* referee-centric shots and dead-ball close-ups look visually similar.
- **Negative exclusion zone: ±60 s** from any card, visible or not shown (was ±15 s in the first plan draft).
  - *Why:* broadcasts replay bookings after the event. A replay inside a negative would be a mislabeled positive. There's plenty of background footage to spare.
- **Val/test clip sets:** the same rules, sliced from full-half features, deterministic with seed 0.

### Features
- **Encoder:** frozen DINOv2 ViT-S/14 (`torch.hub facebookresearch/dinov2`, `dinov2_vits14`).
  - Input is the full 398×224 frame center-cropped to 392×224 (28×16 patches). No rescaling and almost no crop.
  - Per-frame feature: CLS token (384) concatenated with the mean patch token (384) = 768-d, stored as fp16.
  - *Why:* this is not a video backbone, by requirement. DINOv2 gives strong semantic/context features (referee close-up, raised arm, player reactions), which matters because the card itself is only a few pixels at 224p.
- **h-flipped features** are extracted for train clips (`[orig, flip]`) as free augmentation.
- **Frames are never written to disk.** The pipeline streams decode → encode → save. Writes are atomic and resumable.
- **Disk budget:** train positives ~0.33 GB, negative pool ~4.1 GB (90 frames × 2 variants), val+test full halves ~3.3 GB. Total ~8 GB of the 408 GB free.

### Targets and loss
- **Per-frame Gaussian smoothing:** σ = 10 frames (1 s), zeroed beyond 3σ, combined by max over cards.
  - *Why:* SoccerNet annotations carry ~±1 s timing noise. A one-hot target would be wrong and give sparse gradients. σ = 1 s also fits the tight-tolerance metric. Planned ablation: σ ∈ {0.5, 1, 2} s.
- **Clip target:** 1 if a visible card frame is inside the window.
- **Loss:** `BCE(frame logits, soft targets) + 0.5 · BCE(clip logit)`, using plain BCE, not focal.
  - *Why:* focal loss pushes probabilities down, so peaks might never reach the requested 0.70–0.95 confidence thresholds. Plain BCE keeps scores calibrated, and the 4:1 resampling handles the imbalance.

### Model (`src/model.py`)
- Pre-LN transformer encoder: d_model 256, 4 layers, **4 attention heads** (head dim 64), FFN 1024, dropout 0.2, attention dropout 0.1. Learned positional embeddings and a CLS token. ~3.3M params.
  - *Why:* this is sized for ~1.2k train positives. 64-d heads are the standard sweet spot; 8 heads would make them 32-d with no expected gain at this data scale.
- **Output heads:** per-frame confidence (T logits) and clip logit (on CLS).
- **Feature-level augmentation:** temporal jitter, h-flip (p = 0.5), 10% token masking with a learned mask token, and Gaussian noise σ = 0.05 after the input projection + LayerNorm.

### Training (`src/train.py`)
- AdamW with lr 3e-4 and weight decay 0.05 (no decay on norms, biases, pos/cls/mask tokens). 5% linear warmup, then cosine. Batch 128, 40 epochs, AMP fp16, grad clip 1.0.
- **A checkpoint is saved every epoch** (`runs/<name>/epoch_XXX.pt`) holding the model, config and that epoch's val scores.
  - Per-epoch metrics go to `log.jsonl`, a summary line to `train.log`, and scalars to TensorBoard (`tb/`).
  - `best.pt` is the best val tight-mAP. `last.pt` holds the optimizer/scheduler/scaler state for `--resume`.
- Early stopping: patience 10 epochs on val tight-mAP.

### Inference and evaluation
- **Sliding window** over full halves, stride 1 s. Overlapping per-frame probabilities are averaged.
  - Frames in the trained jitter range [margin, T-1-margin] get weight 1; window edges get weight 1e-3.
  - *Why:* the model only saw cards in that core range during training.
- **Peak picking:** greedy NMS with a 5 s window (±2.5 s). *Why:* it keeps two cards shown close together.
- **Metrics:**
  - SoccerNet-style tight a-mAP (δ = 1–5 s) and loose a-mAP (δ = 5–60 s), matching when |Δt| ≤ δ/2.
  - Clip-level ROC-AUC/AP, from both the clip head and the max frame probability.
  - "Not shown" cards are ignored: detections near them count as neither TP nor FP.
- **Confidence threshold grid:** 0.70 → 0.95 in steps of 0.05, at matching tolerances ±1 s, ±2 s and ±5 s. The threshold is picked by best val F1 at ±2 s, then applied once to test.
- **Temperature scaling** is fit on val clips. The final eval reports both raw and calibrated results.
- **Baseline:** logistic regression on mean-pooled clip features (`src/baseline.py`). The transformer must beat it.

### Environment
- **Packages are installed in the project venv only** (`venv/`), never system-wide.
- Fixed `torchaudio` 2.11.0, which was built for CUDA 13 and failed with `libcudart.so.13`, by replacing it with `torchaudio==2.6.0+cu124` to match `torch 2.6.0+cu124`. The driver is 550.163.01, which supports CUDA ≤ 12.4.
- Installed into the venv: `av`, `SoccerNet`, `scikit-learn`, `tensorboard`, `tqdm`, `pandas`, `pyyaml` (plus their dependencies). `pip check` is clean, and there are no cu13 packages.
- Regenerated `requirements.txt` from the venv with the cu124 extra index. The old file pinned incompatible CUDA 13 wheels.
- DINOv2 weights are downloaded by `torch.hub` on first use, so internet is needed once.

### Files added
`configs/default.yaml`, `src/{config,common,index,video,encoder,extract,dataset,model,spotting,train,evaluate,infer,baseline}.py`, `changelog.md`.

### Run order (after the `ballsaction` job finishes)
```bash
source venv/bin/activate
python -m src.index                        # splits + clip specs -> data/
python -m src.extract clips                # train features (~1h incl. decode)
python -m src.extract halves --split val test
python -m src.baseline
python -m src.train --name v1
python -m src.evaluate --ckpt runs/v1/best.pt
python -m src.infer --ckpt runs/v1/best.pt --split test
```
Val half features must exist before training. Per-epoch validation and `best.pt` selection depend on them.

---

## 2026-10-08 — v1 execution results

Pipeline executed end-to-end on RTX 4060:

### Execution summary
1. `src.index`: 300 train / 20 val / 20 test games indexed. 1,174 train pos, 15,000 train neg pool, 290 val clips, 505 test clips.
2. `src.extract clips`: extracted 16,172 clips (1,174 pos, 15,000 neg) with flipped copies -> 4.4 GB. 0 failed.
3. `src.extract halves`: extracted 80 full halves (40 val, 40 test) -> 3.3 GB. Total features cache = 7.4 GB.
4. `src.baseline`: LogisticRegression on mean-pooled features:
   - Val: AUC 0.9452, AP 0.8731
   - Test: AUC 0.9584, AP 0.8839
5. `src.train --name v1`: 23 epochs completed (~6s/epoch), early stopped at epoch 23 (patience 10).
   - Best checkpoint: `epoch_013.pt` (`runs/v1/best.pt`).
   - Val tight a-mAP: **0.5213**, loose a-mAP: **0.7489**.
   - Val best threshold: **0.75** (F1 0.6170, P 0.806, R 0.500 at ±2s).
6. `src.evaluate --ckpt runs/v1/best.pt`:
   - Calibrated temperature: T = 1.270.
   - Test tight a-mAP: **0.3984**, loose a-mAP: **0.6217**.
   - Test clip AUC: **0.9751**, clip AP: **0.9349** (baseline: 0.8839 AP). *Corrected: an earlier version of this entry said 0.9545, which was wrong.*
   - Test performance at selected threshold 0.70 (calibrated) / 0.75 (raw):
     - ±1s: P 0.527, R 0.287, F1 0.372 (29 TP, 26 FP, 72 FN)
     - ±2s: P 0.673, R 0.366, F1 0.474 (37 TP, 18 FP, 64 FN)
     - ±5s: P 0.782, R 0.426, F1 0.551 (43 TP, 12 FP, 58 FN)
7. `src.infer --ckpt runs/v1/best.pt --split test`:
   - Generated SoccerNet-format `results_spotting.json` for all 20 test games under `runs/v1/predictions/test/`.

---

## 2026-10-08 — v1 diagnostics and improvement recommendations

- Added `recommendations.md`: 5 recommendations to push toward F1 ≈ 0.9 with balanced P/R. No code changed.
- Diagnostics on v1, computed over all thresholds:
  - **Recall ceiling** (share of cards with any detection nearby): ±1 s 0.68–0.69, ±2 s 0.96–0.98, ±5 s 1.00.
  - **Best F1 at any threshold, on test:** ±1 s 0.419, ±2 s 0.540, ±5 s 0.615.
  - **Peak scores** near true cards cap at ~0.87, and ~15% of cards peak below 0.3.
  - **Overfitting:** val loss bottoms at epoch 4 and rises afterwards.
- Conclusion: the bottleneck is feature content (mean-pooling discards local evidence) and data quantity, not model capacity.

---

## 2026-10-08 — v2: end-to-end CNN, fixed benchmark, bigger splits (CODE WRITTEN, not yet run)

All changes below ship together as one change, not sequentially. The v1 code is kept, and v1 gets re-scored on the new benchmark as the reference.

### Implementation status
**Files added:**
- `benchmark/{make_split.py, benchmark.py, README.md}`
- `configs/v2.yaml`
- `src/v2/{__init__, common, index, frames, data, model, infer, train, mine, attention}.py`
- `src/rescore_v1.py`

None of it has been executed yet.

**Details settled during implementation:**
- **Train-time spotting** runs on the 20 `val_v1` games, with a 2 s stride, every 3 epochs and on the last epoch. *Why:* a full 50-game pass at a 1 s stride through the CNN costs ~30+ min. The full 50-game val is scored by `src.v2.infer` afterwards.
- **`best.pt` selection** uses raw-score AP at ±5 s on that subset. *Why:* it's threshold-free and independent of calibration.
- **Early stopping** counts spotting evaluations, not epochs: patience 4 × 3 epochs = 12 epochs.
- **Spatial attention starts as uniform pooling** (zero-initialized logits), so it begins like a mean pool and learns where to look from there.
- **Calibration** uses isotonic regression on val detections, fitted at ±5 s. It's stored as `calibrator.json` next to the predictions. `rescore_v1` applies the same calibration to v1.
- **Mining:** each half is cached to a temp directory, scored, then deleted. Expect roughly 1–1.5 min per half, so about 3–5 h per round of 100 games. Lower `mine.games_per_round` to shorten it.
- **Mined hard negatives** keep at most 20 per half, highest-scoring first.

**Run order:**
```bash
source venv/bin/activate
python benchmark/make_split.py                     # once; then record hashes from first benchmark run
python -m src.rescore_v1 --ckpt runs/v1/best.pt    # v1 reference (extracts 60 extra games' DINOv2 features)
python benchmark/benchmark.py --run v1 --val-pred runs/v1/bench/calibrated/val --test-pred runs/v1/bench/calibrated/test
python -m src.v2.index
python -m src.v2.frames clips
python -m src.v2.frames halves --split val test
python -m src.v2.train --name v2_r0
python -m src.v2.infer --ckpt runs/v2_r0/best.pt
python benchmark/benchmark.py --run v2_r0 --val-pred runs/v2_r0/bench/calibrated/val --test-pred runs/v2_r0/bench/calibrated/test
python -m src.v2.attention --ckpt runs/v2_r0/best.pt --n 30
python -m src.v2.mine --ckpt runs/v2_r0/best.pt --round 1
python -m src.v2.train --name v2_r1 --init runs/v2_r0/best.pt --mined data/v2/mined_round1.json
python -m src.v2.infer --ckpt runs/v2_r1/best.pt && python benchmark/benchmark.py --run v2_r1 --val-pred runs/v2_r1/bench/calibrated/val --test-pred runs/v2_r1/bench/calibrated/test
python -m src.v2.mine --ckpt runs/v2_r1/best.pt --round 2
python -m src.v2.train --name v2_r2 --init runs/v2_r1/best.pt --mined data/v2/mined_round1.json data/v2/mined_round2.json
```
If v2 training runs out of VRAM, add `--set model.grad_checkpoint=true` or `--set train.batch_size=6`.

### 1. Fixed benchmark (never changes across model iterations)
- **New `benchmark/` directory, frozen once created:**
  - `benchmark/split.json`: the fixed game lists.
  - `benchmark/benchmark.py`: the metric script.
  - `benchmark/README.md`: the rules.
- **Model-agnostic:** the script only reads SoccerNet-format `results_spotting.json` prediction files (time in seconds plus confidence), plus ground truth straight from `Labels-v2.json`. It never imports model code. Any model, at any fps, is scored the same way.
- **Ground truth:**
  - Every card with `visibility == "visible"` (Yellow, Red, Yellow->red merged).
  - "Not shown" cards are ignored: a detection near one counts as neither TP nor FP.
- **Primary tolerance: ±5 s** (match if |Δt| ≤ 5 s). One prediction matches at most one card, greedily by descending confidence.
  - ±1 s and ±2 s are reported for information only.
- **Fixed thresholds:** 0.70 to 0.95 in steps of 0.05, giving 6 values (user decision). A detection must score ≥ 0.70 to count, which is the "model is sure" bar.
  - Caveat: a raw score only means "sure" if it's calibrated. That's handled on the model side by peak calibration (see Inference). The benchmark itself just applies the fixed grid.
- **Headline numbers** (always the same):
  - **Test F1 / P / R at ±5 s,** at the threshold that maximizes **val** F1 at ±5 s, chosen from the fixed grid and applied once to test.
  - **Test AP at ±5 s and loose a-mAP (δ = 5–60 s)**, which are threshold-free. *Why:* they show ranking progress even when a model's scores sit below 0.70. This tracks the model separately from its calibration.
  - The full threshold × tolerance table for val and test.
- **Integrity:**
  - The script prints the sha256 of itself and of `split.json`.
  - Every run appends a row to `benchmark/results.csv` with run name, hashes and metrics.
  - A hash mismatch against the hashes recorded in this changelog marks the result **non-comparable**.
  - Hashes get recorded here when the benchmark is created.

### 2. New splits: 400 train / 50 val / 50 test
- **Val:** 50 games from the official `valid` split. **Test:** 50 games from the official `test` split.
  - Both are **supersets** of v1's 20 + 20 games, with the extra 30 + 30 sampled at seed 0, so v1-era numbers stay comparable on the 20-game subset.
- **Train:** official `train` (300) + the remaining 50 `valid` + 50 `test` games = **400 games**.
  - That's about +33% positives (~1.17k → ~1.56k visible cards).
- Training clips and hard-negative mining use the 400 train games **only**. The 100 benchmark games are never touched for training.
- *Why:*
  - More cards for evaluation: about 145 val and about 250 test cards (estimated from ~2.9 visible cards per game), against 58 / 101 before. That cuts F1 noise from roughly ±0.06 to roughly ±0.03.
  - More training data attacks v1's overfitting.
- Trade-off: no longer comparable with the official SoccerNet benchmark split. That's accepted.

### 3. Frame rate: 5 fps
- All clips and halves are sampled at **5 fps**. The 5 s window becomes **25 frames**.
- *Why:* the card gesture lasts ~1–2 s, which is 5–10 frames at 5 fps. That's enough for ±5 s evaluation, and it halves decode, storage and compute for the end-to-end model.

### 4. Architecture: end-to-end per-frame CNN + temporal transformer (recommendation 2B)
The frozen DINOv2 features are replaced by a CNN trained on raw pixels.

```
frames 25 x 224x398 (full frame, no crop) @ 5 fps
  -> RegNetY-400MF (ImageNet-pretrained, fine-tuned) + temporal shift (TSM, 1/8 channels) in each block
  -> feature map per frame: 440 ch x 7 x 13 (stride 32)
  -> learned spatial attention pooling (4 heads) -> 256-d per frame
  -> temporal transformer: 4 layers, 4 heads, d=256, + CLS
  -> heads: per-frame confidence | per-frame displacement | clip
```

- **Full frame, learned features:** the network sees every pixel at 224×398, and its filters are trained by the card loss itself. So it can learn "small yellow/red rectangle in a raised hand", which frozen DINOv2 was never taught.
- **Learned spatial attention pooling:**
  - At each of the 7×13 = 91 locations (each ~32×32 px), a small learned layer scores how relevant that location is.
  - Softmax over locations gives weights, and the frame vector is the weighted sum. 4 heads can each specialize, e.g. referee body, hand/arm, close-up face.
  - The weights are learned only from the card / no-card loss.
  - **Verification:** attention maps for val cards are saved as image overlays, so we can check that it looks at the referee and the card, not at the scoreboard or crowd.
- **Temporal shift (TSM):** zero-parameter, zero-FLOP channel shift between neighbouring frames. It gives the CNN short-range motion (hand moving up) before the transformer.
- **Displacement head (recommendation 5a):**
  - Confidence uses a Gaussian target, σ = 1 s = 5 frames.
  - Displacement target is d = (card − t) / R with R = 2 s = 10 frames. It's supervised only within ±R of a card, using a Huber loss.
  - At inference, frames vote for t + d, weighted by confidence.
- **DINOv2 patch-token idea (recommendation 1) is superseded.** Learned attention over the CNN's 91 locations does the same job with learned features, without a ~111 GB token cache.
- **Training defaults:**
  - Input: 5 s window (25 frames), card jittered within frames [7, 17].
  - Pixel augmentation: h-flip, colour jitter, random resized crop at 0.8–1.0 scale.
  - EMA of weights for stable checkpoint selection.
  - AMP, batch ~8 clips × 25 frames. Gradient checkpointing if VRAM requires it; to be verified by a smoke test.
  - Checkpoint every epoch, with its scores in `log.jsonl`, `train.log` and TensorBoard (same as v1).
  - Each epoch computes val clip metrics. Full-half val spotting runs every 3 epochs. *Why:* a full pass over 100 val halves at 5 fps through the CNN takes ~10 min.

### 5. Data pipeline: cached frames instead of cached features
- An end-to-end model needs pixels, so decoded frames are cached as **JPEG (quality 90) at 5 fps, 224×398**:
  - Train positives: 9 s windows (45 frames), ~1.56k clips ≈ 1.8 GB.
  - Train negative pool: ~20k clips × 45 frames ≈ 22 GB.
  - Benchmark halves: 100 games × 2 halves × ~13.5k frames ≈ 68 GB.
  - Total ≈ **92 GB** of the 400 GB free.
- Negatives keep the same rules: centre ≥ 60 s from any card. Mix is 50% random, 30% foul (0–10 s after the foul), 20% stoppage.
- **Class balance: 1 positive to 2.5 negatives per epoch** (user decision; v1 used 1:4). Each epoch draws ~1.56k positives and ~3.9k negatives.
  - Side effect: with fewer negatives, raw scores shift upward, so a raw 0.70 means something different than it did in v1. Peak calibration (see Inference) keeps the 0.70 bar meaningful.
- **Iterative hard-negative mining, 2 rounds:**
  1. Run the current model over the full halves of a rotating batch of 100 train games.
  2. Collect detections scoring ≥ 0.30 that are ≥ 60 s from any card.
  3. Cache them as clips and add them to the pool. Mined negatives are capped at 30% of each epoch's negatives.
  4. Fine-tune from the previous best.
  - Mining **only** uses train games.

### 6. Inference
- Sliding window at a 1 s stride (5 frames), with core-weighted averaging as in v1, plus displacement voting.
- **NMS window 10 s (±5 s), up from 5 s.** *Why:* this matches the ±5 s evaluation tolerance. Two detections 6–9 s apart around one card would otherwise produce one TP and one FP.
- **Peak calibration (new):** after NMS, each detection's raw peak score is mapped by an isotonic regression fitted on **val** detections to the probability that the detection is a true card within ±5 s. A calibrated 0.70 then literally means "~70% of detections at this score are real cards", which makes your 0.70 bar meaningful for every model version.
  - The calibration is fitted on val only; test stays untouched.
  - The calibrator ships with the checkpoint, and `results_spotting.json` carries calibrated confidences.
- Writes `results_spotting.json` for val and test, which `benchmark/benchmark.py` consumes.

### 7. Order of work (single change, implemented together)
1. Build `benchmark/` (split, script, README) and record its hashes in this changelog.
2. Re-score **v1** on the new 50/50 benchmark. This needs v1 features for the extra 60 games, ~45 min. It becomes the reference row.
3. Rebuild the index for 400 train games, then cache JPEG frames for clips and benchmark halves.
4. Write and smoke-test the v2 model (VRAM, speed), then train round 0.
5. Hard-negative mining rounds 1–2.
6. Score each round with the frozen benchmark and log every result here.

### Not doing
- **Higher-resolution / 720p / person crops:** ruled out (720p not available).
- **Bigger temporal transformer on frozen features:** v1 showed overfitting, not a capacity limit.
- **Flat-top targets (5b):** not adopted for now. Peak calibration addresses the v1 score cap (~0.87) relative to the 0.70 bar. Revisit if calibrated peaks still rarely reach 0.70.

---

## 2026-10-09 — v2_r0 benchmark results

Benchmark hashes verified:
- `benchmark.py`: `33a2a0f46327fbbd49de130f112f5219396170996f7644f63fb08cd99e49e09e`
- `split.json`: `7232ea9dc91dd1699208e93d56fe1271f823e48d44dd90f63ef01c690047c74f`

Trained 9 epochs; best checkpoint is epoch 3 (`runs/v2_r0/best.pt`).
Inference on 50 val + 50 test games (200 halves) evaluated with isotonic calibration fitted on val.

### Benchmark metrics:
- **Selected threshold (from Val @ ±5s F1):** 0.75
- **Val @ ±5s:** F1 = 0.6421 | Precision = 0.9143 | Recall = 0.4948 (96 TP, 9 FP, 98 FN, 194 GT)
  - Val AP5 = 0.6794 | Val loose a-mAP = 0.6952
- **Test @ ±5s (Headline):** F1 = **0.5900** | Precision = **0.9009** | Recall = **0.4386** (100 TP, 11 FP, 128 FN, 228 GT)
  - Test AP5 = **0.6588** | Test loose a-mAP = **0.6736**
- **Other tolerances on Test @ thr 0.75:**
  - ±1s: P = 0.8018, R = 0.3904, F1 = 0.5251 (89 TP, 22 FP, 139 FN)
  - ±2s: P = 0.8649, R = 0.4211, F1 = 0.5664 (96 TP, 15 FP, 132 FN)

### Analysis:
- Precision reaches target bar (90.1%), but recall is bottlenecked (43.9%).
- Post-NMS isotonic calibration ensures threshold 0.75 has high precision, but score overlap between true cards and background false alarms forces threshold to suppress true cards (128 FN).
- Solution: Hard-Negative Mining Round 1 (`v2_r1`) on 400 train games to push false alarm scores down and separate genuine cards.

---

## 2026-10-11 — v2_r1 benchmark results (Mining Round 1)

Benchmark hashes verified:
- `benchmark.py`: `33a2a0f46327fbbd49de130f112f5219396170996f7644f63fb08cd99e49e09e`
- `split.json`: `7232ea9dc91dd1699208e93d56fe1271f823e48d44dd90f63ef01c690047c74f`

Evaluated best checkpoint: Epoch 6 (`runs/v2_r1/best.pt`).
Inference on frozen 50 val + 50 test games (200 halves) evaluated with isotonic calibration fitted on val.

### Benchmark metrics:
- **Selected threshold (from Val @ ±5s F1):** 0.80
- **Val @ ±5s:** F1 = 0.6533 | Precision = 0.9250 | Recall = 0.5052 (98 TP, 8 FP, 96 FN, 194 GT)
  - Val AP5 = **0.7472** (vs 0.6794 in `v2_r0`, **+6.78%**)
  - Val loose a-mAP = **0.7686** (vs 0.6952 in `v2_r0`, **+7.34%**)
- **Test @ ±5s (Headline):** F1 = **0.6114** | Precision = **0.8770** | Recall = **0.4693** (107 TP, 15 FP, 121 FN, 228 GT)
  - Test AP5 = **0.6968** (vs 0.6588 in `v2_r0`, **+3.80%**)
  - Test loose a-mAP = **0.7114** (vs 0.6736 in `v2_r0`, **+3.78%**)
- **Other tolerances on Test @ thr 0.80:**
  - ±1s: P = 0.7623, R = 0.4079, F1 = 0.5314 (93 TP, 29 FP, 135 FN)
  - ±2s: P = 0.8279, R = 0.4430, F1 = 0.5771 (101 TP, 21 FP, 127 FN)

### Comparison (v2_r0 -> v2_r1):
- **Test F1:** $0.5900 \to \mathbf{0.6114}$ ($+2.14\%$ absolute gain)
- **Test AP5:** $0.6588 \to \mathbf{0.6968}$ ($+3.80\%$ gain)
- **Test loose a-mAP:** $0.6736 \to \mathbf{0.7114}$ ($+3.78\%$ gain)
- **True Positives:** $100 \to \mathbf{107}$ ($+7$ cards detected)
- **False Negatives:** $128 \to \mathbf{121}$ ($-7$ missed cards)
- **Analysis:** Round 1 hard-negative mining pushed background false alarms lower, unlocking +7 real card detections while maintaining high precision ($87.7\%$). Next step is Mining Round 2 (`v2_r2`) scanning the next rotating block of 100 train games.

---

## 2026-10-12 — T-DEED replica (CODE WRITTEN, not run)

New self-contained directory `tdeed/`. It imports nothing from `src/`, is scored by the frozen `benchmark/`, and has no benchmark changes. Details are in `tdeed/README.md`.

### Decisions (from design interview)
- **Frames:** 5 fps, 224x398. The existing `frames/eval/` val/test cache is reused read-only. New 40 s train windows go to `frames_tdeed/train/` (~33 GB). Model clip = 100 frames (20 s).
- **Backbone:** RegNetY-200MF (timm `regnety_002`, ImageNet init), with GSM in stages s3/s4.
- **Temporal:** SGP encoder / SGP-Mixer decoder: 2 levels, kernel 9, r=4, d=256.
- **Labels:** binary card vs background. T-DEED targets: ±4-frame card labels, frame displacement, CE fg weight 5, displacement MSE. "Not shown" cards are masked ±5 s.
- **Sampling:** every positive window plus 1.5x rule-based negatives (whole window ≥60 s from any card). No v2 mined negatives; T-DEED does its own mining in later rounds.
- **Training:** AdamW 8e-4 head / 2e-4 backbone, wd 0.01, 3-epoch warmup then cosine, 40 epochs, batch 4, AMP, EMA, T-DEED augmentations + mixup.
  - Spotting on `val_v1` every 3 epochs; `best.pt` = best AP@±5 s; patience 4.
- **Inference:** 50% overlapping windows, displacement voting, 10 s NMS, isotonic calibration on val.
  - *Why:* gives the benchmark's 0.70–0.95 grid a calibrated meaning, the same as v2.
- **New dependency:** `timm`, listed in `tdeed/requirements.txt`. torchvision has no RegNetY-200MF.

### Files
`tdeed/{__init__,config.yaml,common,index,extract,gsm,model,augment,data,spotting,infer,train,mine}.py`, `tdeed/README.md`, `tdeed/requirements.txt`.

---

## 2026-10-12 — v2_r2 execution results (Mining Round 2 + Bugfixes)

Full pipeline completed: 3,357 hard negatives mined across rounds 1 and 2, PyTorch 2.6/Python 3.13 stability fixes applied, training completed with early stopping at epoch 15 (best epoch 3), and evaluated against the frozen 50-game benchmark.

### Bugfixes Applied During Run
1. **EMA `state_dict()` loop segfault:** Pre-bound model and EMA tensor pairs during `EMA.__init__`, eliminating per-step state dict reconstructions.
2. **AdamW `_multi_tensor_adamw` `.item()` segfault:** Switched to `AdamW(fused=True)` on CUDA, eliminating hundreds of thousands of Python-level GPU-to-CPU synchronizations and list comprehensions.
3. **Gradient checkpointing generator crash:** Disabled `grad_checkpoint` in model configuration. Peak VRAM without checkpointing is 3.9 GB (fits comfortably in 8.2 GB RTX 4060), avoiding Python 3.13 generator frame deallocation crashes.
4. **File descriptor leak:** Used `with np.load(...) as z:` context manager in `JpegSeq.open_clip`.

### Frozen 50-Game Benchmark Evaluation
Evaluated with `benchmark/benchmark.py` (`sha256: 33a2a0f4...`) against `benchmark/split.json` (`sha256: 7232ea9d...`):

| Model | Val $F_1$ | Val AP5 | Test $F_1$ | Test P | Test R | Test AP5 | Test Loose a-mAP | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| `v2_r0` | 0.6421 | 0.6794 | 0.5900 | 0.9009 | 0.4386 | 0.6588 | 0.6736 | 100 | 11 | 128 |
| `v2_r1` | 0.6533 | 0.7472 | 0.6114 | 0.8770 | 0.4693 | 0.6968 | 0.7114 | 107 | 15 | 121 |
| **`v2_r2`** | **0.7273** | **0.7451** | **0.7092** | **0.8476** | **0.6096** | **0.7119** | **0.7272** | **139** | **25** | **89** |

### Gains vs `v2_r1`
- **$F_1$ Score:** $0.6114 \to \mathbf{0.7092}$ (**+0.0978**, almost +10 full F1 points)
- **Recall:** $46.9\% \to \mathbf{61.0\%}$ (**+14.0%**, +32 additional ground truth cards detected: 139 vs 107)
- **Precision:** **84.8%** maintained (only 25 false positives across 50 full 90-minute test games = 0.5 FP/game)
- **False Negatives:** $121 \to \mathbf{89}$ (**-32 missed cards**)
- **Test AP@±5s:** $0.6968 \to \mathbf{0.7119}$ (**+0.0151**)

---

## 2026-10-13 — tdeed_r0 execution & benchmark results (Round 0)

Full T-DEED Round 0 pipeline completed:
1. Built `data/tdeed/index.json` (400 train / 50 val / 50 test games; 1,544 positive 40s windows, 6,000 negative 40s windows, 485 val clips) and cached 7,544 train windows in `frames_tdeed/train/` (~33 GB).
2. Ported PyTorch 2.6 / Python 3.13 stability fixes to `tdeed/` (`EMA` pre-cached parameter pairs, `AdamW(fused=True)` + resume state handling, `use_reentrant=True` C++ autograd checkpointing in `tdeed/model.py` keeping VRAM at 6.62 GB, `with np.load(...)` in `JpegSeq.open_clip`, and CPU `libjpeg-turbo` decode in `decode_frames` to avoid `nvjpeg` CUDA error 716 on half #21).
3. Trained `tdeed_r0` for 24 epochs (early stopped with `patience: 4`; best checkpoint at **Epoch 12**, `runs/tdeed_r0/best.pt`, `val_v1` `spot/ap5 = 0.7519`, `F1@0.95 = 0.7312`).
4. Scored all 100 val halves + 100 test halves and evaluated with the frozen `benchmark/benchmark.py` (`sha256: 33a2a0f4...`) and `benchmark/split.json` (`sha256: 7232ea9d...`).

### Frozen 50-Game Benchmark Comparison

| Model | Mining Round | Sel. Thr | Val $F_1$ | Val AP5 | Test $F_1$ | Test P | Test R | Test AP5 | Test Loose a-mAP | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2_r0` | Round 0 | 0.75 | 0.6421 | 0.6794 | 0.5900 | 0.9009 | 0.4386 | 0.6588 | 0.6736 | 100 | 11 | 128 |
| `v2_r1` | Round 1 | 0.80 | 0.6533 | 0.7472 | 0.6114 | 0.8770 | 0.4693 | 0.6968 | 0.7114 | 107 | 15 | 121 |
| `v2_r2` | Round 2 | 0.70 | 0.7273 | 0.7451 | **0.7092** | 0.8476 | **0.6096** | 0.7119 | 0.7272 | **139** | 25 | **89** |
| **`tdeed_r0`** | **Round 0** | **0.85** | **0.6553** | **0.7566** | 0.5928 | **0.9340** | 0.4342 | **0.7255** | **0.7392** | 99 | **7** | 129 |

### Key Takeaways (`tdeed_r0`)
- **Best Ranking (AP@±5s & Loose a-mAP) Across All Models:** Even at Round 0 (zero hard-negative mining), `tdeed_r0` achieves **Val AP5 = 0.7566** (`+7.72%` vs `v2_r0`, `+1.15%` vs `v2_r2`) and **Test AP5 = 0.7255** (`+6.67%` vs `v2_r0`, `+1.36%` vs `v2_r2`; raw uncalibrated **Test AP5 = 0.7408**, **Loose a-mAP = 0.7548**).
- **Raw vs. Calibrated Operating Point:**
  - On **raw scores** (`thr = 0.95`), `tdeed_r0` already hits **Val $F_1 = 0.7219$** ($P = 0.8472, R = 0.6289$) and **Test $F_1 = 0.6801$** ($P = 0.7988, R = 0.5921$).
  - After isotonic calibration (`runs/tdeed_r0/bench/calibrator.json`), the marginal PAV bin `[0.9694, 0.9697]` lands at `y = 0.6923` (just `0.0077` below the `0.70` benchmark threshold grid floor) before jumping straight to `y = 0.8750` at `raw >= 0.9749`. Consequently, all grid thresholds `0.70–0.85` map to `raw >= 0.9739`, operating in an ultra-high-precision regime (**Test P = 93.4%**, only **7 FPs across 50 test games**, **R = 43.4%**, **F1 = 0.5928**).
- **Next Step (`tdeed_r1` / `tdeed_r2`):** Running T-DEED hard-negative mining (`python -m tdeed.mine`) on the 400 train games to suppress false alarms in the `0.95–0.97` raw score band will lift marginal calibration in that band above `0.70` and convert `tdeed_r0`'s higher AP5 (`0.7255` calibrated / `0.7408` raw) into higher calibrated $F_1$.

---

## 2026-10-13 — tdeed_r1 execution & benchmark results (Mining Round 1)

Full T-DEED Round 1 pipeline completed:
1. Mined 1,156 hard-negative 40s windows across rotating block of 100 train games (`data/tdeed/mined_round1.json`) using `tdeed/mine.py`.
2. Extracted all 1,156 windows into `frames_tdeed/train/mined/` in 40s.
3. Patched PyTorch 2.6 / Python 3.13 checkpoint serialization (`_to_cpu` detaching all saved tensors to CPU before pickle in `_atomic_save`, pre-importing `torch.utils.hooks.warn_if_has_hooks`, and running `gc.collect()` at epoch boundaries to stabilize RAM at 6.7 GB vs 34 GB).
4. Trained `tdeed_r1` initialized from `runs/tdeed_r0/best.pt` with 3,860 windows/epoch (1,544 pos + 1,621 rule neg + 695 mined neg). Cleanly early-stopped after Epoch 15 (`patience: 4`; best checkpoint at **Epoch 3**, `runs/tdeed_r1/best.pt`, `val_v1` `spot/ap5 = 0.7932`, `F1@0.95 = 0.7573`, `P = 0.867`, `R = 0.672`).
5. Scored all 100 val halves + 100 test halves and evaluated with the frozen `benchmark/benchmark.py` (`sha256: 33a2a0f4...`) and `benchmark/split.json` (`sha256: 7232ea9d...`).

### Frozen 50-Game Benchmark Comparison

| Model | Mining Round | Sel. Thr | Val $F_1$ | Val AP5 | Test $F_1$ | Test P | Test R | Test AP5 | Test Loose a-mAP | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2_r0` | Round 0 | 0.75 | 0.6421 | 0.6794 | 0.5900 | 0.9009 | 0.4386 | 0.6588 | 0.6736 | 100 | 11 | 128 |
| `v2_r1` | Round 1 | 0.80 | 0.6533 | 0.7472 | 0.6114 | 0.8770 | 0.4693 | 0.6968 | 0.7114 | 107 | 15 | 121 |
| `v2_r2` | Round 2 | 0.70 | 0.7273 | 0.7451 | **0.7092** | 0.8476 | **0.6096** | 0.7119 | 0.7272 | **139** | 25 | **89** |
| `tdeed_r0` | Round 0 | 0.85 | 0.6553 | 0.7566 | 0.5928 | **0.9340** | 0.4342 | 0.7255 | 0.7392 | 99 | **7** | 129 |
| **`tdeed_r1`** | **Round 1** | **0.70** | **0.7307** | **0.7638** | **0.6523** | **0.8462** | **0.5307** | **0.7336** | **0.7453** | **121** | 22 | **107** |

### Gains vs `tdeed_r0` and `v2_r1`:
- **$F_1$ Score Jump:** $0.5928 \to \mathbf{0.6523}$ (**+0.0595 / +6.0 F1 points** over `tdeed_r0`; beats `v2_r1`'s 0.6114 by **+4.09 F1 points**).
- **Cards Detected (Recall):** $43.4\% \to \mathbf{53.1\%}$ (**+22 additional ground-truth cards detected**: 121 vs 99; missed cards cut from 129 to 107).
- **Highest Ranking Metric Across All Models:** Test AP@±5s reaches **0.7336** and Test loose a-mAP reaches **0.7453** (uncalibrated raw Test AP5 = **0.7437**, raw F1@0.90 = **0.6853**).
- **False Alarm Suppression:** Total detections on val dropped from 11,162 to 7,454 (-3,708 background detections suppressed by hard negatives), lifting the calibrated 0.70 threshold into the high-recall band while preserving **84.6% Test Precision** (only 22 false alarms in 50 full test games = 0.44 FP/game).

---

## 2026-10-14 — tdeed_r2 execution & benchmark results (Mining Round 2, 8 Epoch Cap)

Full T-DEED Round 2 pipeline completed:
1. Mined 518 hard-negative 40s windows across rotating block of 100 train games (`data/tdeed/mined_round2.json`) using `tdeed/mine.py` (down from 1,156 in R1, showing massive dataset-wide false alarm suppression). Combined cumulative pool: 1,674 hard negatives across R1 + R2.
2. Extracted all 518 windows into `frames_tdeed/train/mined/` in 16s.
3. Trained `tdeed_r2` initialized from `runs/tdeed_r1/best.pt` with combined mined pools (`mined_round1.json` + `mined_round2.json`). Used user-specified training limits: `epochs: 8` and `patience: 2`. Best checkpoint achieved at **Epoch 6** (`runs/tdeed_r2/best.pt`, `val_v1` `spot/ap5 = 0.7840`, `loose = 0.7922`, `F1@0.95 = 0.7647`, `P = 0.886`, `R = 0.672`).
4. Scored all 100 val halves + 100 test halves and evaluated with the frozen `benchmark/benchmark.py` (`sha256: 33a2a0f4...`) and `benchmark/split.json` (`sha256: 7232ea9d...`).

### Frozen 50-Game Benchmark Comparison

| Model | Mining Round | Sel. Thr | Val $F_1$ | Val AP5 | Test $F_1$ | Test P | Test R | Test AP5 | Test Loose a-mAP | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2_r0` | Round 0 | 0.75 | 0.6421 | 0.6794 | 0.5900 | 0.9009 | 0.4386 | 0.6588 | 0.6736 | 100 | 11 | 128 |
| `v2_r1` | Round 1 | 0.80 | 0.6533 | 0.7472 | 0.6114 | 0.8770 | 0.4693 | 0.6968 | 0.7114 | 107 | 15 | 121 |
| `v2_r2` | Round 2 | 0.70 | 0.7273 | 0.7451 | **0.7092** | 0.8476 | **0.6096** | 0.7119 | 0.7272 | **139** | 25 | **89** |
| `tdeed_r0` | Round 0 | 0.85 | 0.6553 | 0.7566 | 0.5928 | **0.9340** | 0.4342 | 0.7255 | 0.7392 | 99 | **7** | 129 |
| `tdeed_r1` | Round 1 | 0.70 | **0.7307** | **0.7638** | 0.6523 | 0.8462 | 0.5307 | 0.7336 | 0.7453 | 121 | 22 | 107 |
| **`tdeed_r2`** | **Round 2** | **0.70** | 0.7188 | 0.7502 | **0.7044** | **0.8509** | **0.6009** | **0.7590** | **0.7678** | **137** | 24 | **91** |

### Gains & Analysis (`tdeed_r2`):
- **All-Time Benchmark High AP@±5s:** Test AP@±5s reached **0.7590** and Test loose a-mAP reached **0.7678** (**+4.71 AP points** over `v2_r2`'s 0.7119). Uncalibrated raw Test AP5 reached **0.7669** and raw loose a-mAP reached **0.7750**.
- **Test $F_1$ Breaks 0.70:** Calibrated Test $F_1 = \mathbf{0.7044}$ with **137 True Positives** (vs 121 in R1 and 99 in R0) and only 24 false alarms across 50 full 90-minute test games (**85.1% Precision**, 0.48 FP/game).
- **Raw Operating Point:** On raw uncalibrated scores at threshold 0.90, `tdeed_r2` achieves **Test $F_1 = 0.7460$** ($P = 0.7799, R = 0.7149$, 163 TP, 46 FP, 65 FN).
- **Training Efficiency:** Capping at 8 epochs with patience 2 saved ~1.5 hours of redundant late-epoch training while cleanly capturing the peak model weights at Epoch 6.

---

## 2026-10-14 — Raising precision and recall: ensemble, dual calibration, round 3, flip TTA (CODE WRITTEN, not run)

### Diagnostics (read-only, on existing benchmark predictions)
Computed with a scratch script over `runs/{v2_r2,tdeed_r1,tdeed_r2}/bench/`. No training, no benchmark change.

| Test (228 cards) | v2_r2 | tdeed_r2 |
|---|---|---|
| Recall at calibrated 0.70 | 0.610 | 0.601 |
| Recall at any score (a detection within ±5 s) | 0.974 | 0.943 |
| Median raw peak score near a true card | 0.93 | 0.98 |
| Cards with raw peak < 0.5 | 23 | 32 |

- **Recall is lost at the threshold, not at detection.** Both models fire near 94–97% of cards.
- **The 0.70 floor binds.** The isotonic calibrator outputs P(this detection is real). On that scale the
  F1-optimal threshold is roughly F1*/2 ≈ 0.37. v2_r2, tdeed_r1 and tdeed_r2 all selected 0.70, the grid floor.
  This is why every run lands near P 0.85 / R 0.60.
- **The models are complementary.** At 0.70 on test: 115 cards found by both, 24 only by v2_r2, 22 only by
  tdeed_r2. Union 161.
- **Quick fusion probe** (calibrated scores averaged, weight and threshold picked on val, any threshold): test
  F1 0.773, P 0.784, R 0.763 (174 TP, 48 FP). Restricted to the grid with marginal scores: test F1 0.642. The
  fusion gain only reaches the headline if the meaning of 0.70 changes too.
- **Selection noise.** tdeed_r1 `best.pt`: AP 0.793 on the 20-game `val_v1` subset vs 0.764 on full val.
  tdeed_r2 is below r1 on full val (0.750 vs 0.764) but above it on test (0.759 vs 0.734).
- **Over-warm fine-tunes.** Each mining round re-warms to lr 8e-4. Val loss rises from epoch 1 (r2: 0.424 → 0.496).
- **Hard floor.** 23–32 test cards score below 0.5 in every model. Thresholds cannot recover them.

### Decisions
- **Both calibration meanings are kept** (user decision). Each fused or recalibrated run is scored twice by the
  unchanged `benchmark.py`:
  - `<name>_marg`: isotonic P(this detection is real). Same meaning as every earlier row.
  - `<name>_cum`: isotonic val precision of all detections scoring ≥ this one. Grid 0.70 then means "≥ 70% of
    what you keep is real".
- **Fusion inputs use Platt scaling per run**, not the existing isotonic outputs. Isotonic steps create ties and
  lose ranking inside a run. Platt is monotone, so ranking is kept.
- **Fusion weights are chosen by val AP@±5 s** (threshold-free), on a simplex grid with step 0.1.
- **Round 3 fine-tune:** lr ×0.25 (head 2e-4, backbone 5e-5), 1-epoch warmup, 5 epochs, patience 2. Spotting on
  the full 50-game val (`train.spot_subset=val`) every 2 epochs. Round 3 scans train games 200–299, never
  mined before.
- **Checkpoint averaging** (`avg.pt`) is a selection-free alternative to `best.pt`. Pick between them on val only.
- **Flip TTA** is inference-only and off by default.

### Files
- **New:** `ensemble/{__init__.py, fuse.py, README.md}`. Model-agnostic. Imports only
  `benchmark.benchmark.{load_gt, load_preds, match, average_precision, SPLIT_PATH}`, so labels follow the
  benchmark rules exactly. Never reads test labels.
- **New:** `tdeed/average.py`. Averages the EMA weights of the listed epochs into `avg.pt`.
- **Changed:** `tdeed/infer.py`.
  - `infer.tta_flip` scores width-mirrored windows too and averages the voted probabilities.
  - The default output dir and printed run name now carry the checkpoint stem and TTA
    (`bench/`, `bench_avg/`, `bench_tta/`, `bench_avg_tta/`), so results never overwrite each other.
- **Changed:** `tdeed/config.yaml`. New key `infer.tta_flip: false`. Older checkpoints default to off.
- **Changed:** `tdeed/README.md`. Status, file table, round-3 recipe, CPU decode note.
- **Unchanged:** `benchmark/benchmark.py`, `benchmark/split.json`.

### Execution — Phase 1 complete (2026-10-14)

Phase 1 evaluated fusion and dual calibration on existing `v2_r2` and `tdeed_r2` predictions. Zero training needed.

1. `tdeed_r2_cal_marg`: exact reproduction of baseline `tdeed_r2` (F1 0.7044, AP5 0.7590, 137 TP, 24 FP, 91 FN). Sanity check passed.
2. `tdeed_r2_cal_cum`: cumulative precision thresholding alone unlocks **F1 0.7290 | P 0.8042 | R 0.6667** (152 TP, 37 FP, 76 FN) at thr 0.85. +2.5 F1 pts and +6.6% recall with single model.
3. `ens_r2_marg`: late fusion with marginal calibration hits **F1 0.7544 | P 0.8922 | R 0.6535** (149 TP, 18 FP, 79 FN) at thr 0.75. Precision touches ~90%.
4. `ens_r2_cum`: late fusion with cumulative calibration reaches **F1 0.7638 | P 0.7689 | R 0.7588** (173 TP, 52 FP, 55 FN) at thr 0.80. Val AP5 jumped to **0.7898**, test AP5 to **0.7902**.

### Execution — Phases 1, 2, 3 & 4 fully completed (2026-10-14)

1. **Phase 1 (Zero-training eval):** Evaluated dual calibration (`_marg` vs `_cum`) and late fusion on `v2_r2` + `tdeed_r2`. `ens_r2_cum` reached F1 0.7638 | P 0.7689 | R 0.7588 (173 TP, 52 FP, 55 FN).
2. **Phase 2 (Mining):** Scanned 100 train games (games 200–299, 200 halves) $\to$ 892 hard negatives mined to `data/tdeed/mined_round3.json`.
3. **Phase 3 (Extraction & Training):** 892 windows extracted (`[mined] done, failed 0`). Fine-tuned `tdeed_r3` with lr ×0.25 on pooled rounds 1+2+3. Best checkpoint hit at **Epoch 2** (spot AP5 0.7401, F1 0.7500). Early stopped at epoch 5.
4. **Phase 4 (Inference, Checkpoint Averaging, Flip TTA & Grand Fusion):**
   - **Checkpoint Selection:** `best.pt` (Epoch 2, Val AP5 0.7555, Val F1 0.7478) beat `avg.pt` (Val AP5 0.7438, Val F1 0.7378). `best.pt` selected by val.
   - **Single Model Records (`tdeed_r3`):**
     - `tdeed_r3_cal_cum`: **F1 0.7371 | P 0.7929 | R 0.6886** (157 TP, 41 FP, 71 FN). New best single model.
   - **Grand Fusion Records (`v2_r2` + `tdeed_r3`):**
     - `ens_r3_best_marg`: F1 0.7200 | **P 0.9184** | R 0.5921 (135 TP, 12 FP, 93 FN). High-precision specialist (only 12 false alarms across 50 full test games).
     - `ens_r3_best_cum`: **F1 0.7729 | P 0.8602 | R 0.7018** (160 TP, 26 FP, 68 FN) at thr 0.90 (val F1 0.7801). **All-time repo record F1.**
     - `ens_r3_tta_cum`: **F1 0.7585 | P 0.7336 | R 0.7851** (179 TP, 65 FP, 49 FN). **All-time repo record Recall: 78.5%** (179 out of 228 cards caught; only 49 missed in 50 matches).



