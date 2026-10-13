# T-DEED replica for card spotting

Self-contained re-implementation of **T-DEED** (Xarles et al., *Temporal-Discriminability Enhancer
Encoder-Decoder for Precise Event Spotting in Sports Videos*, CVPRW 2024; reference code
`github.com/arturxe2/T-DEED`). It is adapted for binary card spotting (Yellow, Red and Yellow->red merged
into one class vs background). It's scored with the frozen `benchmark/` (50 val / 50 test), the same as v1/v2.

Status: **code written, not run.**

## Files
| File | Role |
|---|---|
| `config.yaml` | All hyperparameters. Override any key with `--set key=value`. |
| `common.py` | Config, split/index access, cache paths, JPEG access, batched GPU decode. |
| `index.py` | Builds `data/tdeed/index.json`: halves, 40 s train windows (pos/neg), val clip specs. |
| `extract.py` | Caches train windows as JPEG `.npz` (5 fps, 224x398). Fills missing eval halves only. |
| `gsm.py` | Gate-Shift Module inserted into RegNet stages s3/s4. |
| `model.py` | RegNetY-200MF+GSM, then 368->256 projection, then SGP encoder / SGP-Mixer decoder, then class + displacement heads. |
| `augment.py` | T-DEED clip augmentation (hue/sat/brightness/contrast/blur at p=0.25 each, h-flip) + mixup. |
| `data.py` | Train windows (epoch resampling) and val clips. Items carry raw JPEG bytes. |
| `train.py` | Training loop: AMP, EMA, warmup+cosine, val clip loss, periodic spotting, best/last checkpoints. |
| `infer.py` | Sliding windows, displacement voting, NMS, isotonic calibration, SoccerNet-format output. |
| `mine.py` | Hard-negative mining on train games (for later rounds). |
| `spotting.py` | NMS, metrics, calibration, writers (copied from the v2 logic so `src/` is not imported). |

Shared, read-only: `benchmark/split.json`, `soccernet_data/`, and the 5 fps val/test half cache in `frames/eval/`.
New data goes to `data/tdeed/` and `frames_tdeed/`. Runs go to `runs/tdeed_*`.

## Run order
```bash
venv/bin/pip install -r tdeed/requirements.txt          # timm (RegNetY-200MF); pretrained weights download on first use
venv/bin/python -m tdeed.index                           # index + window specs
venv/bin/python -m tdeed.extract windows --workers 8     # ~7.5k windows x 200 frames, ~33 GB
venv/bin/python -m tdeed.extract eval-halves             # no-op if frames/eval is complete
venv/bin/python -m tdeed.train --name tdeed_r0
venv/bin/python -m tdeed.infer --ckpt runs/tdeed_r0/best.pt
venv/bin/python benchmark/benchmark.py --run tdeed_r0 \
    --val-pred runs/tdeed_r0/bench/calibrated/val --test-pred runs/tdeed_r0/bench/calibrated/test
# later rounds
venv/bin/python -m tdeed.mine --ckpt runs/tdeed_r0/best.pt --round 1
venv/bin/python -m tdeed.extract windows --kind mined --specs data/tdeed/mined_round1.json
venv/bin/python -m tdeed.train --name tdeed_r1 --init runs/tdeed_r0/best.pt --mined data/tdeed/mined_round1.json
```
For long jobs, use `systemd-run --user --unit=<name> ...` (same as v2).

## Design decisions (agreed)
- **Frames:** 5 fps, 224x398 full frame. Clip = 100 frames = 20 s.
  - Train data is 40 s windows: one centred on each visible card, plus 6,000 negative windows. Each epoch draws a random 100-frame crop per window.
  - For positive windows, the card can land anywhere in the crop.
- **Negatives:** 50% random, 30% post-foul, 20% stoppage. The whole window must be at least 60 s from any card.
  - Per epoch: all positives plus 1.5x negatives.
  - No v2 mined negatives. T-DEED does its own mining in later rounds (`mine.py`, capped at 30% of negatives).
- **Backbone:** timm `regnety_002`, ImageNet init. GSM on 1/4 of conv1 input channels in stages s3/s4 (as in T-DEED).
  - Per-stage gradient checkpointing.
- **Temporal:** 2-level encoder-decoder, SGP kernel 9, r=4 (T-DEED SoccerNetBall config). Hidden d=256 via a linear projection from 368.
- **Targets (T-DEED):**
  - Frames within ±4 (0.8 s) of a visible card are labelled card, with displacement target `i - card` in frames.
  - Loss: soft-target cross-entropy with fg weight 5, plus MSE on displacement.
  - Frames within ±5 s of a "not shown" card get zero loss.
- **Training:** AdamW, lr 8e-4 head / 2e-4 backbone, wd 0.01. 3-epoch linear warmup (from 1%), then cosine. 40 epochs, batch 4, AMP, EMA 0.999.
  - Val clip loss every epoch.
  - Spotting on the 20 `val_v1` games every 3 epochs. `best.pt` is the best AP@±5 s. Patience: 4 spotting evals.
- **Inference:**
  - 100-frame windows at 50% overlap, softmax card prob.
  - T-DEED displacement voting (frame t gives its score to t - disp, max aggregation).
  - Average over overlapping windows, then 10 s NMS.
  - Isotonic calibration fitted on val only. Writes both raw and calibrated predictions.

## Deviations from the reference implementation
- Reference SoccerNetBall used RegNetY-800MF + GSF at 25 fps with stride 2 (12.5 fps). Here: 200MF + GSM at 5 fps (agreed). The temporal kernels therefore cover 5x more seconds.
- Reference feature width = backbone width (368 for 200MF). Here: projection to 256 (`model.d_model: null` restores the original).
- The reference used CUDA-only GSM code. Here it's re-implemented device-agnostic, with identical math.
- Mixup partner is drawn from the same batch instead of a second dataset sample.
- The SGP-Mixer upsamples to the skip length dynamically, instead of a fixed `t_size`. Same result for fixed clips.
- Frames are decoded batched on the GPU (nvjpeg). DataLoader workers default to 0, because worker IPC crashed the v2 runs.
- The reference trains on random clips from full videos. Here it uses card-centred + negative windows, since cards are sparse (~1 per 30 min).
- Post-processing adds isotonic calibration, so the benchmark's fixed 0.70-0.95 thresholds are meaningful.

## Rough cost (estimates, unmeasured)
- Disk: ~33 GB of train windows.
- Epoch: ~3.9k clips x 100 frames. Expect ~15-25 min on an RTX 4060.
- Spotting eval on 40 halves: ~10-15 min.
- If VRAM overflows, use `--set train.batch_size=2 train.acc_grad=2`.
