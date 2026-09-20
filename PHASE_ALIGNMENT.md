# Native PoseRAC phase alignment

The old combined dataset assigned opposite binary targets to the same jumping-jack
and push-up endpoints in RepCount and our recordings. A binary phase output is not
an exercise/form-validity probability. Completion timestamps remain review truth;
they must not be edited to match the classifier's label polarity.

`build_dataset.py` now assigns positive to open jumping jacks / bent push-ups, and
negative to closed jumping jacks / straight push-ups. It uses existing normalized
landmarks only, independently per recording and partition. Jumping jacks compare
mean wrist elevation relative to shoulders (minimum separation 0.25 torso lengths).
Push-ups compare mean elbow flexion (minimum separation 15 degrees).
At least 80% of cross-phase endpoint comparisons must agree with the chosen polarity.
Missing, nonfinite, or ambiguous phase groups are excluded and recorded in the
dataset manifest, not silently relabeled. Other exercises retain their labels.

The dataset contains own training/validation records only; test landmark arrays are
not read. Training excludes both own validation and the RepCount video holdout.
Sampling identifies own training examples by `part == "own_train"`, not the source
ID, which is a hash. Their configured 4x weight now actually applies.

## Reproduction

Use an environment with NumPy and PyTorch. Set `POSE_DATA` to the existing local
`data_rtmpose` directory; do not commit landmark datasets, clips, or weights.

```sh
python -m unittest discover -s tests -v
python scripts/build_dataset.py --repcount "$POSE_DATA/repcount" \
  --own "$POSE_DATA/own" --out data_rtmpose/phase-aligned
python scripts/train_rtmpose.py --dataset data_rtmpose/phase-aligned/dataset.npz \
  --out runs/phase-aligned-v2 --fixed-epochs 60 --alpha 0 --own-weight 4 --seed 42
```

The 60-epoch run selects weights using held-out RepCount phase-label loss, not
test footage or training-video repetition counts. Decoder parameters are the fixed
upstream defaults in `scripts/phase_aligned_decode.json`; the old tuned thresholds
are not reused. Pass this file explicitly to `export_candidate.py --decode-tuning`,
along with a newly generated validation report and a new candidate ID. Export rejects
an existing ID so old evaluations remain reproducible.

Validate counts by replaying cached train/validation landmarks through the exact
Data Review decoder, comparing old and new weights on identical inputs. Report
counts, misses, extra counts during exercise, and false counts outside exercise
separately. Counts on own training videos are diagnostic, not generalization evidence.
Do not launch or inspect test evaluations to choose weights or thresholds.

## First candidate: rtmpose-native-9cls-phase-aligned-v2

Trained 60 epochs on 5,300 phase examples (220 own); 560 RepCount holdout examples
selected epoch 46 (zero-based), phase-label loss 0.10173. Nine ambiguous recording
groups / 194 examples were excluded, including three own push-up recordings.
All five own jumping-jack recordings were retained. Seed 42, own weight 4, alpha 0.
Dataset SHA256: `ef4966513adacb24b60ffdee0b32fff0ea8e2469a3ecf9f80bca06f3342e410c`.

Read-only replay of the 26 completed videos in the saved RTMPose Balanced / YOLOX-M /
CSRT / detect-every-10-frames Status run, keeping poses and frozen annotations fixed:

| Exercise | Matched reps, old → new | Extra counts during exercise | Counts outside exercise |
| --- | --- | --- | --- |
| Jumping jacks | 44/50 → 46/50 | 45 → 5 | 9 → 0 |
| Push-ups | 7/49 → 26/49 | 23 → 19 | 7 → 0 |
| Jabs | 34/41 → 33/41 | 19 → 12 | 1 → 1 |

These are training-footage diagnostics, not independent test results. The new
model still misses moves and may count invalid form. It is a separate experimental
candidate, not a replacement for the previous checkpoint. No test evaluations ran.
