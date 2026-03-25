# Silent Speech Interface

This repo now contains a v1 that follows JJ's preprocessing pipeline for the
multichannel silent-speech dataset stored in `data/` (stored locally).

## Current workflow

```text
data/                  # untouched source .npy files
processed/
  blocks/              # clean 8-channel continuous blocks
  trials/              # segmented 8-channel trials
  features/            # reserved for optional model-ready exports
metadata/
  blocks.csv
  trials.csv
  features.csv
  quarantine_blocks.csv
splits/
  by_block.json
  by_session.json
src/
  preprocessing/
    preprocessing.py   # Jayla's similar filtering + detection
    pipeline.py        # end-to-end dataset conversion
  data_loader.py       # feature/split loading
  model.py             # XGBoost model factory
  train.py             # model training entrypoint
```

## Preprocessing truth

The operational preprocessing and segmentation logic mirrors Jayla's code:

- DC removal
- 50 Hz notch, Q=30
- 20-450 Hz 4th-order Butterworth bandpass
- 40 ms RMS window, 20 ms hop
- smoothed RMS thresholding with Jayla's exact decay and merge rules
- per-channel activity masks fused with `active_channels >= 2`

## Commands

Create the processed dataset:

```bash
./.venv/bin/python -m src.preprocessing.pipeline
```

Train the first-pass model:

```bash
./.venv/bin/python -m src.train \
  --features-csv metadata/features.csv \
  --split-path splits/by_session.json \
  --save-path results/models/xgboost.joblib
```

## Notes

- Dodgy blocks have been quarantined and are listed in `metadata/quarantine_blocks.csv`.
- `by_session.json` is the default leakage-safe split for evaluation.
- `xgboost` is required for training and must be installed in the environment.
