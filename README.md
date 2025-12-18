# FairFace Multi-Task Classification (Race + Gender)

This repository contains a reproducible pipeline for:

- preparing FairFace image data + label tables
- optionally detecting/cropping faces and storing the processed tensors in **HDF5**
- training a **multi-output** Keras model (shared backbone + separate heads)
- running **subgroup evaluation** and exporting metrics artifacts

## Repository layout

- `notebooks/`: interactive exploration and training notebooks (kept lightweight; core logic lives in `src/`)
- `src/ff_pipeline/`: reusable modules (data IO, preprocessing, model, training, evaluation)
- `scripts/`: thin CLI wrappers
- `data/`: dataset assets (ignored by git)
- `models/`: model checkpoints (ignored by git)

## Data expectations

Place FairFace under:

```text
data/
  fairface/
    train/*.jpg
    val/*.jpg
  fairface_label_train.csv
  fairface_label_val.csv
```

Label CSV format (minimum):

- `file`: relative path like `train/1.jpg` or `val/1.jpg` (relative to `data/fairface/`)
- `age`: age bucket string
- `gender`: `Male` / `Female`
- `race`: one of 7 categories
- `service_test`: `True` / `False` (kept as metadata)

## Quickstart (recommended)

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 1) Encode labels + create test split

```bash
python scripts/prepare_labels.py \
  --train-csv data/fairface_label_train.csv \
  --val-csv data/fairface_label_val.csv \
  --out-dir data \
  --test-size 0.2 \
  --seed 42
```

### 2) Preprocess images → HDF5 (optional but recommended)

```bash
python scripts/preprocess_to_hdf5.py \
  --data-dir data \
  --train-csv data/fairface_label_train_encoded.csv \
  --val-csv data/fairface_label_val_encoded.csv \
  --test-csv data/fairface_label_test_encoded.csv \
  --out-dir data \
  --image-size 224 \
  --batch-size 512
```

This produces:
- `data/train.h5`, `data/val.h5`, `data/test.h5`
- `data/no_face_detected.log`

### 3) Train

```bash
python scripts/train.py \
  --train-h5 data/train.h5 \
  --val-h5 data/val.h5 \
  --out-dir models \
  --batch-size 16 \
  --epochs 10 \
  --max-train-samples 5000 \
  --max-val-samples 5000
```

### 4) Evaluate (subgroup metrics)

```bash
python scripts/evaluate.py \
  --h5 data/test.h5 \
  --labels-csv data/fairface_label_test_encoded.csv \
  --model models/best_model.keras \
  --out-dir reports
```

Outputs include:

- `reports/overall_metrics.csv`
- `reports/subgroup_metrics.csv` (race / gender / intersection)
- `reports/disparity_summary.csv` (worst-group + max–min gaps)
- `reports/confusion_matrices_long.csv` (overall + per-group)
- `reports/calibration_tables.csv` (confidence-binned reliability tables)
- `reports/predictions.csv` (per-example predictions + confidence)

## Notes

- The `data/` and `models/` directories are intentionally ignored by git (large artifacts).
- If you are on macOS, TensorFlow installation varies by hardware. Use a compatible TensorFlow build for your system.


