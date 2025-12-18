from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from ff_pipeline.evaluation import predict_h5, subgroup_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run subgroup evaluation and export metrics.")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--labels-csv", required=False, default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--calibration-bins", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    pred_df = predict_h5(model, args.h5, batch_size=args.batch_size)
    outputs = subgroup_report(pred_df, labels_csv=args.labels_csv, n_bins=args.calibration_bins)

    pred_path = out_dir / "predictions.csv"
    overall_path = out_dir / "overall_metrics.csv"
    groups_path = out_dir / "subgroup_metrics.csv"
    disparity_path = out_dir / "disparity_summary.csv"
    confusion_path = out_dir / "confusion_matrices_long.csv"
    calibration_path = out_dir / "calibration_tables.csv"

    pred_df.to_csv(pred_path, index=False)
    outputs.overall.to_csv(overall_path, index=False)
    outputs.per_group.to_csv(groups_path, index=False)
    outputs.disparity_summary.to_csv(disparity_path, index=False)
    outputs.confusion_long.to_csv(confusion_path, index=False)
    outputs.calibration.to_csv(calibration_path, index=False)

    print("Wrote:")
    print("-", pred_path)
    print("-", overall_path)
    print("-", groups_path)
    print("-", disparity_path)
    print("-", confusion_path)
    print("-", calibration_path)


if __name__ == "__main__":
    main()


