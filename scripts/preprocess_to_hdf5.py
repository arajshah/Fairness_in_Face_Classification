from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ff_pipeline.hdf5_utils import H5Spec, validate_h5
from ff_pipeline.preprocess import preprocess_labels_to_hdf5


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess labeled splits into HDF5.")
    ap.add_argument("--data-dir", required=True, help="Directory containing fairface/ and CSVs.")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = H5Spec(image_size=args.image_size)
    log_path = out_dir / "no_face_detected.log"

    def run_one(csv_path: str, out_name: str) -> None:
        df = pd.read_csv(csv_path)
        preprocess_labels_to_hdf5(
            df=df,
            data_dir=data_dir,
            out_h5=out_dir / out_name,
            image_size=args.image_size,
            batch_size=args.batch_size,
            log_path=log_path,
            overwrite=True,
        )
        validate_h5(out_dir / out_name, spec)

    run_one(args.train_csv, "train.h5")
    run_one(args.val_csv, "val.h5")
    run_one(args.test_csv, "test.h5")

    print("Wrote:")
    print("-", out_dir / "train.h5")
    print("-", out_dir / "val.h5")
    print("-", out_dir / "test.h5")
    print("-", log_path)


if __name__ == "__main__":
    main()


