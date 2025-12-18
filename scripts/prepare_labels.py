from __future__ import annotations

import argparse
from pathlib import Path

from ff_pipeline.labels import prepare_label_frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode labels and create val/test split.")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = prepare_label_frames(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_size=args.test_size,
        seed=args.seed,
    )

    (out_dir / "fairface_label_train_encoded.csv").write_text(
        frames.train.to_csv(index=False), encoding="utf-8"
    )
    (out_dir / "fairface_label_val_encoded.csv").write_text(
        frames.val.to_csv(index=False), encoding="utf-8"
    )
    (out_dir / "fairface_label_test_encoded.csv").write_text(
        frames.test.to_csv(index=False), encoding="utf-8"
    )

    print("Wrote:")
    print("-", out_dir / "fairface_label_train_encoded.csv")
    print("-", out_dir / "fairface_label_val_encoded.csv")
    print("-", out_dir / "fairface_label_test_encoded.csv")


if __name__ == "__main__":
    main()


