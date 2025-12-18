from __future__ import annotations

import argparse

from ff_pipeline.training import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a multitask model from HDF5.")
    ap.add_argument("--train-h5", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-imagenet", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
        imagenet_weights=not args.no_imagenet,
    )

    train(train_h5=args.train_h5, val_h5=args.val_h5, out_dir=args.out_dir, cfg=cfg)


if __name__ == "__main__":
    main()


