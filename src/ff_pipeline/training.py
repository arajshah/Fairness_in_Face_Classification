from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ff_pipeline.datasets import make_hdf5_dataset
from ff_pipeline.hdf5_utils import H5Spec
from ff_pipeline.modeling import build_multitask_model


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    seed: int = 42
    imagenet_weights: bool = True


def _count_h5_samples(path: str | Path) -> int:
    with h5py.File(path, "r") as h5f:
        return int(h5f["images"].shape[0])


def train(
    *,
    train_h5: str | Path,
    val_h5: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig,
) -> tf.keras.Model:
    """Train a multitask model and write the best checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = H5Spec(image_size=cfg.image_size)

    train_ds = make_hdf5_dataset(
        train_h5,
        spec=spec,
        batch_size=cfg.batch_size,
        shuffle=True,
        max_samples=cfg.max_train_samples,
        seed=cfg.seed,
    )
    val_ds = make_hdf5_dataset(
        val_h5,
        spec=spec,
        batch_size=cfg.batch_size,
        shuffle=False,
        max_samples=cfg.max_val_samples,
        seed=cfg.seed,
    )

    model, base = build_multitask_model(
        image_size=cfg.image_size,
        weights="imagenet" if cfg.imagenet_weights else None,
        base_trainable=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss={
            "race_output": "sparse_categorical_crossentropy",
            "gender_output": "sparse_categorical_crossentropy",
        },
        metrics={
            "race_output": ["accuracy"],
            "gender_output": ["accuracy"],
        },
    )

    # Steps: if max_samples is set, use it; otherwise infer from file.
    n_train = cfg.max_train_samples or _count_h5_samples(train_h5)
    n_val = cfg.max_val_samples or _count_h5_samples(val_h5)
    steps_per_epoch = ceil(n_train / cfg.batch_size)
    validation_steps = ceil(n_val / cfg.batch_size)

    ckpt_path = out_dir / "best_model.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_race_output_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_race_output_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    # Ensure best weights are persisted.
    model.save(str(ckpt_path))
    return model


