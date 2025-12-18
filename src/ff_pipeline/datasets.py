from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import tensorflow as tf

from ff_pipeline.hdf5_utils import H5Spec, validate_h5


def make_hdf5_dataset(
    h5_path: str | Path,
    *,
    spec: H5Spec,
    batch_size: int = 16,
    shuffle: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset streaming from HDF5."""
    validate_h5(h5_path, spec)

    def gen() -> Iterator[tuple[tf.Tensor, dict[str, tf.Tensor]]]:
        with h5py.File(h5_path, "r") as h5f:
            images = h5f["images"]
            races = h5f["races"]
            genders = h5f["genders"]
            n = int(images.shape[0])
            if max_samples is not None:
                n = min(n, int(max_samples))
            for i in range(n):
                img = images[i]
                y = {
                    "race_output": races[i],
                    "gender_output": genders[i],
                }
                yield img, y

    output_signature = (
        tf.TensorSpec(shape=(spec.image_size, spec.image_size, 3), dtype=tf.uint8),
        {
            "race_output": tf.TensorSpec(shape=(), dtype=tf.int32),
            "gender_output": tf.TensorSpec(shape=(), dtype=tf.int32),
        },
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


