from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class H5Spec:
    image_size: int = 224

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.image_size, self.image_size, 3)


def create_h5_datasets(h5f: h5py.File, spec: H5Spec) -> None:
    """Create extendable datasets."""
    str_dt = h5py.string_dtype(encoding="utf-8")
    h5f.create_dataset(
        "images",
        shape=(0, *spec.image_shape),
        maxshape=(None, *spec.image_shape),
        dtype=np.uint8,
        chunks=True,
    )
    h5f.create_dataset("races", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
    h5f.create_dataset("genders", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
    h5f.create_dataset("files", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)


def append_batch(h5f: h5py.File, images: np.ndarray, races: np.ndarray, genders: np.ndarray, files: list[str]) -> None:
    """Append a batch to datasets."""
    if len(images) == 0:
        return
    n0 = int(h5f["images"].shape[0])
    n1 = n0 + int(images.shape[0])

    h5f["images"].resize(n1, axis=0)
    h5f["races"].resize(n1, axis=0)
    h5f["genders"].resize(n1, axis=0)
    h5f["files"].resize(n1, axis=0)

    h5f["images"][n0:n1] = images
    h5f["races"][n0:n1] = races
    h5f["genders"][n0:n1] = genders
    h5f["files"][n0:n1] = np.asarray(files, dtype=h5f["files"].dtype)


def validate_h5(path: str | Path, spec: H5Spec) -> None:
    """Validate required datasets and basic consistency."""
    with h5py.File(path, "r") as h5f:
        for k in ["images", "races", "genders", "files"]:
            if k not in h5f:
                raise ValueError(f"HDF5 missing dataset '{k}': {path}")
        n = int(h5f["images"].shape[0])
        if h5f["images"].shape[1:] != spec.image_shape:
            raise ValueError(f"Unexpected image shape {h5f['images'].shape[1:]} (expected {spec.image_shape})")
        for k in ["races", "genders", "files"]:
            if int(h5f[k].shape[0]) != n:
                raise ValueError(f"Length mismatch: images={n} vs {k}={int(h5f[k].shape[0])}")


