from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from ff_pipeline.hdf5_utils import H5Spec, append_batch, create_h5_datasets


def detect_and_crop_face_haar(image_bgr: np.ndarray, image_size: int) -> np.ndarray | None:
    """Detect and crop a face; return resized BGR image or None."""
    if image_bgr is None:
        return None
    if (image_bgr.shape[0], image_bgr.shape[1]) == (image_size, image_size):
        return image_bgr

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    cropped = image_bgr[y : y + h, x : x + w]
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_labels_to_hdf5(
    df: pd.DataFrame,
    data_dir: str | Path,
    out_h5: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 512,
    log_path: str | Path | None = None,
    overwrite: bool = True,
) -> None:
    """Convert a labeled split into an HDF5 file."""
    data_dir = Path(data_dir)
    out_h5 = Path(out_h5)
    if out_h5.exists() and not overwrite:
        raise FileExistsError(out_h5)

    required = ["file", "race_encoded", "gender_encoded"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    spec = H5Spec(image_size=image_size)
    log_fh = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_h5, "w") as h5f:
            create_h5_datasets(h5f, spec)

            batch_images: list[np.ndarray] = []
            batch_races: list[int] = []
            batch_genders: list[int] = []
            batch_files: list[str] = []

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing → {out_h5.name}"):
                rel = str(row["file"])
                img_path = data_dir / "fairface" / rel
                image_bgr = cv2.imread(str(img_path))
                face_bgr = detect_and_crop_face_haar(image_bgr, image_size=image_size)
                if face_bgr is None:
                    if log_fh:
                        log_fh.write(f"{rel}\n")
                    continue

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                batch_images.append(face_rgb.astype(np.uint8))
                batch_races.append(int(row["race_encoded"]))
                batch_genders.append(int(row["gender_encoded"]))
                batch_files.append(rel)

                if len(batch_images) >= batch_size:
                    append_batch(
                        h5f,
                        images=np.asarray(batch_images, dtype=np.uint8),
                        races=np.asarray(batch_races, dtype=np.int32),
                        genders=np.asarray(batch_genders, dtype=np.int32),
                        files=batch_files,
                    )
                    batch_images.clear()
                    batch_races.clear()
                    batch_genders.clear()
                    batch_files.clear()

            if batch_images:
                append_batch(
                    h5f,
                    images=np.asarray(batch_images, dtype=np.uint8),
                    races=np.asarray(batch_races, dtype=np.int32),
                    genders=np.asarray(batch_genders, dtype=np.int32),
                    files=batch_files,
                )
    finally:
        if log_fh:
            log_fh.close()


