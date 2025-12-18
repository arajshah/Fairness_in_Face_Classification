from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


RACE_MAPPING = {
    "White": 0,
    "Black": 1,
    "East Asian": 2,
    "Southeast Asian": 3,
    "Indian": 4,
    "Latino_Hispanic": 5,
    "Middle Eastern": 6,
}

GENDER_MAPPING = {"Male": 0, "Female": 1}


@dataclass(frozen=True)
class LabelFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """Load a FairFace label table."""
    df = pd.read_csv(csv_path)
    _require_columns(df, ["file", "age", "gender", "race"], name=str(csv_path))
    return df


def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric encodings for race and gender."""
    out = df.copy()
    out["race_encoded"] = out["race"].map(RACE_MAPPING)
    out["gender_encoded"] = out["gender"].map(GENDER_MAPPING)
    if out["race_encoded"].isna().any():
        bad = sorted(out.loc[out["race_encoded"].isna(), "race"].unique().tolist())
        raise ValueError(f"Unmapped race values: {bad}")
    if out["gender_encoded"].isna().any():
        bad = sorted(out.loc[out["gender_encoded"].isna(), "gender"].unique().tolist())
        raise ValueError(f"Unmapped gender values: {bad}")
    out["race_encoded"] = out["race_encoded"].astype(int)
    out["gender_encoded"] = out["gender_encoded"].astype(int)
    return out


def split_val_into_val_test(
    df_val_encoded: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an encoded validation table into new validation and test sets."""
    _require_columns(df_val_encoded, ["race_encoded", "gender_encoded"], name="df_val_encoded")

    # sklearn stratify expects 1D labels; stratify on intersection to preserve both.
    strat = df_val_encoded["race_encoded"].astype(str) + "_" + df_val_encoded["gender_encoded"].astype(str)
    df_val_new, df_test = train_test_split(
        df_val_encoded,
        test_size=test_size,
        stratify=strat,
        random_state=seed,
    )
    return df_val_new.reset_index(drop=True), df_test.reset_index(drop=True)


def prepare_label_frames(
    train_csv: str | Path,
    val_csv: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> LabelFrames:
    """Load raw train/val labels, encode, and split val into val/test."""
    df_train = add_encodings(load_labels(train_csv))
    df_val = add_encodings(load_labels(val_csv))
    df_val_new, df_test = split_val_into_val_test(df_val, test_size=test_size, seed=seed)
    return LabelFrames(train=df_train, val=df_val_new, test=df_test)


