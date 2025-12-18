from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass(frozen=True)
class EvalOutputs:
    per_group: pd.DataFrame
    overall: pd.DataFrame
    disparity_summary: pd.DataFrame
    confusion_long: pd.DataFrame
    calibration: pd.DataFrame


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())

def _bin_confidence(conf: pd.Series, *, n_bins: int) -> pd.Categorical:
    # Right-closed bins: (0.0, 0.1], ... (0.9, 1.0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0] = -1e-9
    edges[-1] = 1.0 + 1e-9
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(n_bins)]
    return pd.cut(conf, bins=edges, labels=labels, include_lowest=True)


def _confusion_long(
    df: pd.DataFrame,
    *,
    task: str,
    true_col: str,
    pred_col: str,
    group_col: str | None,
) -> pd.DataFrame:
    base = df.copy()
    base["grouping"] = "overall" if group_col is None else group_col
    base["group"] = "all" if group_col is None else base[group_col].astype(str)
    out = (
        base.groupby(["grouping", "group", true_col, pred_col], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={true_col: "true_label", pred_col: "pred_label"})
    )
    out.insert(0, "task", task)
    return out


def _calibration_table(
    df: pd.DataFrame,
    *,
    task: str,
    correct_col: str,
    conf_col: str,
    group_col: str | None,
    n_bins: int,
) -> pd.DataFrame:
    base = df.copy()
    base["grouping"] = "overall" if group_col is None else group_col
    base["group"] = "all" if group_col is None else base[group_col].astype(str)
    base["conf_bin"] = _bin_confidence(base[conf_col], n_bins=n_bins)
    out = (
        base.groupby(["grouping", "group", "conf_bin"], dropna=False)
        .agg(
            n=(conf_col, "size"),
            accuracy=(correct_col, "mean"),
            avg_confidence=(conf_col, "mean"),
        )
        .reset_index()
    )
    out.insert(0, "task", task)
    return out


def _disparity_summary(per_group: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [c for c in ["race_accuracy", "gender_accuracy", "both_correct_rate"] if c in per_group.columns]
    for grouping in sorted(per_group["grouping"].unique().tolist()):
        sub = per_group[per_group["grouping"] == grouping].copy()
        if len(sub) == 0:
            continue
        for metric in metrics:
            vals = sub[metric].astype(float)
            mx = float(vals.max())
            mn = float(vals.min())
            gap = mx - mn
            worst_idx = int(vals.idxmin())
            best_idx = int(vals.idxmax())
            rows.append(
                {
                    "grouping": grouping,
                    "metric": metric,
                    "max": mx,
                    "min": mn,
                    "max_minus_min": gap,
                    "worst_group": str(sub.loc[worst_idx, "group"]),
                    "worst_value": float(sub.loc[worst_idx, metric]),
                    "best_group": str(sub.loc[best_idx, "group"]),
                    "best_value": float(sub.loc[best_idx, metric]),
                    "worst_over_best": float(sub.loc[worst_idx, metric]) / float(sub.loc[best_idx, metric])
                    if float(sub.loc[best_idx, metric]) != 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def predict_h5(
    model: tf.keras.Model,
    h5_path: str | Path,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run predictions and return a row-aligned table."""
    rows: list[dict] = []
    with h5py.File(h5_path, "r") as h5f:
        images = h5f["images"]
        races = h5f["races"][:]
        genders = h5f["genders"][:]
        files = h5f["files"][:]

        n = int(images.shape[0])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = images[start:end].astype(np.float32) / 255.0
            pr, pg = model.predict(x, verbose=0)
            y_r = np.argmax(pr, axis=1).astype(int)
            y_g = np.argmax(pg, axis=1).astype(int)
            conf_r = np.max(pr, axis=1).astype(float)
            conf_g = np.max(pg, axis=1).astype(float)
            for i in range(end - start):
                rows.append(
                    {
                        "file": files[start + i].decode("utf-8") if isinstance(files[start + i], (bytes, bytearray)) else str(files[start + i]),
                        "race_true": int(races[start + i]),
                        "gender_true": int(genders[start + i]),
                        "race_pred": int(y_r[i]),
                        "gender_pred": int(y_g[i]),
                        "race_conf": float(conf_r[i]),
                        "gender_conf": float(conf_g[i]),
                    }
                )
    return pd.DataFrame(rows)


def subgroup_report(
    pred_df: pd.DataFrame,
    labels_csv: str | Path | None = None,
    *,
    n_bins: int = 10,
) -> EvalOutputs:
    """Compute overall and subgroup summary tables."""
    df = pred_df.copy()
    df["race_correct"] = df["race_true"] == df["race_pred"]
    df["gender_correct"] = df["gender_true"] == df["gender_pred"]

    if labels_csv is not None:
        meta = pd.read_csv(labels_csv)
        # Join only metadata columns that exist.
        keep = [c for c in ["file", "age", "race", "gender", "service_test"] if c in meta.columns]
        meta = meta[keep]
        df = df.merge(meta, on="file", how="left")

    overall = pd.DataFrame(
        [
            {
                "n": len(df),
                "race_accuracy": _accuracy(df["race_true"].to_numpy(), df["race_pred"].to_numpy()),
                "gender_accuracy": _accuracy(df["gender_true"].to_numpy(), df["gender_pred"].to_numpy()),
                "both_correct_rate": float((df["race_correct"] & df["gender_correct"]).mean()) if len(df) else float("nan"),
                "race_avg_conf": float(df["race_conf"].mean()) if "race_conf" in df.columns and len(df) else float("nan"),
                "gender_avg_conf": float(df["gender_conf"].mean()) if "gender_conf" in df.columns and len(df) else float("nan"),
            }
        ]
    )

    # Base groups: by race, by gender, and by intersection (race x gender).
    per_group_rows: list[dict] = []

    for g in ["race_true", "gender_true", "race_true+gender_true"]:
        if g == "race_true" and "race_true" in df.columns:
            gb = df.groupby(["race_true"], dropna=False)
            for k, sub in gb:
                per_group_rows.append(
                    {
                        "grouping": "race_true",
                        "group": str(k),
                        "n": len(sub),
                        "race_accuracy": float(sub["race_correct"].mean()) if len(sub) else float("nan"),
                        "gender_accuracy": float(sub["gender_correct"].mean()) if len(sub) else float("nan"),
                        "both_correct_rate": float((sub["race_correct"] & sub["gender_correct"]).mean()) if len(sub) else float("nan"),
                        "race_avg_conf": float(sub["race_conf"].mean()) if "race_conf" in sub.columns and len(sub) else float("nan"),
                        "gender_avg_conf": float(sub["gender_conf"].mean()) if "gender_conf" in sub.columns and len(sub) else float("nan"),
                    }
                )
        elif g == "gender_true" and "gender_true" in df.columns:
            gb = df.groupby(["gender_true"], dropna=False)
            for k, sub in gb:
                per_group_rows.append(
                    {
                        "grouping": "gender_true",
                        "group": str(k),
                        "n": len(sub),
                        "race_accuracy": float(sub["race_correct"].mean()) if len(sub) else float("nan"),
                        "gender_accuracy": float(sub["gender_correct"].mean()) if len(sub) else float("nan"),
                        "both_correct_rate": float((sub["race_correct"] & sub["gender_correct"]).mean()) if len(sub) else float("nan"),
                        "race_avg_conf": float(sub["race_conf"].mean()) if "race_conf" in sub.columns and len(sub) else float("nan"),
                        "gender_avg_conf": float(sub["gender_conf"].mean()) if "gender_conf" in sub.columns and len(sub) else float("nan"),
                    }
                )
        elif g == "race_true+gender_true" and all(c in df.columns for c in ["race_true", "gender_true"]):
            gb = df.groupby(["race_true", "gender_true"], dropna=False)
            for (r, ge), sub in gb:
                per_group_rows.append(
                    {
                        "grouping": "race_true+gender_true",
                        "group": f"{r}_{ge}",
                        "race_true": int(r),
                        "gender_true": int(ge),
                        "n": len(sub),
                        "race_accuracy": float(sub["race_correct"].mean()) if len(sub) else float("nan"),
                        "gender_accuracy": float(sub["gender_correct"].mean()) if len(sub) else float("nan"),
                        "both_correct_rate": float((sub["race_correct"] & sub["gender_correct"]).mean()) if len(sub) else float("nan"),
                        "race_avg_conf": float(sub["race_conf"].mean()) if "race_conf" in sub.columns and len(sub) else float("nan"),
                        "gender_avg_conf": float(sub["gender_conf"].mean()) if "gender_conf" in sub.columns and len(sub) else float("nan"),
                    }
                )

    per_group = pd.DataFrame(per_group_rows).sort_values(["grouping", "n"], ascending=[True, False])
    per_group = per_group.reset_index(drop=True)

    disparity_summary = _disparity_summary(per_group)

    # Build intersection group label for confusion/calibration.
    if "race_true" in df.columns and "gender_true" in df.columns:
        df["race_true+gender_true"] = df["race_true"].astype(str) + "_" + df["gender_true"].astype(str)

    confusion_long = pd.concat(
        [
            _confusion_long(df, task="race", true_col="race_true", pred_col="race_pred", group_col=None),
            _confusion_long(df, task="gender", true_col="gender_true", pred_col="gender_pred", group_col=None),
            _confusion_long(df, task="race", true_col="race_true", pred_col="race_pred", group_col="gender_true"),
            _confusion_long(df, task="gender", true_col="gender_true", pred_col="gender_pred", group_col="race_true"),
            _confusion_long(df, task="race", true_col="race_true", pred_col="race_pred", group_col="race_true+gender_true")
            if "race_true+gender_true" in df.columns
            else pd.DataFrame(),
        ],
        ignore_index=True,
    )

    calibration = pd.concat(
        [
            _calibration_table(df, task="race", correct_col="race_correct", conf_col="race_conf", group_col=None, n_bins=n_bins),
            _calibration_table(df, task="gender", correct_col="gender_correct", conf_col="gender_conf", group_col=None, n_bins=n_bins),
            _calibration_table(df, task="race", correct_col="race_correct", conf_col="race_conf", group_col="race_true", n_bins=n_bins),
            _calibration_table(df, task="gender", correct_col="gender_correct", conf_col="gender_conf", group_col="gender_true", n_bins=n_bins),
            _calibration_table(df, task="race", correct_col="race_correct", conf_col="race_conf", group_col="race_true+gender_true", n_bins=n_bins),
        ],
        ignore_index=True,
    )

    return EvalOutputs(
        per_group=per_group,
        overall=overall,
        disparity_summary=disparity_summary,
        confusion_long=confusion_long,
        calibration=calibration,
    )


