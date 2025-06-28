"""
playground/predict.py
=====================

Single-row inference utilities – imported by main.py or can be run
directly for quick ad-hoc predictions.

Key exports
-----------
preprocess_single(dict)  →  1×N float32 torch.Tensor
predict(dict, threshold) →  (probability, decision)

The code auto-loads:
    • scaler.pkl
    • feature_names.pkl
    • best_heartnet.pth   (unless a custom path is supplied)

So training must have produced those artefacts first.
"""

from pathlib import Path
import pickle
from typing import Tuple, Dict

import pandas as pd
import torch

# ────────────────────────────────────────────────────────────────
# Config – adjust paths if you keep artefacts elsewhere
# ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model/best_heartnet.pth")
SCALER_PATH = Path("model/scaler.pkl")
FEATURES_PATH = Path("model/feature_names.pkl")
RAW_CSV_PATH = Path("data/heart.csv")   # used only for auto-metadata


# ────────────────────────────────────────────────────────────────
# 1.  Load artefacts once
# ────────────────────────────────────────────────────────────────
with open(SCALER_PATH, "rb") as f:
    _scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    _feature_names = pickle.load(f)

# import here to avoid circular deps
from modules.trainer import HeartNet

_model = HeartNet(num_features=len(_feature_names))
_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
_model.eval()


# ────────────────────────────────────────────────────────────────
# 2.  Column-metadata helper (for nicer prompts)
# ────────────────────────────────────────────────────────────────
def _column_metadata(csv_path: Path = RAW_CSV_PATH) -> Dict[str, Dict]:
    df = pd.read_csv(csv_path)
    meta = {}
    for col in df.columns:
        if col == "HeartDisease":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            meta[col] = {
                "kind": "numeric",
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": round(df[col].mean(), 1),
            }
        else:
            uniq = df[col].dropna().unique().tolist()
            meta[col] = {
                "kind": "categorical",
                "choices": uniq[:6],
                "n_unique": len(uniq),
            }
    return meta


_META = _column_metadata()


# ────────────────────────────────────────────────────────────────
# 3.  Preprocess ONE sample ➜ tensor
# ────────────────────────────────────────────────────────────────
def preprocess_single(sample: Dict[str, object]) -> torch.Tensor:
    """
    Convert raw dict → 1×N float32 tensor in the *exact* order/scale
    the network was trained on.
    """
    df = pd.DataFrame([sample])

    # identify cats/nums
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # one-hot with same drop_first rule
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # add any missing dummies
    for col in _feature_names:
        if col not in df.columns:
            df[col] = 0

    # order columns
    df = df[_feature_names]

    # scale numerics
    df[num_cols] = _scaler.transform(df[num_cols])

    return torch.tensor(df.values.astype("float32"))


# ────────────────────────────────────────────────────────────────
# 4.  Predict helper
# ────────────────────────────────────────────────────────────────
def predict(sample: Dict[str, object], *, threshold: float = 0.5) -> Tuple[float, int]:
    x = preprocess_single(sample)
    with torch.no_grad():
        prob = float(_model(x).item())
    decision = int(prob >= threshold)
    return prob, decision


# ────────────────────────────────────────────────────────────────
# 5.  Optional: interactive demo if run directly
# ────────────────────────────────────────────────────────────────
def _interactive():
    raw_cols = list(_META.keys())
    print("\nEnter values (press Return after each):")
    example = {}
    for col in raw_cols:
        meta = _META[col]
        if meta["kind"] == "numeric":
            hint = f"{meta['min']}–{meta['max']}, mean={meta['mean']}"
        elif meta["kind"] == "categorical":
            if meta["n_unique"] <= 2:
                hint = f"{meta['choices'][0]}|{meta['choices'][1]}"
            else:
                preview = "|".join(str(c) for c in meta['choices'])
                hint = f"choices={preview}{'…' if meta['n_unique']>6 else ''}"
        else:
            hint = "value"

        raw = input(f"  {col} [{hint}] : ").strip()
        # naive cast
        if meta["kind"] == "numeric":
            try:
                raw = float(raw)
            except ValueError:
                pass
        example[col] = raw

    prob, decision = predict(example)
    print(f"\nProbability = {prob:.3f}")
    print(f"Decision    = {decision} "
          f"({'Positive' if decision else 'Negative'})")


if __name__ == "__main__":
    _interactive()