"""
main.py  –  Universal CLI entry point
=====================================

$ python main.py train
$ python main.py evaluate -m model/best_heartnet.pth
$ python main.py predict  -m model/best_heartnet.pth
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pickle

import pandas as pd
import torch

# ────────────────────────────────────────────────────────────────
#  0.  Import *your* library code
#      (make sure modules/ is a package → add an empty __init__.py)
# ────────────────────────────────────────────────────────────────
from modules.pipeline import training_pipeline
from modules.preprocess import (
    load_data,
    prepare_features,
    TabularDataset,
)
from modules.trainer import HeartNet, evaluate                    # includes FMR / FNMR


# ────────────────────────────────────────────────────────────────
#  1.  CLI set-up
# ────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Heart-disease net – CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train ----------------------------------------------------------------
    train_p = sub.add_parser("train", help="Run full training pipeline")
    train_p.add_argument(
        "--csv", type=Path, default=Path("data/heart.csv"),
        help="Path to raw CSV")

    # evaluate -------------------------------------------------------------
    eval_p = sub.add_parser("evaluate", help="Evaluate saved model")
    eval_p.add_argument("-m", "--model", type=Path, required=True,
                        help="Path to .pth weights file")
    eval_p.add_argument("--csv", type=Path, default=Path("data/heart.csv"))
    eval_p.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold (omit to use EER point)")

    # predict --------------------------------------------------------------
    pred_p = sub.add_parser("predict", help="Interactive prediction")
    pred_p.add_argument("-m", "--model", type=Path, required=True,
                        help="Path to .pth weights file")
    pred_p.add_argument("--scaler", type=Path, default=Path("model/scaler.pkl"))
    pred_p.add_argument("--features", type=Path, default=Path("model/feature_names.pkl"))
    pred_p.add_argument("--threshold", type=float, default=0.5)

    return p


# ────────────────────────────────────────────────────────────────
#  2.  Helpers
# ────────────────────────────────────────────────────────────────
def load_artefacts(
    model_path: Path,
    scaler_path: Path = Path("model/scaler.pkl"),
    features_path: Path = Path("model/feature_names.pkl"),
) -> tuple[HeartNet, object, list[str]]:
    """Load network weights, StandardScaler, & feature-name order."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)

    net = HeartNet(num_features=len(feature_names))
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net, scaler, feature_names


# --------------------------------------------------------------------------------
# helper that runs *once* to grab metadata from the raw CSV
# --------------------------------------------------------------------------------
def _column_metadata(csv_path: Path = Path("data/heart.csv")) -> dict[str, dict]:
    df = pd.read_csv(csv_path)

    meta: dict[str, dict] = {}
    for col in df.columns:
        if col == "HeartDisease":          # skip label
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
                "choices": uniq[:6],        # truncate long lists
                "n_unique": len(uniq),
            }
    return meta


# cache so we don’t read the CSV every time
_META = _column_metadata()


# --------------------------------------------------------------------------------
# interactive prompt
# --------------------------------------------------------------------------------
def interactive_example(raw_columns: list[str]) -> dict:
    """
    Ask the user for each raw feature value via stdin, displaying helpful
    type / range / choice hints derived from heart.csv.
    """
    print("\nEnter values (press Return after each):")
    example: dict[str, object] = {}

    for col in raw_columns:
        m = _META.get(col, {"kind": "unknown"})

        if m["kind"] == "numeric":
            hint = f"numeric, {m['min']}–{m['max']}, mean={m['mean']}"
        elif m["kind"] == "categorical":
            if m["n_unique"] <= 2:
                hint = f"binary {m['choices'][0]}|{m['choices'][1]}"
            else:
                choice_preview = "|".join(str(c) for c in m["choices"])
                more = "…" if m["n_unique"] > 6 else ""
                hint = f"categorical, choices={choice_preview}{more}"
        else:
            hint = "value"

        # prompt ↓
        raw = input(f"  {col} [{hint}] : ").strip()

        # cast if numeric
        if m["kind"] == "numeric":
            try:
                raw = float(raw)
            except ValueError:
                print(f"    ⚠️  Expected a number; keeping as string.")
        example[col] = raw

    return example


# ────────────────────────────────────────────────────────────────
#  3.  Main dispatch
# ────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        # just pass through to your pipeline
        training_pipeline()
        return

    if args.cmd == "evaluate":
        net, scaler, feature_names = load_artefacts(args.model)
        # Re-prep the DATASET exactly like training did
        df = load_data(args.csv)
        (
            _Xtr, _ytr, _Xval, _yval, Xtest, ytest, _sc, _feat_names
        ) = prepare_features(df)
        # Note: we ignore the scaler returned here – we already loaded the
        # one fitted on TRAIN; DataLoader uses *those* scaled values.
        from torch.utils.data import DataLoader, TensorDataset
        test_loader = DataLoader(
            TensorDataset(Xtest, ytest), batch_size=256, shuffle=False
        )
        metrics = evaluate(
            net, test_loader, device="cpu", threshold=args.threshold
        )
        print("\nTest-set metrics:")
        for k, v in metrics.items():
            print(f"  {k:>10s}: {v:.4f}")
        return

    if args.cmd == "predict":
        net, scaler, feature_names = load_artefacts(
            args.model, args.scaler, args.features
        )
        # reconstruct raw column list by reading *one* row of CSV header
        raw_cols = pd.read_csv("data/heart.csv", nrows=0).columns.tolist()
        raw_cols.remove("HeartDisease")  # drop label
        sample = interactive_example(raw_cols)

        # --- reuse the preprocess_single logic from predict.py ----------
        from modules.predict import preprocess_single as _pre  # type: ignore

        x = _pre(sample)
        with torch.no_grad():
            prob = float(net(x).item())
        decision = int(prob >= args.threshold)
        print(f"\nProbability = {prob:.3f}")
        print(f"Decision    = {decision} "
              f"({'Positive' if decision else 'Negative'})")
        return


if __name__ == "__main__":
    main()