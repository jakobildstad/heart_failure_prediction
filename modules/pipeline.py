"""
main.py
-------

Single entry point.  Run `python main.py` from the repo root
and the script will:

    1. Pre-process data
    2. Train the network (with early stopping)
    3. Evaluate on hold-out test split
    4. Save artefacts (scaler + model weights)

Feel free to tweak hyper-parameters here and re-run.
"""

from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader

from modules.preprocess import (
    load_data,
    prepare_features,
    TabularDataset,
    save_scaler,
)
from modules.trainer import HeartNet, train, evaluate

# ────────────────────────────────────────────────────────────────
# ✦ 0. Config  (change here, not in library code)
# ────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/heart.csv")
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cpu"  # change to "cuda" if you have a GPU


def training_pipeline():
    # 1. Load + preprocess
    print("Loading data …")
    df = load_data(DATA_PATH)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        feature_names,
    ) = prepare_features(df)

    # Save scaler for production use
    save_scaler(scaler, "model/scaler.pkl")

    # 2. Build DataLoaders
    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TabularDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    # 3. Instantiate model
    model = HeartNet(num_features=X_train.shape[1])
    print(model)

    # 4. Train
    model = train(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE,
        patience=10,
        ckpt_path="model/best_heartnet.pth",
    )

    # 5. Evaluate
    metrics = evaluate(model, test_loader, device=DEVICE)
    print("\nTest-set metrics:")
    for k, v in metrics.items():
        print(f"  {k:>8s}: {v:.4f}")

    # 6. Save final model (already saved best weights inside `train()`)
    with open("model/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    torch.save(model.state_dict(), "model/best_heartnet.pth")
    print("\nAll artefacts saved - ready for inference!")