"""
modules/preprocess.py
---------------------

Utility functions that *only* deal with tabular data preparation.
Keeping this logic out of the training loop lets you reuse it
for any future model without touching the ML code.
"""

from pathlib import Path
import pickle
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


# ────────────────────────────────────────────────────────────────
# ✦ 1. Load raw CSV
# ────────────────────────────────────────────────────────────────
def load_data(csv_path: Path | str) -> pd.DataFrame:
    """
    Read the raw CSV into a pandas DataFrame.  We don't touch
    anything here - just load & return - because you sometimes
    want a quick look before preprocessing.
    """
    df = pd.read_csv(csv_path)
    return df


# ────────────────────────────────────────────────────────────────
# ✦ 2. Convert categories → numbers, scale numerics
# ────────────────────────────────────────────────────────────────
def prepare_features(
    df: pd.DataFrame,
    target_col: str = "HeartDisease",
    test_size: float = 0.15, # proportion of data for test set (0-1)
    val_size: float = 0.15, # proportion of data for validation set (0-1)
    # random_state for reproducibility
    # This ensures that the same random splits are made every time you run the code.
    random_state: int = 42,
) -> Tuple[
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    StandardScaler, 
    List[str]
]:
    """
    Returns:
        X_train, y_train, X_val, y_val, X_train, y_train as *Torch tensors*
        fitted StandardScaler
        feature_names (list) - useful when you deploy and need to know order
    Steps:
        1. One-hot encode *all* categorical columns
        2. Split into train / val / test
        3. Fit scaler *only* on train numerics
        4. Convert to float32 tensors
    """
    df = df.copy() #create a copy of the DataFrame to avoid modifying the original data

    # 1. Separate target -----------------------------------------
    y = df[target_col].astype("float32").values  # shape (N,) #Fetches the target column
    X = df.drop(columns=[target_col]) # Drop the target column from the DataFrame to get the feature matrix

    # Identify categorical columns automatically (object or category dtype)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # One-hot encode categoricals – get_dummies preserves column order
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_names = X.columns.tolist()

    # 2. Train / temp split first, then temp → val / test --------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state, stratify=np.array(y)
    )
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state, stratify=y_temp
    )

    # 3. Scale numerics (fit *only* on train)
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 4. Convert to tensors --------------------------------------
    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr.astype("float32"))

    X_train, y_train = to_tensor(X_train.values), to_tensor(y_train)
    X_val, y_val = to_tensor(X_val.values), to_tensor(y_val)
    X_test, y_test = to_tensor(X_test.values), to_tensor(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names


# ────────────────────────────────────────────────────────────────
# ✦ 3. PyTorch Dataset
# ────────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    """
    Minimal wrapper so DataLoader can batch the rows.

    __len__  tells PyTorch how many samples
    __getitem__ returns (features, label) pair as tensors
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ────────────────────────────────────────────────────────────────
# ✦ 4. Convenience saver / loader for scaler when you deploy
# ────────────────────────────────────────────────────────────────
def save_scaler(scaler: StandardScaler, path: Path | str):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path | str) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)