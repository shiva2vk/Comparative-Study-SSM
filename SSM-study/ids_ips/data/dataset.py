"""
Dataset loader and preprocessor.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]
CAT2IDX: Dict[str, int] = {c: i for i, c in enumerate(ATTACK_CATEGORIES)}

CAT_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat", "label"]

class UNSWN15Preprocessor:
    """
    Stateful preprocessor: fit on train, transform on train/test.
    Stores label encoders and scaler for reproducibility.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_cols: Optional[list] = None

    _RAW_TO_CANON = {c.lower(): c for c in ATTACK_CATEGORIES}

    def _load_raw(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False)

        df["attack_cat"] = (
            df["attack_cat"]
            .str.strip()
            .str.lower()
            .map(lambda x: self._RAW_TO_CANON.get(x, "Normal"))
        )
        return df

    def fit_transform(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and fit-transform the training CSV. Returns (X, y)."""
        df = self._load_raw(csv_path)

        for col in CAT_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        y = df["attack_cat"].map(CAT2IDX).values.astype(np.int64)

        X_df = df.drop(columns=DROP_COLS, errors="ignore")
        self.feature_cols = list(X_df.columns)

        X = self.scaler.fit_transform(X_df.values.astype(np.float32))
        return X, y

    def transform(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and transform a held-out CSV using fitted parameters."""
        df = self._load_raw(csv_path)

        for col in CAT_COLS:
            le = self.label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

        y = df["attack_cat"].map(CAT2IDX).values.astype(np.int64)

        X_df = df.drop(columns=DROP_COLS, errors="ignore")

        X_df = X_df[self.feature_cols]
        X = self.scaler.transform(X_df.values.astype(np.float32))
        return X, y

    @property
    def n_features(self) -> int:
        return len(self.feature_cols) if self.feature_cols else 0

def apply_smote(X: np.ndarray, y: np.ndarray,
                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance minority attack classes."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(
            "SMOTE applied: %d  %d samples", len(y), len(y_res)
        )
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not available; skipping SMOTE.")
        return X, y

class NetworkFlowDataset(Dataset):
    """
    Wraps (X, y) numpy arrays as a PyTorch Dataset.

    Each sample x has shape (n_features,).
    Models receive it as (batch, n_features) and internally reshape
    to (batch, seq_len, d_in) for sequence processing.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def get_kfold_splits(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     seed: int = 42):
    """Yield (train_idx, val_idx) for stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(X, y)

def load_unsw_nb15(
    train_csv: Path,
    test_csv: Path,
    use_smote: bool = False,
    seed: int = 42,
) -> Tuple["UNSWN15Preprocessor", NetworkFlowDataset, NetworkFlowDataset]:
    """
    Full pipeline: load  encode  scale  (optionally SMOTE)  Dataset.

    Returns
    -------
    preprocessor : fitted UNSWN15Preprocessor
    train_ds     : NetworkFlowDataset for training
    test_ds      : NetworkFlowDataset for held-out evaluation
    """
    preprocessor = UNSWN15Preprocessor()

    logger.info("Loading training data from %s", train_csv)
    X_train, y_train = preprocessor.fit_transform(train_csv)
    logger.info(
        "Train: %d samples, %d features, %d classes",
        len(y_train), X_train.shape[1], len(np.unique(y_train))
    )

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=seed)

    logger.info("Loading test data from %s", test_csv)
    X_test, y_test = preprocessor.transform(test_csv)
    logger.info("Test:  %d samples", len(y_test))

    train_ds = NetworkFlowDataset(X_train, y_train)
    test_ds  = NetworkFlowDataset(X_test, y_test)
    return preprocessor, train_ds, test_ds
