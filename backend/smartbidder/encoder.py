"""A minimal, fast one-hot featurizer.

scikit-learn's ``ColumnTransformer`` carries several milliseconds of Python dispatch
overhead *per call*, independent of batch size. In a real-time bidder that scores one
auction at a time (and evaluates a whole bid grid through the win-rate model), that
overhead dominates latency. This encoder does the same job with plain NumPy so a single
row featurizes in microseconds, keeping the engine inside its single-digit-ms budget.

The layout is deterministic: ``[one-hot blocks for each categorical feature..., numeric features...]``
so the same vector shape is produced at train and serve time.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class FastOneHotEncoder:
    def __init__(self, categorical: List[str], numeric: List[str]) -> None:
        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.categories_: Dict[str, List[str]] = {}
        self.cat_index_: Dict[str, Dict[str, int]] = {}
        self.offsets_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.n_cat_: int = 0
        self.n_features_: int = 0

    def fit(self, df: pd.DataFrame) -> "FastOneHotEncoder":
        offset = 0
        for f in self.categorical:
            cats = sorted(df[f].astype(str).unique().tolist())
            self.categories_[f] = cats
            self.cat_index_[f] = {v: i for i, v in enumerate(cats)}
            self.offsets_[f] = offset
            self.feature_names_ += [f"{f}={c}" for c in cats]
            offset += len(cats)
        self.n_cat_ = offset
        self.feature_names_ += list(self.numeric)
        self.n_features_ = self.n_cat_ + len(self.numeric)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized transform for a batch (training)."""
        n = len(df)
        out = np.zeros((n, self.n_features_), dtype=np.float32)
        for f in self.categorical:
            idx = self.cat_index_[f]
            base = self.offsets_[f]
            cols = df[f].astype(str).map(idx)
            valid = cols.notna().to_numpy()
            rows = np.arange(n)[valid]
            positions = base + cols[valid].astype(int).to_numpy()
            out[rows, positions] = 1.0  # unknown categories -> all-zero block (handle_unknown="ignore")
        for j, f in enumerate(self.numeric):
            out[:, self.n_cat_ + j] = df[f].astype(np.float32).to_numpy()
        return out

    def transform_row(self, context: Dict) -> np.ndarray:
        """Fast single-row transform for serving — no pandas, just dict lookups."""
        out = np.zeros((1, self.n_features_), dtype=np.float32)
        for f in self.categorical:
            pos = self.cat_index_[f].get(str(context.get(f)))
            if pos is not None:
                out[0, self.offsets_[f] + pos] = 1.0
        for j, f in enumerate(self.numeric):
            out[0, self.n_cat_ + j] = float(context.get(f, 0.0))
        return out
