"""Feature engineering shared between training and serving.

Keeping this in one place guarantees the exact same transformation is applied when
a model is trained and when it scores a live auction — the classic source of
train/serve skew is avoided by construction.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Raw context fields that arrive with every auction (what the API receives).
CATEGORICAL_FEATURES: List[str] = [
    "audience_segment",
    "ad_category",
    "device_type",
    "ad_position",
    "ad_size",
    "user_gender",
]

# Numeric fields fed to the model (cyclical encodings are derived below).
NUMERIC_FEATURES: List[str] = [
    "base_cpm",
    "user_age",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

MODEL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine encodings of hour-of-day and day-of-week.

    Cyclical encoding lets the model treat 23:00 and 00:00 as adjacent rather than
    maximally distant — important for time-of-day engagement patterns.
    """
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame containing exactly ``MODEL_FEATURES`` columns."""
    enriched = add_cyclical_features(df)
    missing = [c for c in MODEL_FEATURES if c not in enriched.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return enriched[MODEL_FEATURES]


def context_to_frame(context: Dict) -> pd.DataFrame:
    """Turn a single auction-context dict into a one-row feature-ready DataFrame."""
    return build_feature_frame(pd.DataFrame([context]))


def enrich_context(context: Dict) -> Dict:
    """Add derived cyclical numeric fields to a raw context dict (for fast row serving)."""
    hour = context.get("hour_of_day", 0)
    dow = context.get("day_of_week", 0)
    enriched = dict(context)
    enriched["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    enriched["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    enriched["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    enriched["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    return enriched
