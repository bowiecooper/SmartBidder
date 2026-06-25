"""Loads trained models + metadata once and serves them to the rest of the app.

A single module-level ``ModelStore`` keeps the XGBoost boosters, isotonic calibrators
and feature encoder warm in memory so per-request scoring is just a NumPy featurize +
native booster forward pass — essential for the single-digit-millisecond latency budget
of real-time bidding.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, Optional

import joblib
import numpy as np

from smartbidder import config


class ModelStore:
    def __init__(self) -> None:
        self.encoder = None
        self.ctr_model = None
        self.ctr_calib = None
        self.cvr_model = None
        self.cvr_calib = None
        self.winrate_model = None
        self.winrate_calib = None
        self.metadata: Dict = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> Optional[str]:
        return self.metadata.get("model_version")

    @property
    def category_value(self) -> Dict[str, float]:
        return self.metadata.get("category_value", {})

    def load(self) -> "ModelStore":
        if self._loaded:
            return self
        required = [
            config.ENCODER_PATH, config.CTR_MODEL_PATH, config.CTR_CALIB_PATH,
            config.CVR_MODEL_PATH, config.CVR_CALIB_PATH, config.WINRATE_MODEL_PATH,
            config.WINRATE_CALIB_PATH, config.METADATA_PATH,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(
                    f"Model artifact missing: {path}. Run `python -m smartbidder.train` first."
                )
        self.encoder = joblib.load(config.ENCODER_PATH)
        self.ctr_model = joblib.load(config.CTR_MODEL_PATH)
        self.ctr_calib = joblib.load(config.CTR_CALIB_PATH)
        self.cvr_model = joblib.load(config.CVR_MODEL_PATH)
        self.cvr_calib = joblib.load(config.CVR_CALIB_PATH)
        self.winrate_model = joblib.load(config.WINRATE_MODEL_PATH)
        self.winrate_calib = joblib.load(config.WINRATE_CALIB_PATH)
        with open(config.METADATA_PATH) as f:
            self.metadata = json.load(f)
        self._loaded = True
        return self

    # ---- fast inference helpers (operate on pre-encoded NumPy rows) ---- #
    def predict_ctr(self, X: np.ndarray) -> np.ndarray:
        return self.ctr_calib.transform(self.ctr_model.get_booster().inplace_predict(X))

    def predict_cvr(self, X: np.ndarray) -> np.ndarray:
        return self.cvr_calib.transform(self.cvr_model.get_booster().inplace_predict(X))

    def predict_winrate(self, X: np.ndarray) -> np.ndarray:
        return self.winrate_calib.transform(self.winrate_model.get_booster().inplace_predict(X))


@lru_cache(maxsize=1)
def get_store() -> ModelStore:
    """Return the process-wide, lazily loaded model store."""
    return ModelStore().load()
