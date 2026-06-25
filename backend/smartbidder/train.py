"""Training pipeline for the SmartBidder RTB engine.

Trains three models and persists them with evaluation metadata:

1. ``ctr_model``     — P(click | context)
2. ``cvr_model``     — P(conversion | click, context)
3. ``winrate_model`` — P(win | bid, context)  (a "bid landscape" model)

Each model is an XGBoost classifier paired with an isotonic **calibrator** fit on a
held-out split — calibration matters because the bid optimizer multiplies these
probabilities together as expected value, so miscalibrated scores produce wrong bids.

Serving uses a hand-rolled NumPy featurizer (``FastOneHotEncoder``) + native XGBoost
``inplace_predict`` so a full bid decision runs in well under a millisecond.

Run:  python -m smartbidder.train [--samples 100000]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.data_generator import AdAuctionDataGenerator  # noqa: E402
from smartbidder import config, features  # noqa: E402
from smartbidder.encoder import FastOneHotEncoder  # noqa: E402

RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"


def _xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,  # bounded: avoids OMP thread oversubscription
        random_state=RANDOM_STATE,
    )


def _predict(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Fast positive-class probability via the native booster."""
    return model.get_booster().inplace_predict(X)


def _eval(y_true: np.ndarray, proba: np.ndarray) -> Dict:
    # Calibration curve points (quantile bins) for the dashboard reliability plot.
    order = np.argsort(proba)
    bins = np.array_split(order, 10)
    mean_pred, frac_pos = [], []
    for b in bins:
        if len(b) == 0:
            continue
        mean_pred.append(round(float(proba[b].mean()), 4))
        frac_pos.append(round(float(y_true[b].mean()), 4))
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
        "n_test": int(len(y_true)),
        "calibration_curve": {"mean_predicted": mean_pred, "fraction_positive": frac_pos},
    }


def _train_one(X: np.ndarray, y: np.ndarray) -> Tuple[XGBClassifier, IsotonicRegression, Dict]:
    """Train an XGB model + isotonic calibrator on disjoint splits; evaluate on a test split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.35, random_state=RANDOM_STATE, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp)

    model = _xgb()
    model.fit(X_tr, y_tr)

    # Fit isotonic calibration on the held-out calibration split.
    raw_cal = _predict(model, X_cal)
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calib.fit(raw_cal, y_cal)

    cal_test = calib.transform(_predict(model, X_te))
    return model, calib, _eval(y_te, cal_test)


def _winrate_examples(df: pd.DataFrame, encoder: FastOneHotEncoder,
                      rng: np.random.Generator, bids_per_auction: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Build (base features + log_bid -> win?) examples for the bid-landscape model."""
    base = encoder.transform(features.build_feature_frame(df))
    market = df["winning_bid"].to_numpy()
    Xs, ys = [], []
    for _ in range(bids_per_auction):
        candidate = market * rng.lognormal(0.0, 0.6, size=len(df))
        logbid = np.log1p(candidate).astype(np.float32).reshape(-1, 1)
        Xs.append(np.hstack([base, logbid]))
        ys.append((candidate >= market).astype(int))
    return np.vstack(Xs), np.concatenate(ys)


def train(n_samples: int = 100_000) -> Dict:
    print(f"[1/6] Generating {n_samples:,} synthetic impressions...")
    gen = AdAuctionDataGenerator(seed=RANDOM_STATE)
    df = gen.generate_auction_data(n_samples=n_samples)
    rng = np.random.default_rng(RANDOM_STATE)

    print("[2/6] Fitting feature encoder...")
    encoder = FastOneHotEncoder(features.CATEGORICAL_FEATURES, features.NUMERIC_FEATURES)
    encoder.fit(features.build_feature_frame(df))
    X = encoder.transform(features.build_feature_frame(df))

    print("[3/6] Training calibrated CTR model  P(click | context)...")
    ctr_model, ctr_calib, ctr_metrics = _train_one(X, df["clicked"].to_numpy())
    print(f"      CTR  AUC={ctr_metrics['auc']:.4f}  logloss={ctr_metrics['log_loss']:.4f}  brier={ctr_metrics['brier']:.4f}")

    print("[4/6] Training calibrated CVR model  P(conversion | click, context)...")
    clicks = df[df["clicked"] == 1]
    Xc = encoder.transform(features.build_feature_frame(clicks))
    cvr_model, cvr_calib, cvr_metrics = _train_one(Xc, clicks["converted"].to_numpy())
    print(f"      CVR  AUC={cvr_metrics['auc']:.4f}  logloss={cvr_metrics['log_loss']:.4f}  brier={cvr_metrics['brier']:.4f}")

    print("[5/6] Training win-rate (bid landscape) model  P(win | bid, context)...")
    Xw, yw = _winrate_examples(df, encoder, rng)
    winrate_model, winrate_calib, winrate_metrics = _train_one(Xw, yw)
    print(f"      WIN  AUC={winrate_metrics['auc']:.4f}  logloss={winrate_metrics['log_loss']:.4f}  brier={winrate_metrics['brier']:.4f}")

    print("[6/6] Persisting artifacts...")
    importances = _feature_importances(ctr_model, encoder)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, config.ENCODER_PATH)
    joblib.dump(ctr_model, config.CTR_MODEL_PATH)
    joblib.dump(ctr_calib, config.CTR_CALIB_PATH)
    joblib.dump(cvr_model, config.CVR_MODEL_PATH)
    joblib.dump(cvr_calib, config.CVR_CALIB_PATH)
    joblib.dump(winrate_model, config.WINRATE_MODEL_PATH)
    joblib.dump(winrate_calib, config.WINRATE_CALIB_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "features": features.MODEL_FEATURES,
        "category_value": AdAuctionDataGenerator.CATEGORY_VALUE,
        "metrics": {"ctr": ctr_metrics, "cvr": cvr_metrics, "winrate": winrate_metrics},
        "feature_importances": importances,
        "data_stats": {
            "overall_ctr": float(df["clicked"].mean()),
            "overall_cvr": float(df["converted"].mean()),
            "conversion_rate_of_clicks": float(df.loc[df.clicked == 1, "converted"].mean()),
        },
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Models + metadata written to {config.MODELS_DIR}")
    return metadata


def _feature_importances(model: XGBClassifier, encoder: FastOneHotEncoder) -> Dict[str, float]:
    importances = model.feature_importances_
    names = encoder.feature_names_
    top = sorted(zip(names, importances), key=lambda kv: kv[1], reverse=True)[:15]
    return {str(n): round(float(v), 4) for n, v in top}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartBidder models.")
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()
    train(n_samples=args.samples)
