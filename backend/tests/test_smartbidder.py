"""Test suite for the SmartBidder engine.

Covers the fast featurizer, the bid-optimization economics, and the API contract.
Models are trained once (small sample) into a temp dir so tests are self-contained
and don't depend on a prior `train` run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from data.data_generator import AdAuctionDataGenerator  # noqa: E402
from smartbidder import features  # noqa: E402
from smartbidder.encoder import FastOneHotEncoder  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _trained_models():
    """Train tiny models once for the whole test session."""
    from smartbidder import train
    train.train(n_samples=8000)


# --------------------------------------------------------------------- #
# Data generator
# --------------------------------------------------------------------- #
def test_generator_produces_binary_labels_and_signal():
    df = AdAuctionDataGenerator(seed=1).generate_auction_data(5000)
    assert set(df["clicked"].unique()) <= {0, 1}
    assert set(df["converted"].unique()) <= {0, 1}
    # A matched segment/category must out-click a mismatched one (audience matching works).
    matched = df[(df.audience_segment == "gaming_community") & (df.ad_category == "gaming")]
    mismatched = df[(df.audience_segment == "gaming_community") & (df.ad_category == "food")]
    assert matched["clicked"].mean() > mismatched["clicked"].mean()
    # Conversions only happen on clicks.
    assert (df.loc[df.clicked == 0, "converted"] == 0).all()


# --------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------- #
def test_encoder_row_matches_batch():
    df = AdAuctionDataGenerator(seed=2).generate_auction_data(200)
    enc = FastOneHotEncoder(features.CATEGORICAL_FEATURES, features.NUMERIC_FEATURES)
    frame = features.build_feature_frame(df)
    enc.fit(frame)
    batch = enc.transform(frame)
    row0 = enc.transform_row(features.enrich_context(df.iloc[0].to_dict()))
    assert batch.shape[1] == enc.n_features_
    np.testing.assert_allclose(batch[0], row0[0], rtol=1e-5)


def test_encoder_handles_unknown_category():
    df = AdAuctionDataGenerator(seed=3).generate_auction_data(100)
    enc = FastOneHotEncoder(features.CATEGORICAL_FEATURES, features.NUMERIC_FEATURES)
    enc.fit(features.build_feature_frame(df))
    row = enc.transform_row(features.enrich_context(
        {"audience_segment": "aliens", "ad_category": "gaming", "device_type": "mobile",
         "ad_position": "top", "ad_size": "300x250", "user_gender": "M",
         "base_cpm": 5.0, "user_age": 30, "hour_of_day": 10, "day_of_week": 2}))
    assert row.shape == (1, enc.n_features_)  # unknown segment -> zero block, no crash


# --------------------------------------------------------------------- #
# Bidder economics
# --------------------------------------------------------------------- #
def _ctx(**kw):
    base = dict(audience_segment="gaming_community", ad_category="gaming",
                device_type="desktop", ad_position="top", ad_size="300x250",
                user_gender="M", user_age=24, hour_of_day=20, day_of_week=5,
                base_cpm=12.0, value_per_conversion=35.0)
    base.update(kw)
    return base


def test_strong_match_beats_weak_match():
    from smartbidder.bidder import optimize_bid
    strong = optimize_bid(_ctx())
    weak = optimize_bid(_ctx(audience_segment="food_lovers", ad_category="electronics",
                             ad_position="bottom", device_type="mobile"))
    assert strong.expected_value > weak.expected_value
    assert strong.p_ctr > weak.p_ctr


def test_bid_is_shaded_below_expected_value():
    from smartbidder.bidder import optimize_bid
    d = optimize_bid(_ctx())
    assert d.should_bid
    assert 0 < d.bid <= d.expected_value  # never bid above expected value
    assert 0.0 <= d.win_probability <= 1.0


def test_zero_value_means_no_bid():
    from smartbidder.bidder import optimize_bid
    d = optimize_bid(_ctx(value_per_conversion=0.0))
    assert not d.should_bid
    assert d.bid == 0.0


def test_pacing_reduces_bid():
    from smartbidder.bidder import optimize_bid
    full = optimize_bid(_ctx(), pacing=1.0)
    throttled = optimize_bid(_ctx(), pacing=0.3)
    assert throttled.bid <= full.bid


# --------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from smartbidder.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["models_loaded"] is True


def test_bid_endpoint(client):
    r = client.post("/bid", json=_ctx())
    assert r.status_code == 200
    body = r.json()
    assert body["should_bid"] is True
    assert 0 < body["bid"] <= body["expected_value"]
    assert body["latency_ms"] >= 0
    assert 0.0 <= body["p_ctr"] <= 1.0


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "model_metrics" in r.json()
    assert r.json()["model_metrics"]["ctr"]["auc"] > 0.5


def test_options_endpoint(client):
    r = client.get("/options")
    assert r.status_code == 200
    assert "gaming" in r.json()["ad_categories"]
