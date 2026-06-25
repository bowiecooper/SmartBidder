"""Auction stream simulator.

Produces realistic random bid-requests (using the same vocabulary the models were
trained on) so the dashboard can show a live feed of the engine making decisions.
This is what makes "real-time" tangible in the demo.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from data.data_generator import AdAuctionDataGenerator as G

# Sample from the same distributions used in training so the feed looks realistic.
_DEVICE_P = [0.6, 0.3, 0.1]
_POS_P = [0.3, 0.2, 0.3, 0.2]
_SIZE_P = [0.4, 0.3, 0.2, 0.1]


def random_auction(rng: np.random.Generator) -> Dict:
    segment = rng.choice(G.AUDIENCE_SEGMENTS)
    category = rng.choice(G.AD_CATEGORIES)
    return {
        "audience_segment": str(segment),
        "ad_category": str(category),
        "device_type": str(rng.choice(G.DEVICE_TYPES, p=_DEVICE_P)),
        "ad_position": str(rng.choice(G.AD_POSITIONS, p=_POS_P)),
        "ad_size": str(rng.choice(G.AD_SIZES, p=_SIZE_P)),
        "user_gender": str(rng.choice(["M", "F", "O"], p=[0.48, 0.48, 0.04])),
        "user_age": int(np.clip(rng.normal(35, 10), 18, 65)),
        "hour_of_day": int(rng.integers(0, 24)),
        "day_of_week": int(rng.integers(0, 7)),
        "base_cpm": float(rng.lognormal(2.5, 0.5)),
        "value_per_conversion": float(G.CATEGORY_VALUE[str(category)]),
    }
