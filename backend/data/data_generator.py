"""Synthetic ad-auction data generator with *learnable* structure.

Unlike a naive generator that draws click/conversion rates from feature-independent
distributions (which leaves nothing for a model to learn), this generator wires the
outcome probabilities to the auction context through an interpretable latent model:

    logit(p_click) = base
                     + affinity(audience_segment, ad_category)   # "audience matching"
                     + device_effect + position_effect + size_effect
                     + hour_of_day_effect + recency/age effects
                     + gaussian noise

Binary ``clicked`` / ``converted`` outcomes are then sampled from these probabilities.
Because the signal is real but noisy, a well-tuned model lands at a realistic AUC
(~0.75-0.85) rather than a suspicious 1.0 — exactly what you want for a portfolio piece.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from faker import Faker


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AdAuctionDataGenerator:
    """Generate simulated real-time-bidding (RTB) ad-auction impressions."""

    AUDIENCE_SEGMENTS = [
        "tech_enthusiasts", "fashion_shoppers", "sports_fans",
        "travel_enthusiasts", "food_lovers", "business_professionals",
        "students", "parents", "gaming_community",
    ]

    AD_CATEGORIES = [
        "electronics", "fashion", "sports", "travel", "food",
        "business", "education", "family", "gaming",
    ]

    DEVICE_TYPES = ["mobile", "desktop", "tablet"]
    AD_POSITIONS = ["top", "sidebar", "content", "bottom"]
    AD_SIZES = ["300x250", "728x90", "160x600", "320x50"]

    # Per-category advertiser value of a conversion (USD). Used downstream by the
    # bid optimizer to turn predicted probabilities into an expected impression value.
    CATEGORY_VALUE = {
        "electronics": 45.0, "fashion": 30.0, "sports": 25.0, "travel": 90.0,
        "food": 15.0, "business": 70.0, "education": 50.0, "family": 20.0,
        "gaming": 35.0,
    }

    # Strong affinities: which segments naturally engage with which categories.
    # Everything else gets a mild/negative baseline. This is the "audience matching"
    # signal — the single biggest driver of click probability.
    _AFFINITY_PAIRS = {
        "tech_enthusiasts": {"electronics": 1.6, "gaming": 1.0},
        "fashion_shoppers": {"fashion": 1.8, "travel": 0.4},
        "sports_fans": {"sports": 1.7, "food": 0.4},
        "travel_enthusiasts": {"travel": 1.8, "food": 0.5},
        "food_lovers": {"food": 1.7, "family": 0.4},
        "business_professionals": {"business": 1.6, "education": 0.9, "travel": 0.5},
        "students": {"education": 1.5, "gaming": 0.8, "electronics": 0.5},
        "parents": {"family": 1.8, "education": 0.6, "food": 0.4},
        "gaming_community": {"gaming": 1.9, "electronics": 0.8},
    }

    DEVICE_EFFECT = {"mobile": 0.15, "desktop": 0.0, "tablet": -0.10}
    POSITION_EFFECT = {"top": 0.55, "content": 0.20, "sidebar": -0.15, "bottom": -0.45}
    SIZE_EFFECT = {"300x250": 0.20, "728x90": 0.05, "160x600": -0.10, "320x50": -0.20}

    # Conversion (post-click) drivers. These vary independently of the click decision,
    # so they remain learnable even though clicked rows are already affinity-selected.
    CONV_DEVICE_EFFECT = {"desktop": 0.75, "tablet": 0.15, "mobile": -0.40}
    CONV_CATEGORY_EFFECT = {
        "travel": 0.9, "business": 0.75, "education": 0.6, "electronics": 0.3,
        "fashion": 0.0, "gaming": -0.15, "sports": -0.3, "family": -0.45, "food": -0.75,
    }

    def __init__(self, seed: int = 42):
        self.faker = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._rng = np.random.default_rng(seed)
        self.affinity = self._build_affinity_matrix()

    def _build_affinity_matrix(self) -> Dict[str, Dict[str, float]]:
        matrix: Dict[str, Dict[str, float]] = {}
        for seg in self.AUDIENCE_SEGMENTS:
            row = {cat: -0.4 for cat in self.AD_CATEGORIES}  # mild mismatch baseline
            for cat, strength in self._AFFINITY_PAIRS.get(seg, {}).items():
                row[cat] = strength
            matrix[seg] = row
        return matrix

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_auction_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate ``n_samples`` simulated impressions with binary outcomes."""
        segments = self._rng.choice(self.AUDIENCE_SEGMENTS, size=n_samples)
        categories = self._rng.choice(self.AD_CATEGORIES, size=n_samples)
        devices = self._rng.choice(self.DEVICE_TYPES, size=n_samples, p=[0.6, 0.3, 0.1])
        positions = self._rng.choice(self.AD_POSITIONS, size=n_samples, p=[0.3, 0.2, 0.3, 0.2])
        sizes = self._rng.choice(self.AD_SIZES, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
        hours = self._rng.integers(0, 24, size=n_samples)
        days = self._rng.integers(0, 7, size=n_samples)
        ages = np.clip(self._rng.normal(35, 10, n_samples).astype(int), 18, 65)
        genders = self._rng.choice(["M", "F", "O"], size=n_samples, p=[0.48, 0.48, 0.04])

        affinity = np.array([self.affinity[s][c] for s, c in zip(segments, categories)])
        device_eff = np.array([self.DEVICE_EFFECT[d] for d in devices])
        pos_eff = np.array([self.POSITION_EFFECT[p] for p in positions])
        size_eff = np.array([self.SIZE_EFFECT[s] for s in sizes])

        # Evening/prime-time engagement bump (peaks ~20:00), and a small weekend lift.
        hour_eff = 0.35 * np.cos(2 * np.pi * (hours - 20) / 24)
        weekend_eff = np.where(days >= 5, 0.12, 0.0)
        # Younger users click a touch more; effect is mild and centered.
        age_eff = -0.012 * (ages - 35)

        click_noise = self._rng.normal(0, 0.35, n_samples)
        click_logit = (
            -3.0 + affinity + device_eff + pos_eff + size_eff
            + hour_eff + weekend_eff + age_eff + click_noise
        )
        p_click = _sigmoid(click_logit)
        clicked = (self._rng.random(n_samples) < p_click).astype(int)

        # Conversion is only meaningful conditional on a click. Driven by device,
        # ad-category intent, and age — features that vary among clicked rows, so the
        # CVR model has genuine signal (affinity alone would be near-constant post-click).
        conv_device = np.array([self.CONV_DEVICE_EFFECT[d] for d in devices])
        conv_category = np.array([self.CONV_CATEGORY_EFFECT[c] for c in categories])
        conv_noise = self._rng.normal(0, 0.30, n_samples)
        conv_logit = (
            -1.4 + 0.20 * affinity + conv_device + conv_category
            + 0.25 * (ages - 35) / 10 + conv_noise
        )
        p_conv_given_click = _sigmoid(conv_logit)
        converted = ((clicked == 1) & (self._rng.random(n_samples) < p_conv_given_click)).astype(int)

        base_cpm = self._rng.lognormal(mean=2.5, sigma=0.5, size=n_samples)
        # The market's clearing price, expressed per-impression on the SAME scale as the
        # engine's expected value (EV = pCTR·pConv·value), so bids and prices are
        # directly comparable. It correlates with intrinsic value (affinity), giving the
        # win-rate model real signal — competitors also bid up high-value impressions.
        winning_bid = self._rng.lognormal(mean=-0.4 + 0.15 * affinity, sigma=0.5, size=n_samples)

        df = pd.DataFrame({
            "timestamp": self._generate_timestamps(n_samples),
            "auction_id": [f"AUCTION_{i:06d}" for i in range(n_samples)],
            "base_cpm": base_cpm,
            "winning_bid": winning_bid,
            "audience_segment": segments,
            "ad_category": categories,
            "device_type": devices,
            "hour_of_day": hours,
            "day_of_week": days,
            "ad_position": positions,
            "ad_size": sizes,
            "user_location": [self.faker.country_code() for _ in range(n_samples)],
            "user_age": ages,
            "user_gender": genders,
            "value_per_conversion": [self.CATEGORY_VALUE[c] for c in categories],
            # Latent ground-truth probabilities (kept for analysis/teaching; NOT used as features).
            "true_ctr": p_click,
            "true_cvr": p_conv_given_click,
            # Observed binary outcomes — these are the supervised labels.
            "clicked": clicked,
            "converted": converted,
        })
        return df

    def _generate_timestamps(self, n_samples: int) -> List[datetime]:
        end_date = datetime(2025, 6, 10)
        start_date = end_date - timedelta(days=30)
        return [self.faker.date_time_between(start_date, end_date) for _ in range(n_samples)]


if __name__ == "__main__":
    generator = AdAuctionDataGenerator()
    df = generator.generate_auction_data(n_samples=5000)
    print("Generated dataset shape:", df.shape)
    print(f"\nClick-through rate: {df['clicked'].mean():.3%}")
    print(f"Conversion rate (overall): {df['converted'].mean():.3%}")
    print(f"Conversion rate (of clicks): {df.loc[df.clicked == 1, 'converted'].mean():.3%}")
    print("\nCTR by audience/category affinity (top matches should dominate):")
    pivot = df.pivot_table(values="clicked", index="audience_segment",
                           columns="ad_category", aggfunc="mean")
    print((pivot * 100).round(1))
