"""The bid optimizer — turns model predictions into an optimal, shaded bid.

Economics
---------
For an impression with context ``x`` the expected value we get *if we win* is

    EV(x) = P(click | x) · P(conversion | click, x) · value_per_conversion

In a first-price auction we pay our bid ``b`` when we win, so the expected surplus is

    surplus(b) = P(win | b, x) · (EV(x) − b)

We pick the bid that maximizes expected surplus over a price grid. Because winning
gets more likely as ``b`` rises but the per-win surplus shrinks, the optimum is
typically *below* EV — i.e. the model performs **bid shading**, like a real DSP.
An optional ``pacing`` factor (0–1) scales willingness-to-pay down when a campaign
budget is depleting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from smartbidder import features
from smartbidder.model_store import ModelStore, get_store

# Resolution of the bid search grid (points between 0 and EV).
_GRID_POINTS = 40


@dataclass
class BidDecision:
    bid: float
    p_ctr: float
    p_conversion: float
    expected_value: float
    value_per_conversion: float
    win_probability: float
    expected_surplus: float
    should_bid: bool
    pacing: float
    model_version: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _no_bid(p_ctr, p_conv, ev, value, pacing, version) -> BidDecision:
    return BidDecision(
        bid=0.0, p_ctr=p_ctr, p_conversion=p_conv, expected_value=round(ev, 4),
        value_per_conversion=value, win_probability=0.0, expected_surplus=0.0,
        should_bid=False, pacing=pacing, model_version=version,
    )


def optimize_bid(context: Dict, pacing: float = 1.0,
                 store: Optional[ModelStore] = None) -> BidDecision:
    """Compute the optimal shaded bid for a single auction context."""
    store = store or get_store()
    pacing = float(np.clip(pacing, 0.0, 1.0))

    enriched = features.enrich_context(context)
    base_row = store.encoder.transform_row(enriched)  # (1, n_features)

    p_ctr = float(store.predict_ctr(base_row)[0])
    p_conv = float(store.predict_cvr(base_row)[0])

    # Use the explicit per-conversion value if provided (including 0), else the
    # per-category default. `is None` check is deliberate — 0.0 is a valid value.
    value = context.get("value_per_conversion")
    if value is None:
        value = store.category_value.get(context.get("ad_category"), 0.0)
    value = float(value)
    expected_value = p_ctr * p_conv * value
    willingness = expected_value * pacing

    if willingness <= 1e-6:
        return _no_bid(p_ctr, p_conv, expected_value, value, pacing, store.model_version)

    # Evaluate the whole bid grid through the win-rate model in one forward pass.
    bids = np.linspace(1e-3, willingness, _GRID_POINTS, dtype=np.float32)
    grid = np.repeat(base_row, _GRID_POINTS, axis=0)
    grid = np.hstack([grid, np.log1p(bids).reshape(-1, 1)]).astype(np.float32)
    win_probs = store.predict_winrate(grid)
    surplus = win_probs * (expected_value - bids)

    best = int(np.argmax(surplus))
    if float(surplus[best]) <= 0:
        return _no_bid(p_ctr, p_conv, expected_value, value, pacing, store.model_version)

    return BidDecision(
        bid=round(float(bids[best]), 4),
        p_ctr=p_ctr,
        p_conversion=p_conv,
        expected_value=round(expected_value, 4),
        value_per_conversion=value,
        win_probability=round(float(win_probs[best]), 4),
        expected_surplus=round(float(surplus[best]), 4),
        should_bid=True,
        pacing=pacing,
        model_version=store.model_version,
    )
