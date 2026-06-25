"""Pydantic request/response models for the SmartBidder API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AuctionContext(BaseModel):
    """The bid-request context the engine scores. Mirrors a real OpenRTB bid request."""

    audience_segment: str = Field(..., examples=["tech_enthusiasts"])
    ad_category: str = Field(..., examples=["electronics"])
    device_type: str = Field(..., examples=["mobile"])
    ad_position: str = Field(..., examples=["top"])
    ad_size: str = Field(..., examples=["300x250"])
    user_gender: str = Field("M", examples=["M", "F", "O"])
    user_age: int = Field(35, ge=13, le=100)
    hour_of_day: int = Field(12, ge=0, le=23)
    day_of_week: int = Field(2, ge=0, le=6)
    base_cpm: float = Field(10.0, ge=0)
    value_per_conversion: Optional[float] = Field(
        None, description="Advertiser value of a conversion (USD). Defaults to per-category value."
    )
    pacing: float = Field(1.0, ge=0.0, le=1.0,
                          description="Budget pacing factor; 1.0 = bid full value.")


class BidResponse(BaseModel):
    bid: float
    p_ctr: float
    p_conversion: float
    expected_value: float
    value_per_conversion: float
    win_probability: float
    expected_surplus: float
    should_bid: bool
    pacing: float
    model_version: Optional[str]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_version: Optional[str]
