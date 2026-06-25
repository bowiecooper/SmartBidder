"""SmartBidder FastAPI service.

Endpoints
---------
GET  /health        liveness + model version
GET  /metrics       model evaluation metadata + live latency/throughput
GET  /options       valid categorical values (powers the dashboard form)
POST /bid           score one auction -> optimal shaded bid (with latency_ms)
WS   /ws/stream     live simulated auction feed with bid decisions
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from data.data_generator import AdAuctionDataGenerator as G
from smartbidder import simulator
from smartbidder.bidder import optimize_bid
from smartbidder.model_store import get_store
from smartbidder.schemas import AuctionContext, BidResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load + warm the models so the first real request isn't penalized.
    store = get_store()
    optimize_bid(simulator.random_auction(np.random.default_rng(0)), store=store)
    yield


app = FastAPI(
    title="SmartBidder",
    description="Real-time ML engine for optimizing ad bids and audience matching.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo service; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class LiveStats:
    """Rolling latency/throughput counters for the /metrics endpoint."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._latencies: Deque[float] = deque(maxlen=maxlen)
        self.total_bids = 0
        self.total_wins = 0
        self.total_surplus = 0.0

    def record(self, latency_ms: float, won: bool, surplus: float) -> None:
        self._latencies.append(latency_ms)
        self.total_bids += 1
        self.total_wins += int(won)
        self.total_surplus += surplus

    def snapshot(self) -> Dict:
        lat = list(self._latencies)
        return {
            "total_bids": self.total_bids,
            "win_rate": (self.total_wins / self.total_bids) if self.total_bids else 0.0,
            "avg_surplus": (self.total_surplus / self.total_bids) if self.total_bids else 0.0,
            "latency_ms": {
                "p50": float(np.percentile(lat, 50)) if lat else 0.0,
                "p95": float(np.percentile(lat, 95)) if lat else 0.0,
                "p99": float(np.percentile(lat, 99)) if lat else 0.0,
            },
        }


stats = LiveStats()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    return HealthResponse(status="ok", models_loaded=store.loaded,
                          model_version=store.model_version)


@app.get("/metrics")
def metrics() -> Dict:
    store = get_store()
    md = store.metadata
    return {
        "model_version": md.get("model_version"),
        "trained_at": md.get("trained_at"),
        "n_samples": md.get("n_samples"),
        "model_metrics": md.get("metrics"),
        "feature_importances": md.get("feature_importances"),
        "data_stats": md.get("data_stats"),
        "live": stats.snapshot(),
    }


@app.get("/options")
def options() -> Dict:
    return {
        "audience_segments": G.AUDIENCE_SEGMENTS,
        "ad_categories": G.AD_CATEGORIES,
        "device_types": G.DEVICE_TYPES,
        "ad_positions": G.AD_POSITIONS,
        "ad_sizes": G.AD_SIZES,
        "category_value": G.CATEGORY_VALUE,
    }


@app.post("/bid", response_model=BidResponse)
def bid(context: AuctionContext) -> BidResponse:
    start = time.perf_counter()
    decision = optimize_bid(context.model_dump(), pacing=context.pacing)
    latency_ms = (time.perf_counter() - start) * 1000.0
    stats.record(latency_ms, decision.should_bid, decision.expected_surplus)
    return BidResponse(latency_ms=round(latency_ms, 3), **decision.to_dict())


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    """Stream simulated auctions and live bid decisions to the dashboard."""
    await websocket.accept()
    rng = np.random.default_rng()
    store = get_store()
    try:
        while True:
            ctx = simulator.random_auction(rng)
            start = time.perf_counter()
            decision = optimize_bid(ctx, store=store)
            latency_ms = (time.perf_counter() - start) * 1000.0
            # Simulate the market outcome: did we actually win at our bid?
            won = decision.should_bid and (rng.random() < decision.win_probability)
            stats.record(latency_ms, decision.should_bid, decision.expected_surplus)
            await websocket.send_json({
                "context": ctx,
                "decision": {**decision.to_dict(), "latency_ms": round(latency_ms, 3)},
                "won": bool(won),
            })
            await asyncio.sleep(0.7)
    except WebSocketDisconnect:
        return
