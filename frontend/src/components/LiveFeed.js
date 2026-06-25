// Live auction feed driven by the backend WebSocket. Each row is one simulated
// bid-request and the engine's real-time decision.
import React from "react";

export default function LiveFeed({ events, connected }) {
  return (
    <div className="card feed-card">
      <h2>Live Auction Stream
        <span className={`dot ${connected ? "on" : "off"}`} />
        <span className="feed-sub">{connected ? "streaming" : "disconnected"}</span>
      </h2>
      <div className="feed">
        <div className="feed-head">
          <span>Audience → Ad</span><span>EV</span><span>Bid</span>
          <span>Win%</span><span>Result</span><span>ms</span>
        </div>
        {events.map((e, i) => {
          const d = e.decision;
          const match = e.context.audience_segment.split("_")[0];
          return (
            <div className={`feed-row ${i === 0 ? "fresh" : ""}`} key={e._id}>
              <span className="pair">
                <b>{match}</b> → {e.context.ad_category}
              </span>
              <span>${d.expected_value.toFixed(2)}</span>
              <span>{d.should_bid ? `$${d.bid.toFixed(2)}` : "—"}</span>
              <span>{(d.win_probability * 100).toFixed(0)}%</span>
              <span className={e.won ? "won" : d.should_bid ? "lost" : "skip"}>
                {e.won ? "WON" : d.should_bid ? "lost" : "skip"}
              </span>
              <span className="ms">{d.latency_ms.toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
