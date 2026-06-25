import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { getJSON, WS_BASE } from "./api";
import LiveFeed from "./components/LiveFeed";
import ModelPanel from "./components/ModelPanel";
import BidForm from "./components/BidForm";

const MAX_FEED = 18;

function Kpi({ label, value, sub }) {
  return (
    <div className="kpi">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export default function App() {
  const [options, setOptions] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const idRef = useRef(0);

  // Load static options + poll metrics.
  useEffect(() => {
    getJSON("/options").then(setOptions).catch((e) => setError(String(e)));
    const pull = () => getJSON("/metrics").then(setMetrics).catch(() => {});
    pull();
    const t = setInterval(pull, 3000);
    return () => clearInterval(t);
  }, []);

  // Live WebSocket feed with auto-reconnect.
  useEffect(() => {
    let ws, retry, alive = true;
    const connect = () => {
      ws = new WebSocket(`${WS_BASE}/ws/stream`);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (alive) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        const e = JSON.parse(msg.data);
        e._id = idRef.current++;
        setEvents((prev) => [e, ...prev].slice(0, MAX_FEED));
      };
    };
    connect();
    return () => { alive = false; clearTimeout(retry); if (ws) ws.close(); };
  }, []);

  const live = metrics?.live;
  const lat = live?.latency_ms;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◆</span> SmartBidder
          <span className="tag">Real-Time RTB ML Engine</span>
        </div>
        <div className="brand-sub">pCTR × pConv → expected value → shaded bid</div>
      </header>

      {error && <div className="banner error">
        Cannot reach backend. Set <code>REACT_APP_API_URL</code> or start the API. ({error})
      </div>}

      <section className="kpis">
        <Kpi label="Auctions scored" value={(live?.total_bids ?? 0).toLocaleString()} />
        <Kpi label="Win rate" value={`${((live?.win_rate ?? 0) * 100).toFixed(1)}%`} />
        <Kpi label="Avg surplus" value={`$${(live?.avg_surplus ?? 0).toFixed(3)}`} />
        <Kpi label="Latency p50" value={`${(lat?.p50 ?? 0).toFixed(2)} ms`}
             sub={`p99 ${(lat?.p99 ?? 0).toFixed(2)} ms`} />
        <Kpi label="Model" value={metrics?.model_version ? `v${metrics.model_version}` : "—"}
             sub={metrics?.n_samples ? `${metrics.n_samples.toLocaleString()} samples` : ""} />
      </section>

      <div className="grid">
        <div className="col">
          <LiveFeed events={events} connected={connected} />
          <BidForm options={options} />
        </div>
        <div className="col">
          <ModelPanel metrics={metrics} />
        </div>
      </div>

      <footer className="foot">
        Synthetic RTB engine · XGBoost + isotonic calibration · FastAPI · sub-ms inference
      </footer>
    </div>
  );
}
