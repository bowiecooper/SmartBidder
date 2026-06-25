// Interactive "place a bid" panel — submit a custom auction context and see the
// engine's full decision breakdown. The centerpiece for a live interview demo.
import React, { useState } from "react";
import { postJSON } from "../api";

const Select = ({ label, value, onChange, options }) => (
  <label className="field">
    <span>{label}</span>
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  </label>
);

export default function BidForm({ options }) {
  const [ctx, setCtx] = useState({
    audience_segment: "gaming_community",
    ad_category: "gaming",
    device_type: "desktop",
    ad_position: "top",
    ad_size: "300x250",
    user_gender: "M",
    user_age: 24,
    hour_of_day: 20,
    day_of_week: 5,
    base_cpm: 12.0,
    pacing: 1.0,
  });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  if (!options) return null;
  const set = (k) => (v) => setCtx((c) => ({ ...c, [k]: v }));

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      setResult(await postJSON("/bid", {
        ...ctx,
        user_age: Number(ctx.user_age),
        hour_of_day: Number(ctx.hour_of_day),
        day_of_week: Number(ctx.day_of_week),
        base_cpm: Number(ctx.base_cpm),
        pacing: Number(ctx.pacing),
      }));
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  return (
    <div className="card">
      <h2>Place a Bid</h2>
      <form onSubmit={submit} className="bid-form">
        <Select label="Audience segment" value={ctx.audience_segment}
                onChange={set("audience_segment")} options={options.audience_segments} />
        <Select label="Ad category" value={ctx.ad_category}
                onChange={set("ad_category")} options={options.ad_categories} />
        <Select label="Device" value={ctx.device_type}
                onChange={set("device_type")} options={options.device_types} />
        <Select label="Position" value={ctx.ad_position}
                onChange={set("ad_position")} options={options.ad_positions} />
        <Select label="Ad size" value={ctx.ad_size}
                onChange={set("ad_size")} options={options.ad_sizes} />
        <Select label="Gender" value={ctx.user_gender}
                onChange={set("user_gender")} options={["M", "F", "O"]} />
        <label className="field"><span>Age: {ctx.user_age}</span>
          <input type="range" min="18" max="65" value={ctx.user_age}
                 onChange={(e) => set("user_age")(e.target.value)} /></label>
        <label className="field"><span>Hour: {ctx.hour_of_day}:00</span>
          <input type="range" min="0" max="23" value={ctx.hour_of_day}
                 onChange={(e) => set("hour_of_day")(e.target.value)} /></label>
        <label className="field"><span>Budget pacing: {Number(ctx.pacing).toFixed(2)}</span>
          <input type="range" min="0" max="1" step="0.05" value={ctx.pacing}
                 onChange={(e) => set("pacing")(e.target.value)} /></label>
        <button type="submit" disabled={busy} className="primary">
          {busy ? "Scoring…" : "Compute optimal bid"}
        </button>
      </form>

      {err && <div className="error">{err}</div>}
      {result && (
        <div className="bid-result">
          <div className={`verdict ${result.should_bid ? "bid" : "nobid"}`}>
            {result.should_bid ? `BID $${result.bid.toFixed(3)}` : "NO BID"}
            <span className="lat">{result.latency_ms.toFixed(2)} ms</span>
          </div>
          <div className="breakdown">
            <Stat label="P(click)" value={`${(result.p_ctr * 100).toFixed(1)}%`} />
            <Stat label="P(conv | click)" value={`${(result.p_conversion * 100).toFixed(1)}%`} />
            <Stat label="Conv. value" value={`$${result.value_per_conversion.toFixed(0)}`} />
            <Stat label="Expected value" value={`$${result.expected_value.toFixed(3)}`} />
            <Stat label="Win probability" value={`${(result.win_probability * 100).toFixed(1)}%`} />
            <Stat label="Expected surplus" value={`$${result.expected_surplus.toFixed(3)}`} />
            {result.should_bid && (
              <Stat label="Shading (bid/EV)"
                    value={`${((result.bid / result.expected_value) * 100).toFixed(0)}%`} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const Stat = ({ label, value }) => (
  <div className="stat"><span className="stat-label">{label}</span>
    <span className="stat-value">{value}</span></div>
);
