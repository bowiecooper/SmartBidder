// Lightweight dependency-free charts (inline SVG / CSS), so the bundle stays small
// and we avoid React-19 peer-dependency friction with heavier chart libraries.
import React from "react";

// Reliability diagram: predicted probability (x) vs observed frequency (y).
// A well-calibrated model hugs the diagonal.
export function CalibrationChart({ curve, title }) {
  const W = 240, H = 240, pad = 28;
  if (!curve || !curve.mean_predicted?.length) return null;
  const xs = curve.mean_predicted;
  const ys = curve.fraction_positive;
  const max = Math.max(0.001, ...xs, ...ys) * 1.05;
  const sx = (v) => pad + (v / max) * (W - 2 * pad);
  const sy = (v) => H - pad - (v / max) * (H - 2 * pad);
  const pts = xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join(" ");

  return (
    <div className="chart">
      <div className="chart-title">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
        {/* perfect-calibration diagonal */}
        <line x1={sx(0)} y1={sy(0)} x2={sx(max)} y2={sy(max)}
              stroke="#3a4a63" strokeDasharray="4 4" />
        <polyline points={pts} fill="none" stroke="#4ade80" strokeWidth="2" />
        {xs.map((x, i) => (
          <circle key={i} cx={sx(x)} cy={sy(ys[i])} r="3" fill="#4ade80" />
        ))}
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#2a3650" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#2a3650" />
        <text x={W / 2} y={H - 4} className="axis" textAnchor="middle">predicted</text>
        <text x={10} y={H / 2} className="axis" textAnchor="middle"
              transform={`rotate(-90 10 ${H / 2})`}>observed</text>
      </svg>
    </div>
  );
}

// Horizontal bars for the top feature importances.
export function ImportanceBars({ importances }) {
  if (!importances) return null;
  const entries = Object.entries(importances).slice(0, 10);
  const max = Math.max(...entries.map(([, v]) => v), 0.0001);
  return (
    <div className="importances">
      {entries.map(([name, val]) => (
        <div className="imp-row" key={name}>
          <span className="imp-name" title={name}>{name}</span>
          <div className="imp-track">
            <div className="imp-fill" style={{ width: `${(val / max) * 100}%` }} />
          </div>
          <span className="imp-val">{val.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
