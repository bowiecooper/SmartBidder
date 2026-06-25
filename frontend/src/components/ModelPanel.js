// Model evaluation panel: AUC/log-loss/Brier per model, calibration reliability
// diagrams, and the top feature importances.
import React from "react";
import { CalibrationChart, ImportanceBars } from "./Charts";

const MODELS = [
  { key: "ctr", label: "CTR  ·  P(click)" },
  { key: "cvr", label: "CVR  ·  P(conv | click)" },
  { key: "winrate", label: "Win-rate  ·  P(win | bid)" },
];

export default function ModelPanel({ metrics }) {
  if (!metrics?.model_metrics) return null;
  const m = metrics.model_metrics;
  return (
    <div className="card">
      <h2>Model Quality
        <span className="feed-sub">v{metrics.model_version} · {metrics.n_samples?.toLocaleString()} samples</span>
      </h2>

      <div className="model-metrics">
        {MODELS.map(({ key, label }) => (
          <div className="mm" key={key}>
            <div className="mm-label">{label}</div>
            <div className="mm-stats">
              <span>AUC <b>{m[key].auc.toFixed(3)}</b></span>
              <span>LogLoss <b>{m[key].log_loss.toFixed(3)}</b></span>
              <span>Brier <b>{m[key].brier.toFixed(3)}</b></span>
            </div>
          </div>
        ))}
      </div>

      <div className="charts-row">
        <CalibrationChart curve={m.ctr.calibration_curve} title="CTR calibration" />
        <CalibrationChart curve={m.winrate.calibration_curve} title="Win-rate calibration" />
      </div>

      <h3>Top feature importances (CTR model)</h3>
      <ImportanceBars importances={metrics.feature_importances} />
    </div>
  );
}
