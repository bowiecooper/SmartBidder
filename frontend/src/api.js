// Central place for the backend base URL. Set REACT_APP_API_URL at build time
// (e.g. on Vercel) to point at the deployed backend; falls back to localhost in dev.
export const API_BASE =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

export const WS_BASE = API_BASE.replace(/^http/, "ws");

export async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}
