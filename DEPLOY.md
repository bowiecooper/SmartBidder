# Deploying SmartBidder

The app is two pieces: a **FastAPI backend** (with WebSockets) and a **static React
frontend**. The split below uses free tiers and takes ~10 minutes.

## 1. Backend → Render

The repo includes [`render.yaml`](./render.yaml), a Render Blueprint.

1. Push this repo to GitHub.
2. In the [Render dashboard](https://dashboard.render.com): **New +** → **Blueprint**, select the repo.
3. Render reads `render.yaml`, builds `backend/Dockerfile`, and deploys `smartbidder-api`.
4. Note the service URL, e.g. `https://smartbidder-api.onrender.com`.

Render's free tier sleeps after ~15 min idle, so the first request cold-starts (~30 s).
WebSockets are supported on the free web service.

> CLI alternative: `render blueprint launch` after `render login`.

## 2. Frontend → Vercel

The React app reads the backend URL from the build-time env var `REACT_APP_API_URL`.

**Dashboard:**
1. [vercel.com](https://vercel.com) → **Add New** → **Project** → import the repo.
2. Set **Root Directory** to `frontend`.
3. Add an environment variable: `REACT_APP_API_URL = https://smartbidder-api.onrender.com`
   (your Render URL from step 1).
4. Deploy. Vercel auto-detects Create React App.

**CLI:**
```bash
cd frontend
vercel                       # link/create project (set root dir = current)
vercel env add REACT_APP_API_URL production    # paste your Render URL
vercel --prod
```

## 3. Verify

- Open the Vercel URL — the dashboard should connect (green "streaming" dot) and the
  live feed should fill in within a couple of seconds.
- If the feed stays disconnected, confirm `REACT_APP_API_URL` is set and the Render
  service is awake (`curl https://<your-api>/health`).

## Updating the screenshot

```bash
# with backend on :8000 and `serve -s frontend/build` on :3000
python .dev/shoot.py        # writes docs/dashboard.png
```
