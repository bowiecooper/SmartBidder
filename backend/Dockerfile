# SmartBidder API image. Pre-trained model artifacts are committed under
# backend/smartbidder/models, so the container is self-contained and starts fast.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
WORKDIR /app/backend

# If models are missing (e.g. fresh checkout without artifacts), train them at build.
RUN test -f smartbidder/models/metadata.json || python -m smartbidder.train --samples 100000

EXPOSE 8000
CMD uvicorn smartbidder.api:app --host 0.0.0.0 --port ${PORT}
