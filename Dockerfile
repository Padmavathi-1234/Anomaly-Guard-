FROM python:3.11-slim

LABEL description="AnomalyGuard — Explainable AI RL Environment for Cybersecurity"

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify openenv installed correctly
RUN python -c "import openenv; print('openenv OK:', dir(openenv))"

COPY app/ ./app/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

RUN useradd -m -u 1000 anomalyguard && chown -R anomalyguard:anomalyguard /app
USER anomalyguard

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
