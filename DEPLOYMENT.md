# Deployment Guide

## Local API

```powershell
pip install -r requirements.txt
python DataIngestion.py
python scripts/run_api.py
```

Default API URL: `http://127.0.0.1:8000`.

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /papers`
- `POST /query`

Example:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"question":"list papers","top_k":5}'
```

## Docker Compose

```powershell
docker compose up --build
```

The container expects index artifacts under `outputs/index/faiss_index`. Run ingestion before starting the API or mount prebuilt artifacts.

## Secrets

Store `GROQ_API_KEY` in `.env` for local development. In production, inject secrets through the deployment platform rather than baking them into images.

