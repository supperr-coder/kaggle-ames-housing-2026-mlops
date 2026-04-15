# Kaggle Ames Housing 2026 — MLOps Project

An end-to-end MLOps project that trains a LightGBM model on the Ames Housing dataset and serves predictions via a live REST API.

## Live API

The API is deployed on a Hetzner cloud server and publicly accessible:

- **Health check:** http://188.245.167.150:8000
- **Interactive docs:** http://188.245.167.150:8000/docs
- **Predict endpoint:** `POST http://188.245.167.150:8000/predict`

## What was built

### 1. Model
- LightGBM regressor trained on the Ames Housing dataset
- Target: log-transformed sale price (reverse-transformed in the API response)
- Feature engineering: house age, remodel age, total SF, bathroom ratios, log lot area, cyclical month encoding
- Experiment tracking with MLflow

### 2. API (FastAPI)
- `GET /` — health check, confirms model is loaded
- `POST /predict` — accepts house features, returns `log_sale_price` and `estimated_sale_price`
- All fields optional with sensible defaults so partial payloads are accepted

### 3. Containerisation (Docker)
- `python:3.11-slim` base image
- Model is NOT baked into the image — mounted from the host server at runtime
- Image published to GitHub Container Registry (`ghcr.io`)

### 4. CI Pipeline (GitHub Actions)
- Runs on every push and pull request to `main`
- Installs dependencies, runs `pytest` against the FastAPI endpoints
- Model and file I/O are mocked so tests run without real model files

### 5. CD Pipeline (GitHub Actions)
- Triggers automatically when CI passes on `main`
- Builds Docker image and pushes to `ghcr.io`
- SSHes into Hetzner server, pulls latest image, restarts container
- Zero-downtime deploy: old container stopped, new one started with volume-mounted model

## Project structure

```
├── src/
│   ├── api.py            # FastAPI app
│   ├── train.py          # Model training
│   ├── preprocess.py     # Feature engineering
│   ├── evaluate.py       # Model evaluation
│   └── split_data.py     # Train/test split
├── tests/
│   └── test_api.py       # Pytest tests for API endpoints
├── .github/workflows/
│   ├── ci.yml            # CI pipeline
│   └── cd.yml            # CD pipeline
├── Dockerfile
├── requirements-api.txt  # Production dependencies
└── requirements.txt      # Full dev dependencies
```

## Tech stack

| Layer | Tool |
|---|---|
| Model | LightGBM |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Registry | GitHub Container Registry (ghcr.io) |
| CI/CD | GitHub Actions |
| Hosting | Hetzner Cloud (CX22) |
