# Modal Streamlit Setup Guide

## Quick Start

```bash
# 1. Deploy Modal backend (if not already deployed)
cd D:\CODE\prompt-injection
modal deploy modal_phase3.py

# 2. Configure local environment
echo "BACKEND_URL=https://smallbrat--prompttrace-backend-fastapi-app.modal.run" > .env

# 3. Install frontend dependencies
pip install -r requirements_frontend.txt

# 4. Run Streamlit locally
streamlit run streamlit_app.py --server.port 8501

# 5. Open browser
# http://localhost:8501
```

## Configuration

The `.env` file controls the backend connection:

```bash
# .env
BACKEND_URL=https://smallbrat--prompttrace-backend-fastapi-app.modal.run
```

### Runtime Override
In the Streamlit sidebar → Settings → Backend URL, you can change the URL at runtime. The app will automatically reconnect.

## Expected Latencies

| Operation | Time |
|-----------|------|
| Cold start (first request) | ~6-7 minutes |
| PDF upload | ~17 seconds |
| Simulation (warm) | 90-110 seconds |
| Metrics/Logs/Graph | ~1-2 seconds |

## Workflow

1. **Upload PDF** page: Upload a PDF, optionally enable "Inject Malicious Content"
2. **Agent Workflow** page: Enter query, choose Safe/Malicious, click "Run Simulation"
3. **Wait** ~90-110 seconds for completion
4. **View results** in tabs: Retrieval, Research, Analysis, Action, Propagation
4. **Export** results as JSON or CSV
5. **Metrics Dashboard**: View aggregate metrics across all runs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Backend unavailable" | Check Modal app is deployed and running |
| CORS errors | Backend has `allow_origins=["*"]` - should work |
| Simulation timeout | Increase `SIMULATION_TIMEOUT` in `api_client.py` (default 900s) |
| Cold start too slow | Modal `scaledown_window=300` keeps container warm 5min |
| Upload fails | Check PDF is valid, <50MB |

## Backend Endpoint

```
https://smallbrat--prompttrace-backend-fastapi-app.modal.run
```

## Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/api/gpu` | GPU status & VRAM |
| `/api/upload` | PDF upload (POST) |
| `/api/simulate` | Run simulation (POST) |
| `/api/metrics` | Aggregate metrics |
| `/api/logs/status` | Log status |
| `/api/graph` | Propagation graph |
| `/api/workflow/status` | Component status |