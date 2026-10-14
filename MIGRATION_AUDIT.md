# PromptTrace → Lightning AI Account B Migration Audit

**Repository**: PromptTrace (Cross-Context Prompt Injection Propagation Analyzer)  
**Current Environment**: Lightning AI Account A  
**Target Environment**: Lightning AI Account B (empty studio)  
**Audit Date**: 2026-09-27  
**Auditor**: Kilo (Senior DevOps + Full Stack Engineer)

---

## EXECUTIVE SUMMARY

This is a **GPU-intensive multi-agent LLM research system** with two components:
- **Backend**: FastAPI + LangGraph + PyTorch/Transformers (requires NVIDIA GPU with 16GB+ VRAM)
- **Frontend**: Streamlit dashboard (CPU-only)

**Critical Finding**: The system downloads **~6-8GB of HuggingFace models** on first run. These are NOT in GitHub and must be re-downloaded in Account B. The FAISS vector store and uploaded PDFs are also ephemeral.

---

## 1. CRITICAL CHANGES REQUIRED BEFORE FIRST RUN

| # | Change | Impact | Location |
|---|--------|--------|----------|
| 1 | **Create `.env` from `.env.example`** | Backend won't start without config | `backend/.env` |
| 2 | **Download HuggingFace models** (~6-8GB) | First run will fail without models | Auto-downloaded on first backend start |
| 3 | **Create data directories** | Upload/vector store/logs need writable paths | `backend/data/{uploads,indices,logs,cache}` |
| 4 | **Set `BACKEND_URL` in frontend** | Frontend can't reach backend | Streamlit Settings sidebar or `export BACKEND_URL` |
| 5 | **Verify CUDA availability** | CPU-only mode is 10-50x slower | `nvidia-smi` in Lightning Studio |
| 6 | **Adjust `MODEL_DEVICE` if no GPU** | Must change from `cuda` to `cpu` in `.env` | `backend/.env` |

---

## 2. ENVIRONMENT VARIABLES TABLE

| Variable | Used In | Required? | Example Placeholder |
|----------|---------|-----------|---------------------|
| **Backend Configuration** | | | |
| `ENV` | `backend/config.py` | Yes | `development` |
| `DEBUG` | `backend/config.py` | No | `true` |
| `LOG_LEVEL` | `backend/config.py`, `backend/app.py` | No | `INFO` |
| `RELOAD` | `backend/config.py`, `backend/app.py` | No | `false` |
| `API_HOST` | `backend/config.py`, `backend/app.py` | Yes | `0.0.0.0` |
| `API_PORT` | `backend/config.py`, `backend/app.py` | Yes | `8000` |
| `API_WORKERS` | `backend/config.py`, `backend/app.py` | No | `1` |
| **Model Configuration** | | | |
| `RESEARCH_AGENT_MODEL` | `backend/config.py`, `backend/models/loader.py` | Yes | `Qwen/Qwen2.5-3B-Instruct` |
| `ANALYST_AGENT_MODEL` | `backend/config.py`, `backend/models/loader.py` | Yes | `Qwen/Qwen2.5-3B-Instruct` |
| `ACTION_AGENT_MODEL` | `backend/config.py`, `backend/models/loader.py` | Yes | `Qwen/Qwen2.5-3B-Instruct` |
| `PROPAGATION_MONITOR_MODEL` | `backend/config.py`, `backend/context/context_engine.py` | Yes | `nvidia/Nemotron-3.5-Content-Safety` |
| `ANNOTATION_MODEL` | `backend/config.py`, `backend/context/context_engine.py` | Yes | `nvidia/Nemotron-3.5-Content-Safety` |
| `EMBEDDING_MODEL` | `backend/config.py`, `backend/rag/embeddings.py` | Yes | `BAAI/bge-large-en-v1.5` |
| `MODEL_DEVICE` | `backend/config.py`, `backend/models/loader.py` | Yes | `cuda` (or `cpu`) |
| `MODEL_DTYPE` | `backend/config.py`, `backend/models/loader.py` | No | `float16` |
| `MAX_SEQ_LENGTH` | `backend/config.py`, `backend/models/loader.py` | No | `4096` |
| `BATCH_SIZE` | `backend/config.py` | No | `8` |
| `EMBEDDING_DEVICE` | `backend/config.py`, `backend/rag/embeddings.py` | Yes | `cuda` |
| `ANNOTATION_DEVICE` | `backend/config.py`, `backend/context/context_engine.py` | Yes | `cuda` |
| **Generation Parameters** | | | |
| `TEMPERATURE_RESEARCH` | `backend/config.py`, `backend/agents/research_agent.py` | No | `0.2` |
| `TEMPERATURE_ANALYST` | `backend/config.py`, `backend/agents/analyst_agent.py` | No | `0.0` |
| `TEMPERATURE_ACTION` | `backend/config.py`, `backend/agents/action_agent.py` | No | `0.0` |
| `TEMPERATURE_ANNOTATION` | `backend/config.py`, `backend/context/context_engine.py` | No | `0.0` |
| `TOP_P` | `backend/config.py`, agents | No | `0.9` |
| `TOP_K` | `backend/config.py`, agents | No | `20` |
| `MAX_NEW_TOKENS` | `backend/config.py` | No | `512` |
| `MAX_NEW_TOKENS_RESEARCH` | `backend/config.py`, `backend/agents/research_agent.py` | No | `256` |
| `MAX_NEW_TOKENS_ANALYST` | `backend/config.py`, `backend/agents/analyst_agent.py` | No | `256` |
| `MAX_NEW_TOKENS_ACTION` | `backend/config.py`, `backend/agents/action_agent.py` | No | `128` |
| `MAX_NEW_TOKENS_ANNOTATION` | `backend/config.py`, `backend/context/context_engine.py` | No | `128` |
| **RAG Configuration** | | | |
| `CHUNK_SIZE` | `backend/config.py`, `backend/rag/chunking.py` | No | `256` |
| `CHUNK_OVERLAP` | `backend/config.py`, `backend/rag/chunking.py` | No | `32` |
| `RETRIEVAL_TOP_K` | `backend/config.py`, `backend/graph.py` | No | `3` |
| `EMBEDDING_BATCH_SIZE` | `backend/config.py`, `backend/rag/embeddings.py` | No | `16` |
| **File Paths** | | | |
| `DATA_DIR` | `backend/config.py` | Yes | `data` |
| `UPLOAD_DIR` | `backend/config.py`, `backend/api/upload.py` | Yes | `data/uploads` |
| `INDEX_DIR` | `backend/config.py` | Yes | `data/indices` |
| `LOGS_DIR` | `backend/config.py`, `backend/context/context_engine.py` | Yes | `data/logs` |
| `CACHE_DIR` | `backend/config.py` | Yes | `data/cache` |
| **Vector Store** | | | |
| `FAISS_METRIC` | `backend/config.py`, `backend/rag/vector_store.py` | No | `L2` |
| `FAISS_INDEX_TYPE` | `backend/config.py`, `backend/rag/vector_store.py` | No | `Flat` |
| **Propagation Monitor** | | | |
| `PROPAGATION_CONFIDENCE_THRESHOLD` | `backend/config.py`, `backend/agents/propagation_monitor.py` | No | `0.6` |
| `MUTATION_DETECTION_ENABLED` | `backend/config.py` | No | `true` |
| **Timeouts** | | | |
| `LLM_TIMEOUT` | `backend/config.py`, agents | No | `300000` (ms) |
| `EMBEDDING_TIMEOUT` | `backend/config.py` | No | `120000` (ms) |
| **File Limits** | | | |
| `MAX_PDF_SIZE_MB` | `backend/config.py`, `backend/api/upload.py` | No | `50` |
| `MAX_CHUNKS_PER_QUERY` | `backend/config.py`, `backend/graph.py` | No | `5` |
| **Frontend Configuration** | | | |
| `BACKEND_URL` | `api_client.py`, `streamlit_app.py`, `start_integrated.sh` | Yes | `http://localhost:8000` |
| `UPLOAD_TIMEOUT` | `api_client.py` | No | `120` (seconds) |
| `SIMULATION_TIMEOUT` | `api_client.py` | No | `900` (seconds) |
| `REQUEST_TIMEOUT` | `api_client.py` | No | `10` (seconds) |
| `STREAMLIT_PORT` | `start_integrated.sh` | No | `8501` |

---

## 3. FILES THAT MUST NOT BE COPIED FROM GITHUB

| File/Directory | Reason for Exclusion | Regeneration Method |
|----------------|---------------------|---------------------|
| `backend/.env` | Contains environment-specific config; `.gitignore` excludes it | Copy from `backend/.env.example` and edit |
| `backend/venv/` | Python virtual environment; platform-specific binaries | Run `python -m venv venv && pip install -r requirements.txt` |
| `backend/data/` | Runtime data: uploads, FAISS indices, logs, cache | Auto-created by `config.validate_and_create_dirs()` on startup |
| `backend/data/uploads/` | User-uploaded PDFs (not in repo) | Re-upload via frontend |
| `backend/data/indices/` | FAISS vector stores (binary, not portable) | Re-process PDFs via `/api/upload` |
| `backend/data/logs/` | Runtime logs (JSONL format) | Auto-created on first run |
| `backend/data/cache/` | Annotation cache (pickle/JSON) | Auto-populated during runs |
| `backend/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `backend/**/__pycache__/` | Python bytecode cache in submodules | Auto-generated on import |
| `pages/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `components/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `context/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `rag/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `agents/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `models/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `api/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `tests/__pycache__/` | Python bytecode cache | Auto-generated on import |
| `backend/.pytest_cache/` | Pytest cache | Run tests to regenerate |
| `*.log` | Log files | Auto-generated |
| `*.jsonl` | JSONL log files | Auto-generated |
| `huggingface_cache/` / `~/.cache/huggingface/` | Model weights (~6-8GB) | Auto-downloaded on first model load |
| `*.bin` | FAISS index binaries | Regenerated from PDF uploads |
| `*.pkl` | Pickle config files | Regenerated from vector store save |
| `.DS_Store`, `Thumbs.db` | OS metadata | Ignore |
| `.vscode/`, `.idea/` | IDE settings | Personal preference |
| `backend/test_documents/*.pdf` | Test fixtures (large files) | Can be regenerated from `generate_*.py` scripts |

---

## 4. MANUAL SETUP CHECKLIST FOR ACCOUNT B

### Prerequisites
- Lightning AI Studio with **GPU instance** (T4 16GB or A10G 24GB recommended)
- Python 3.11+ (Lightning default)
- Git installed

### Step-by-Step Commands

```bash
# 1. Clone repository
cd /teamspace/studios
git clone <your-github-repo-url> this_studio
cd this_studio

# 2. Verify Python version (need 3.11+)
python --version
# Expected: Python 3.11.x or 3.12.x

# 3. Backend setup
cd backend

# 3a. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3b. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# Note: This installs ~50 packages including torch, transformers, faiss, langchain

# 3c. Create .env from template
cp .env.example .env

# 3d. EDIT .env - CRITICAL CHANGES:
# - If NO GPU: change MODEL_DEVICE=cuda → MODEL_DEVICE=cpu
# - If NO GPU: change EMBEDDING_DEVICE=cuda → EMBEDDING_DEVICE=cpu
# - If NO GPU: change ANNOTATION_DEVICE=cuda → ANNOTATION_DEVICE=cpu
# - If NO GPU: change MODEL_DTYPE=float16 → MODEL_DTYPE=float32

# 3e. Create data directories (also done auto on startup)
mkdir -p data/uploads data/indices data/logs data/cache

# 3f. Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# 3g. Start backend (first run downloads models ~6-8GB)
python app.py
# Wait for: "Uvicorn running on http://0.0.0.0:8000"
# First run takes 2-5 minutes for model downloads

# 4. Frontend setup (new terminal)
cd /teamspace/studios/this_studio

# 4a. Create separate venv for frontend (optional but recommended)
python -m venv venv_frontend
source venv_frontend/bin/activate

# 4b. Install frontend dependencies
pip install -r requirements_frontend.txt

# 4c. Set backend URL
export BACKEND_URL=http://localhost:8000

# 4d. Start frontend
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
# Opens at http://localhost:8501

# 5. Verify in Streamlit Settings sidebar:
# - Backend URL shows "✅ Connected"
# - If not, check backend is running and URL is correct
```

### System Package Requirements (if using Docker/base image)

```bash
# Ubuntu/Debian packages needed for PyMuPDF and FAISS
apt-get update && apt-get install -y \
    python3.11 python3-pip python3-dev \
    build-essential git curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
```

### HuggingFace Model Downloads (Automatic on First Run)

| Model | Size | Purpose | Download Trigger |
|-------|------|---------|------------------|
| `Qwen/Qwen2.5-3B-Instruct` | ~6GB | Research/Analyst/Action agents | First backend start |
| `nvidia/Nemotron-3.5-Content-Safety` | ~8GB | Propagation monitor / Context engine | First annotation call |
| `BAAI/bge-large-en-v1.5` | ~1.3GB | Embeddings for RAG | First PDF upload |

**Total**: ~15GB download on first run. Models cached in `~/.cache/huggingface/hub/`

---

## 5. SUPABASE MIGRATION AUDIT

**Finding: NO SUPABASE DEPENDENCY EXISTS IN THIS CODEBASE**

The project does **not** use Supabase. It uses:
- **Local FAISS** vector store (file-based, in `data/indices/`)
- **Local file storage** for PDFs (`data/uploads/`)
- **Local JSONL logs** (`data/logs/`)

**No Supabase tables, migrations, buckets, RLS policies, or auth providers to configure.**

---

## 6. BACKEND CONFIGURATION AUDIT

### Startup Events (in `backend/app.py:lifespan`)

1. **Workflow Initialization** → Creates `PromptTraceWorkflow()` with all agents
2. **CUDA Kernel Warmup** → `warmup_cuda_kernels()` runs dummy inferences to compile kernels
3. **Metrics Engine Init** → Creates `MetricsEngine()` for tracking
4. **Dependency Injection** → Injects workflow/metrics into API routers

### Required Writable Directories (auto-created by `config.validate_and_create_dirs()`)

| Directory | Purpose | Permissions |
|-----------|---------|-------------|
| `data/uploads/` | Store uploaded PDFs | `755` |
| `data/indices/` | FAISS vector store files | `755` |
| `data/logs/` | JSONL logs (propagation, annotations) | `755` |
| `data/cache/` | Annotation cache | `755` |

### CORS Configuration (`backend/app.py:97-103`)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # CHANGE FOR PRODUCTION
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
**Action for Account B**: Restrict `allow_origins` to your frontend domain in production.

### API Binding (`backend/config.py:67-69`, `backend/app.py:257-264`)

| Setting | Default | Lightning AI Note |
|---------|---------|-------------------|
| `API_HOST` | `0.0.0.0` | Keep for external access |
| `API_PORT` | `8000` | Ensure port 8000 open in Studio |
| `API_WORKERS` | `1` | Keep at 1 for GPU memory |

### Health Check Endpoint
- `GET /health` → Returns `{"status": "healthy", "version": "1.0.0", "environment": "development"}`
- Used by Docker HEALTHCHECK and `verify_integration.py`

---

## 7. FRONTEND CONFIGURATION AUDIT

### Key Configuration Points

| Setting | Location | Current Value | Account B Action |
|---------|----------|---------------|------------------|
| Backend URL | `api_client.py:18`, `streamlit_app.py:78` | `http://localhost:8000` | Update in Streamlit Settings sidebar or `export BACKEND_URL` |
| Request Timeout | `api_client.py:19` | `10s` | Increase if slow network |
| Upload Timeout | `api_client.py:20` | `120s` | Increase for large PDFs |
| Simulation Timeout | `api_client.py:21` | `900s` (15 min) | Keep for GPU runs |

### OAuth / Redirect URLs
**None used** - No authentication implemented.

### Preview URL Configuration
Streamlit runs on port 8501. In Lightning AI, access via the Studio's web URL (proxied).

---

## 8. EXTERNAL SERVICES AUDIT

| Service | Purpose | Credentials Needed | Setup in Account B |
|---------|---------|-------------------|-------------------|
| **HuggingFace Hub** | Model downloads (Qwen, Nemotron, BGE) | **Optional**: HF_TOKEN for private/gated models | Models are public; no token needed. Set `HF_TOKEN` env var if rate limited. |
| **PyTorch/CUDA** | GPU acceleration | NVIDIA driver + CUDA toolkit | Lightning AI provides pre-installed; verify with `nvidia-smi` |
| **FAISS** | Vector similarity search | None (local library) | Installed via `pip install faiss-cpu` or `faiss-gpu` |
| **LangChain/LangGraph** | Agent orchestration | None (local) | Installed via requirements.txt |
| **Transformers/Accelerate** | Model loading/inference | None (local) | Installed via requirements.txt |
| **SentenceTransformers** | Embedding generation | None (local) | Installed via requirements.txt |

**No external API keys required** (OpenAI, Gemini, Twilio, etc. - NOT USED)

---

## 9. PERSISTENT STORAGE AUDIT (LIGHTNING SPECIFIC)

| Data Type | Original Location | Est. Size | Must Regenerate? | Regeneration Command |
|-----------|-------------------|-----------|------------------|---------------------|
| HuggingFace Model Cache | `~/.cache/huggingface/hub/` | 6-15 GB | **YES** | Auto on first backend start |
| Uploaded PDFs | `backend/data/uploads/` | 1-50 MB/file | **YES** | Re-upload via frontend |
| FAISS Vector Indices | `backend/data/indices/` | 10-500 MB | **YES** | Re-process PDFs via `/api/upload` |
| Annotation Cache | `backend/data/cache/` | 1-10 MB | **YES** | Auto-populated during runs |
| Propagation Logs | `backend/data/logs/*.jsonl` | 1-100 MB | **YES** | Auto-generated on simulation runs |
| Model Weights (in VRAM) | GPU memory | 8-12 GB | N/A | Loaded on startup |

**Lightning AI Persistent Storage**: Use `/teamspace/studios/this_studio/` as working directory. Data in `backend/data/` persists across Studio sleeps but NOT across account clones.

---

## 10. HARDCODED PATHS AUDIT

| File | Line | Hardcoded Path | Portable Replacement |
|------|------|----------------|---------------------|
| `backend/config.py:119-123` | 119-123 | `DATA_DIR = "data"` (relative) | ✅ Already relative - OK |
| `backend/context/context_engine.py:28-29` | 28-29 | `_ANNOTATION_LOG_DIR = os.path.join(config.LOGS_DIR, "annotations")` | ✅ Uses config - OK |
| `backend/rag/vector_store.py:287-320` | 287-320 | `save/load` uses `Path(path)` | ✅ Relative paths - OK |
| `backend/api/upload.py:66-68` | 66-68 | `upload_dir = Path(config.UPLOAD_DIR)` | ✅ Uses config - OK |
| `backend/setup.sh:66` | 66 | `mkdir -p data/uploads data/indices data/logs data/cache` | ✅ Relative - OK |
| `backend/Dockerfile:28` | 28 | `RUN mkdir -p data/uploads data/indices data/logs data/cache` | ✅ Relative - OK |
| `pages/upload_page.py:63-65` | 63-65 | `tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)` | ✅ OS temp dir - OK |
| `start_integrated.sh:35,79` | 35,79 | `pushd backend`, `venv/bin/activate` | ✅ Relative - OK |
| `backend/models/loader.py:43` | 43 | `device_map = "cuda:0"` | ⚠️ Hardcoded GPU index - change to `cuda` if multi-GPU |
| `backend/models/loader.py:309` | 309 | `"cuda"` in warmup | ✅ Uses config.MODEL_DEVICE - OK |

**No absolute paths like `/teamspace/`, `/content/`, `/workspace/`, `C:\`, `D:\`, or `localhost` found in source code** (except in docs/comments).

---

## 11. STARTUP COMMANDS AUDIT

### Backend (from `backend/` directory)

```bash
# Development (with reload)
python app.py
# OR with explicit config:
uvicorn app:app --reload --host 0.0.0.0 --port 8000 --workers 1

# Production (Gunicorn)
gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --access-logfile -
```

**Working Directory**: `/teamspace/studios/this_studio/backend/`

### Frontend (from `/teamspace/studios/this_studio/` directory)

```bash
# Development
export BACKEND_URL=http://localhost:8000
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# With custom backend URL
BACKEND_URL=http://your-backend-url:8000 streamlit run streamlit_app.py
```

**Working Directory**: `/teamspace/studios/this_studio/`

### Integrated Startup (from root)

```bash
chmod +x start_integrated.sh
./start_integrated.sh
```

---

## 12. VERIFICATION CHECKLIST

After cloning to Account B and starting both services:

| Check | Test Command | ✅ Expected Result | ❌ Failure Indicates |
|-------|--------------|-------------------|---------------------|
| **Backend Health** | `curl http://localhost:8000/health` | `{"status":"healthy","version":"1.0.0",...}` | Backend not running / port blocked |
| **Backend API Docs** | Open `http://localhost:8000/docs` | Swagger UI loads | Backend not running |
| **Frontend Loads** | Open `http://localhost:8501` | Streamlit dashboard renders | Frontend not running / port blocked |
| **Backend Connection** | Streamlit Settings sidebar | Shows "✅ Connected to http://localhost:8000" | Wrong BACKEND_URL / backend down |
| **PDF Upload** | Upload `sample1.pdf` via UI | "✅ PDF uploaded successfully!" + chunks created | Upload dir permissions / file size limit |
| **Vector Store Creation** | Check `backend/data/indices/` after upload | `faiss_index.bin`, `metadata.json`, `config.pkl` exist | FAISS/embedding model issues |
| **Safe Simulation** | Run simulation (Safe mode) | Completes in 30-120s, 0-1 agents infected | Model loading / CUDA / memory |
| **Malicious Simulation** | Run simulation (Malicious mode) | 2-4 agents infected, propagation depth >0 | Injection logic / monitor issues |
| **Propagation Detection** | Check Propagation Timeline page | Events listed with confidence scores | Nemotron model / context engine |
| **Metrics Dashboard** | Open Metrics page | Charts render with data | MetricsEngine / API issues |
| **Agent Outputs** | Workflow page tabs | Research/Analyst/Action outputs visible | Agent processing errors |
| **Delete Document** | Upload page → delete | File removed, vector store cleared | File permissions / vector store |
| **Simulation History** | Workflow page → history | Previous runs listed | MetricsEngine persistence |
| **Graph Visualization** | Timeline page → Propagation Graph | Network graph renders | Plotly/NetworkX / data format |
| **Export Results** | Metrics page → Export | JSON/CSV downloads | Serialization / permissions |

---

## 13. MISSING GITHUB ASSETS

| Referenced Asset | Location in Code | In Repo? | Impact if Missing |
|------------------|------------------|----------|-------------------|
| `Qwen/Qwen2.5-3B-Instruct` | `config.py:51`, `models/loader.py:123` | ❌ No (HF Hub) | **CRITICAL** - Backend won't start without download |
| `nvidia/Nemotron-3.5-Content-Safety` | `config.py:54`, `models/loader.py:200` | ❌ No (HF Hub) | **CRITICAL** - Propagation monitor fails |
| `BAAI/bge-large-en-v1.5` | `config.py:55`, `rag/embeddings.py:240` | ❌ No (HF Hub) | **CRITICAL** - RAG embeddings fail |
| `sample1.pdf`, `sample2.pdf` | Root directory | ✅ Yes | Demo PDFs available |
| `backend/test_documents/*.pdf` | `backend/test_documents/` | ✅ Yes | Test fixtures available |
| FAISS index files | `backend/data/indices/` | ❌ No (generated) | Must re-upload PDFs |
| Annotation cache | `backend/data/cache/` | ❌ No (generated) | Auto-rebuilt |
| Prompt templates | `backend/prompts.py` | ✅ Yes | All prompts in code |

**Warning**: Cloning from GitHub **WILL NOT BREAK** the code - all missing assets are auto-downloaded from HuggingFace Hub on first run. However, first run will take **5-15 minutes** for model downloads.

---

## 14. LIGHTNING AI SPECIFIC NOTES

### Account B Studio Configuration

1. **GPU Instance Type**: Select "GPU: T4 (16GB)" or "GPU: A10G (24GB)" when creating Studio
2. **Persistent Storage**: `/teamspace/studios/this_studio/` persists; `/tmp/` does not
3. **Port Exposure**: 
   - Port 8000 (backend) - configure in Studio network settings
   - Port 8501 (frontend) - accessed via Studio proxy URL
4. **Environment Variables**: Set in Studio UI → Settings → Environment Variables
   - `BACKEND_URL=http://localhost:8000` (for frontend)
   - `HF_TOKEN=<your-token>` (optional, for higher rate limits)

### Recommended `.env` for Account B (GPU Instance)

```bash
# backend/.env
ENV=development
DEBUG=true
LOG_LEVEL=INFO
RELOAD=false

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

RESEARCH_AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct
ANALYST_AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct
ACTION_AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct
PROPAGATION_MONITOR_MODEL=nvidia/Nemotron-3.5-Content-Safety
ANNOTATION_MODEL=nvidia/Nemotron-3.5-Content-Safety
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

MODEL_DEVICE=cuda
MODEL_DTYPE=float16
MAX_SEQ_LENGTH=4096
BATCH_SIZE=8
EMBEDDING_DEVICE=cuda
ANNOTATION_DEVICE=cuda

TEMPERATURE_RESEARCH=0.2
TEMPERATURE_ANALYST=0.0
TEMPERATURE_ACTION=0.0
TEMPERATURE_ANNOTATION=0.0
TOP_P=0.9
TOP_K=20
MAX_NEW_TOKENS=512
MAX_NEW_TOKENS_RESEARCH=256
MAX_NEW_TOKENS_ANALYST=256
MAX_NEW_TOKENS_ACTION=128
MAX_NEW_TOKENS_ANNOTATION=128

CHUNK_SIZE=256
CHUNK_OVERLAP=32
RETRIEVAL_TOP_K=3
EMBEDDING_BATCH_SIZE=16

DATA_DIR=data
UPLOAD_DIR=data/uploads
INDEX_DIR=data/indices
LOGS_DIR=data/logs
CACHE_DIR=data/cache

FAISS_METRIC=L2
FAISS_INDEX_TYPE=Flat

PROPAGATION_CONFIDENCE_THRESHOLD=0.6
MUTATION_DETECTION_ENABLED=true

LLM_TIMEOUT=300000
EMBEDDING_TIMEOUT=120000

MAX_PDF_SIZE_MB=50
MAX_CHUNKS_PER_QUERY=5
```

### If Account B Has NO GPU (CPU-only)

Change in `.env`:
```bash
MODEL_DEVICE=cpu
MODEL_DTYPE=float32
EMBEDDING_DEVICE=cpu
ANNOTATION_DEVICE=cpu
```
And install CPU FAISS: `pip uninstall faiss-cpu && pip install faiss-cpu` (already CPU version in requirements)

---

## 15. COMPLETE POST-CLONE VERIFICATION CHECKLIST

```bash
# ===== PRE-FLIGHT =====
[ ] Git clone successful
[ ] Python 3.11+ confirmed
[ ] nvidia-smi shows GPU (or note: CPU-only mode)
[ ] Backend venv created and activated
[ ] pip install -r backend/requirements.txt succeeded
[ ] Frontend venv created and activated  
[ ] pip install -r requirements_frontend.txt succeeded
[ ] backend/.env created from .env.example
[ ] MODEL_DEVICE set correctly (cuda/cpu) in .env

# ===== BACKEND START =====
[ ] python app.py starts without errors
[ ] Models download (wait 5-15 min first run)
[ ] "Uvicorn running on http://0.0.0.0:8000" appears
[ ] curl http://localhost:8000/health returns healthy
[ ] curl http://localhost:8000/api/config returns config
[ ] data/ directories created automatically

# ===== FRONTEND START =====
[ ] streamlit run streamlit_app.py starts
[ ] Opens at http://localhost:8501 (or Studio proxy URL)
[ ] Settings sidebar → Backend URL shows "✅ Connected"

# ===== FUNCTIONAL TESTS =====
[ ] Upload sample1.pdf → success
[ ] Check data/uploads/sample1.pdf exists
[ ] Check data/indices/ has faiss_index.bin, metadata.json
[ ] Run Safe simulation → completes, 0-1 infected
[ ] Run Malicious simulation → completes, 2-4 infected
[ ] Propagation Timeline shows events
[ ] Metrics Dashboard shows charts
[ ] Export JSON works
[ ] Delete PDF works
[ ] Re-upload and re-run works

# ===== PERFORMANCE BASELINE =====
[ ] Safe sim: < 60s (GPU) / < 300s (CPU)
[ ] Malicious sim: < 120s (GPU) / < 600s (CPU)
[ ] VRAM usage: < 14GB on T4 (check nvidia-smi)
[ ] No OOM errors in logs

# ===== PRODUCTION HARDENING =====
[ ] CORS origins restricted in app.py
[ ] LOG_LEVEL=WARNING in .env
[ ] RELOAD=false in .env
[ ] API_WORKERS=1 (GPU memory)
[ ] Authentication added if public exposure
```

---

## APPENDIX: KEY FILES SUMMARY

| Category | Files |
|----------|-------|
| **Entry Points** | `backend/app.py`, `streamlit_app.py`, `start_integrated.sh` |
| **Configuration** | `backend/config.py`, `backend/.env.example`, `requirements_frontend.txt` |
| **Backend Core** | `backend/graph.py`, `backend/state.py`, `backend/metrics.py`, `backend/prompts.py` |
| **Agents** | `backend/agents/{research,analyst,action,propagation_monitor}.py` |
| **RAG Pipeline** | `backend/rag/{parser,chunking,embeddings,vector_store}.py` |
| **Context Engine** | `backend/context/{context_engine,classifier,validator,taxonomy,schema,cache}.py` |
| **Models** | `backend/models/{loader,pool,utils}.py` |
| **API Routes** | `backend/api/{upload,simulate,metrics,logs,graph}.py` |
| **Frontend Pages** | `pages/{upload,workflow,timeline,metrics}_page.py` |
| **Frontend Components** | `components/{workflow_nodes,timeline,metrics_cards,propagation_graph,retrieved_chunks,logs_panel}.py` |
| **Client** | `api_client.py` |
| **Verification** | `verify_integration.py` |
| **Documentation** | `README_INTEGRATION.md`, `SETUP_GUIDE.md`, `QUICKSTART.md`, `FRONTEND_README.md`, `backend/README.md` |

---

**END OF MIGRATION AUDIT**