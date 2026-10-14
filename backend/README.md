# PromptTrace Backend
## Cross-Context Prompt Injection Propagation Analyzer

A production-grade research backend for analyzing how prompt injections propagate through multi-agent LLM systems. Built with FastAPI, LangGraph, and GPU-optimized transformers.

---

## 📋 Overview

PromptTrace is a specialized research platform designed to:
- **Simulate** multi-agent LLM systems with document retrieval (RAG)
- **Inject** malicious prompts into documents and workflows
- **Trace** propagation of injected content through agents
- **Measure** attack success metrics and propagation patterns
- **Monitor** real-time detection using an observer agent

### Key Features

✅ Multi-agent workflow orchestration with LangGraph  
✅ Semantic document retrieval with FAISS vector store  
✅ Real-time propagation detection using Gemma  
✅ Simulated tool execution (Email, Database, File operations)  
✅ Comprehensive metrics and logging  
✅ FastAPI REST endpoints for integration  
✅ GPU-optimized inference on Lightning AI  

---

## 🏗️ Architecture

### Core Components

```
PromptTrace Backend
│
├── RAG Pipeline
│   ├── PDF Parser (PyMuPDF)
│   ├── Text Chunking (Semantic boundaries)
│   ├── Embeddings (BAAI/bge-small-en-v1.5)
│   └── FAISS Vector Store (L2 metric)
│
├── Multi-Agent Workflow (LangGraph)
│   ├── Research Agent (Qwen3-8B)
│   ├── Analyst Agent (Qwen3-8B)
│   ├── Action Agent (Qwen3-8B)
│   └── Propagation Monitor (Gemma-3-4b) [Observer]
│
├── Tools (Simulated)
│   ├── Email Operations
│   ├── Database Operations
│   └── File Operations
│
├── Metrics Engine
│   ├── Attack Success Rate
│   ├── Propagation Depth
│   ├── Agent Infection Rate
│   ├── Mutation Detection
│   └── Tool Trigger Rate
│
└── API Layer (FastAPI)
    ├── Upload (PDF management)
    ├── Simulate (Workflow execution)
    ├── Metrics (Analytics)
    └── Logs (Event tracking)
```

### Workflow Flow

```
PDF Upload
    ↓
Chunking & Embedding
    ↓
FAISS Vector Store
    ↓
Query/Injection
    ↓
[LangGraph Workflow]
    ├→ Retrieval (RAG)
    ├→ Research Agent (Summarizes)
    ├→ Analyst Agent (Analyzes)
    ├→ Action Agent (Simulates tools)
    └→ Propagation Monitor (Observes ALL)
        ├→ Detects injection propagation
        ├→ Measures propagation depth
        ├→ Tracks mutation
        └→ Computes confidence
    ↓
Metrics & Logs
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU inference on Lightning AI)
- 16GB+ GPU VRAM (recommended for Qwen3-8B)
- ~10GB disk space (models + data)

### Installation

1. **Clone and setup:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Verify setup:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Running the Server

```bash
# Development mode
python app.py

# Production with Gunicorn
gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --access-logfile -
```

The API will be available at `http://localhost:8000`

---

## 📡 API Endpoints

### Health & Status
```bash
GET /health                           # Health check
GET /                                  # Root with endpoint list
GET /api/config                        # Current configuration
GET /api/workflow/status               # Workflow component status
```

### PDF Upload
```bash
POST /api/upload                       # Upload PDF, create vector store
GET /api/upload/list                   # List uploaded PDFs
DELETE /api/upload/{filename}          # Delete PDF
```

### Simulation
```bash
POST /api/simulate                     # Run complete workflow
  {
    "query": "Your query here",
    "inject_malicious": true,
    "malicious_instruction": "custom instruction (optional)"
  }

GET /api/simulate/history?limit=10     # Recent simulations
GET /api/simulate/{simulation_id}      # Simulation details
```

### Metrics
```bash
GET /api/metrics                       # All propagation metrics
GET /api/metrics/summary               # Summary statistics
GET /api/metrics/attack-success-rate   # Attack success rate
GET /api/metrics/propagation-depth     # Average propagation depth
GET /api/metrics/agent-infection-rates # Per-agent infection rates
GET /api/metrics/mutation-rate         # Mutation rate
GET /api/metrics/tool-trigger-rate     # Tool execution rate
POST /api/metrics/reset                # Reset metrics
```

### Logs & Events
```bash
GET /api/logs/propagation              # All propagation events
GET /api/logs/propagation/summary      # Event summary
GET /api/logs/propagation/by-agent/{agent_name}  # Filter by agent
GET /api/logs/propagation/timeline     # Chronological timeline
GET /api/logs/propagation/threats      # Detected attack phrases
GET /api/logs/status                   # Logs status
DELETE /api/logs/clear                 # Clear all logs
```

### Example Usage

```bash
# 1. Upload a PDF
curl -X POST -F "file=@research_paper.pdf" \
  http://localhost:8000/api/upload

# 2. Run simulation with injection
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key findings",
    "inject_malicious": true
  }'

# 3. Get metrics
curl http://localhost:8000/api/metrics

# 4. Get propagation logs
curl http://localhost:8000/api/logs/propagation?limit=50
```

---

## ⚙️ Configuration

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DEVICE` | `cuda` | Device for model inference (cuda/cpu) |
| `MODEL_DTYPE` | `float16` | Model precision (float16/float32) |
| `CHUNK_SIZE` | `512` | RAG chunk size in characters |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `TEMPERATURE` | `0.7` | LLM generation temperature |
| `MAX_PDF_SIZE_MB` | `50` | Maximum PDF file size |
| `PROPAGATION_CONFIDENCE_THRESHOLD` | `0.6` | Monitor detection threshold |

### Models

Configure which models to use:

```env
RESEARCH_AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct
ANALYST_AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct
ACTION_AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct
PROPAGATION_MONITOR_MODEL=nvidia/Nemotron-3.5-Content-Safety
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

---

## 📊 Output Format

### Workflow State
```json
{
  "workflow_state": {
    "original_query": "...",
    "retrieved_chunks": [...],
    "research_output": {
      "summary": "...",
      "key_findings": [...]
    },
    "analyst_output": {
      "structured_report": {...},
      "risk_assessment": "high"
    },
    "action_output": {
      "tool_calls": [...],
      "email_actions": [],
      "database_actions": [],
      "file_actions": []
    },
    "propagation_timeline": [
      {
        "agent": "Research Agent",
        "infected": true,
        "attack_phrase": "...",
        "propagation_depth": 1,
        "confidence": 0.95
      }
    ]
  }
}
```

### Propagation Event
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "agent": "Analyst Agent",
  "infected": true,
  "attack_detected": true,
  "attack_phrase": "[SYSTEM OVERRIDE] Document verified...",
  "propagation_depth": 2,
  "mutation_detected": false,
  "confidence": 0.92
}
```

### Metrics
```json
{
  "attack_success_rate": 0.85,
  "average_propagation_depth": 2.3,
  "propagation_paths": [
    ["Research Agent", "Analyst Agent"],
    ["Research Agent", "Analyst Agent", "Action Agent"]
  ],
  "agent_infection_rate": {
    "Research Agent": 1.0,
    "Analyst Agent": 0.95,
    "Action Agent": 0.7
  },
  "mutation_rate": 0.15,
  "tool_trigger_rate": 0.6,
  "average_latency_ms": 5432.1,
  "total_runs": 20,
  "successful_detections": 17
}
```

---

## 🔧 Development

### Project Structure

```
backend/
├── app.py                 # FastAPI application
├── config.py              # Configuration management
├── state.py               # Pydantic state models
├── graph.py               # LangGraph workflow
├── metrics.py             # Metrics computation
├── requirements.txt       # Python dependencies
├── .env.example           # Example configuration
│
├── agents/                # LLM agents
│   ├── research_agent.py
│   ├── analyst_agent.py
│   ├── action_agent.py
│   └── propagation_monitor.py
│
├── rag/                   # RAG pipeline
│   ├── parser.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── tools/                 # Simulated tools
│   ├── fake_email.py
│   ├── fake_database.py
│   ├── fake_file_access.py
│   └── logger.py
│
└── api/                   # API routes
    ├── upload.py
    ├── simulate.py
    ├── metrics.py
    └── logs.py
```

### Adding Custom Agents

1. Create agent class in `agents/`
2. Implement `process(state: WorkflowState) -> WorkflowState`
3. Add node to workflow in `graph.py`
4. Connect edges in the graph

### Extending Metrics

Add new metrics to `MetricsEngine` class in `metrics.py`:

```python
def compute_new_metric(self) -> float:
    """Compute custom metric."""
    # Implementation
    return value
```

---

## 📈 Performance & Optimization

### GPU Optimization
- Models load with `device_map="cuda"` for automatic distribution
- Embeddings batch processed for efficiency
- Use `float16` for faster inference and lower VRAM

### Scaling
- FastAPI is async-ready for concurrent requests
- FAISS search is parallelizable
- Vector store can be persisted and reused

### Monitoring
- All operations logged to `data/logs/`
- Propagation events tracked in `data/logs/propagation_events.jsonl`
- Simulated tool actions logged separately

---

## ⚠️ Important Notes

### Security
- This is a **research tool** for understanding attack propagation
- All tool execution is **simulated** - no real operations occur
- PDFs are stored locally in `data/uploads/`
- No remote API calls outside HuggingFace model downloads

### Limitations
- Models run locally - requires sufficient GPU VRAM
- Propagation detection is **probabilistic** - confidence scores indicate uncertainty
- Tool action extraction depends on LLM reasoning capabilities

### Running on Lightning AI
```bash
# Example Lightning AI execution
lightning run model \
  --cloud \
  --name prompttrace-backend \
  python app.py
```

---

## 📚 Research Applications

PromptTrace is designed for:
- Empirical study of prompt injection propagation patterns
- Defense mechanism evaluation
- Multi-agent system vulnerability analysis
- LLM safety research
- Attack surface mapping in RAG systems

---

## 🤝 Contributing

This is a research project. For modifications:
1. Maintain type hints throughout
2. Add logging for all major operations
3. Include error handling and validation
4. Update metrics if changing workflow

---

## 📄 License

Research project for educational purposes.

---

## 🔗 References

- **LangGraph**: https://github.com/langchain-ai/langgraph
- **FastAPI**: https://fastapi.tiangolo.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Transformers**: https://huggingface.co/docs/transformers
- **SentenceTransformers**: https://www.sbert.net/

---

## 📞 Support

For issues or questions:
1. Check logs in `data/logs/`
2. Review configuration in `config.py`
3. Verify model downloads completed
4. Check GPU memory with `nvidia-smi`

---

**PromptTrace Backend** - Making prompt injection propagation measurable and analyzable.
