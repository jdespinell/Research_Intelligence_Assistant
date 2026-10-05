# 🔬 Research Intelligence Assistant

A scientific literature intelligence system that combines **Vector RAG** and **GraphRAG** to answer complex questions about research papers — with exact citations and concept relationships.

## Features

- 📄 **PDF ingestion** — Extract, chunk, and embed scientific papers
- 🔍 **Semantic search** — Dense vector retrieval with `pgvector`
- 🕸️ **Knowledge graph** — Papers, authors, concepts, theorems in Neo4j
- 🤖 **Hybrid RAG** — Combine vector + graph search for multi-hop reasoning
- 📊 **Evaluation** — Retrieval metrics (Precision@k, MRR) and faithfulness scoring
- 🔄 **Multi-provider LLM** — Switch between Ollama (local) and Gemini API via config

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url>
cd Research_Intelligence_Assistant
cp .env.example .env
# Edit .env with your credentials and LLM_PROVIDER choice
```

### 2. Start databases

```bash
docker-compose up -d postgres neo4j
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify connectivity

```bash
python scripts/check_connectivity.py
```

### 5. Ingest papers

```bash
# Place PDFs in data/papers/
python scripts/ingest.py --papers-dir data/papers/
```

### 6. Ask questions

```bash
# Start the UI
streamlit run ui/app.py

# Or use the API
uvicorn src.api.main:app --reload
```

## LLM Provider Configuration

Set `LLM_PROVIDER` in your `.env`:

| Value | Description |
|-------|-------------|
| `ollama` | Local inference via [Ollama](https://ollama.ai). No API key needed. |
| `gemini` | Google Gemini API. Requires `GEMINI_API_KEY`. |

**Ollama setup:**
```bash
ollama pull llama3.1:8b
```

**Gemini setup:**
```bash
# Add to .env:
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

## Project Structure

```
src/
├── config.py          # Centralized settings (pydantic-settings)
├── ingestion/         # PDF → chunks → embeddings
├── db/                # PostgreSQL + Neo4j clients
├── retrieval/         # Vector search, graph search, hybrid
├── graph/             # Entity extraction + graph construction
├── llm/               # LLM abstraction (Ollama & Gemini)
├── evaluation/        # RAG evaluation metrics
└── api/               # FastAPI endpoints
ui/
└── app.py             # Streamlit interface
scripts/
├── check_connectivity.py
├── ingest.py
├── build_graph.py
└── evaluate.py
```

## Development Roadmap

See [PLAN.md](PLAN.md) for the full development plan.

- [x] **Phase 0** — Setup & infrastructure
- [ ] **Phase 1** — Basic RAG pipeline
- [ ] **Phase 2** — Evaluation framework
- [ ] **Phase 3** — Knowledge graph
- [ ] **Phase 4** — Hybrid GraphRAG

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| API | FastAPI |
| Vector DB | PostgreSQL 16 + pgvector |
| Graph DB | Neo4j 5 |
| Embeddings | sentence-transformers (BAAI/bge-base-en-v1.5) |
| LLM (local) | Ollama |
| LLM (cloud) | Google Gemini |
| UI | Streamlit |
| Infrastructure | Docker + Docker Compose |
