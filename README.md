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

- [x] **Phase 0** — Setup & infrastructure (Docker, PG 16 + pgvector, Neo4j, Ollama & Gemini)
- [x] **Phase 1** — Base RAG pipeline (PyMuPDF, word chunking, bge embeddings, pgvector, FastAPI)
- [x] **Phase 2** — Evaluation framework (Precision@k, Recall@k, MRR, Faithfulness, Relevance CLI)
- [x] **Phase 3** — Knowledge Graph (LLM extraction of Concepts/Theorems/Methods/Authors, Neo4j builder)
- [x] **Phase 4** — Hybrid GraphRAG (Intelligent query router, dual-context synthesis, Streamlit Explorer)

### Running Tests

```bash
pytest -v
```

### Running the End-to-End Workflow

```bash
# 1. Download sample arXiv papers (or copy your PDFs to data/papers/)
python scripts/download_sample_papers.py

# 2. Ingest papers into PostgreSQL + pgvector
python scripts/ingest.py --papers-dir data/papers/

# 3. Build the Neo4j Knowledge Graph with LLM extraction
python scripts/build_graph.py

# 4. Run automated evaluation & generate report
python scripts/evaluate.py

# 5. Start the FastAPI backend
uvicorn src.api.main:app --reload

# 6. Launch the Streamlit Research Assistant UI
streamlit run ui/app.py
```

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
