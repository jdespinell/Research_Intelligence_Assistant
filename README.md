# 🔬 Research Intelligence Assistant

A scientific literature intelligence system combining **Vector RAG** (`pgvector`) and **GraphRAG** (`Neo4j`) to answer complex mathematical and research questions with exact paper citations and knowledge graph relationships.

---

## ✨ Features

- 📄 **Robust PDF Ingestion** — Text extraction with PyMuPDF, word-level sliding window chunking, NUL-byte sanitization, and incremental streaming.
- 🔍 **Dense Semantic Search** — High-performance vector embeddings using `BAAI/bge-base-en-v1.5` stored in PostgreSQL 16 with `pgvector` (IVFFlat index).
- 🕸️ **Neo4j Knowledge Graph** — Extracts semantic entities (`Concept`, `Theorem`, `Method`, `Author`) and ontological relations (`WROTE`, `DISCUSSES`, `PROVES`, `USES_METHOD`, `RELATED_TO`, `GENERALIZES`, `CITES`).
- 🤖 **Hybrid GraphRAG** — Intelligent query router that detects factual, relational, or hybrid questions and synthesizes both textual excerpts and graph paths into a unified prompt.
- 📊 **Evaluation Framework** — Automated retrieval metrics (`Precision@k`, `Recall@k`, `MRR`) and LLM-as-a-judge generation metrics (`Faithfulness`, `Answer Relevance`).
- 🔄 **Multi-Provider LLM** — Supports local inference with **Ollama** (`gemma4:12b`, `llama3.1`, etc.) and cloud inference with **Google Gemini API** via configuration.
- 💻 **Interactive UI** — Streamlit interface featuring a research Q&A mode with citation badges, graph relation pills, and an interactive Knowledge Graph Explorer (1-hop neighborhood & multi-hop path finding).

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone <repo-url>
cd Research_Intelligence_Assistant
cp .env.example .env
# Edit .env with your credentials and LLM_PROVIDER choice
```

### 2. Start Databases (Docker)

```bash
docker compose up -d postgres neo4j
```

> **Note:** PostgreSQL is exposed on port `5433` (to avoid conflicting with local Postgres instances), and Neo4j is on `7687` (Bolt) / `7474` (Browser).

### 3. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify Connectivity

```bash
python scripts/check_connectivity.py
```

---

## 📖 End-to-End Workflow

### Step 1: Ingest Papers (PDFs)
Place your research PDFs in `data/papers/` (or use the sample downloader):

```bash
# Optional: Download open-access arXiv sample papers
python scripts/download_sample_papers.py

# Ingest papers into PostgreSQL + pgvector (incremental & streamed)
python scripts/ingest.py --papers-dir data/papers/
```

### Step 2: Build the Knowledge Graph (Neo4j)
Extract theorems, concepts, authors, and methods using the configured LLM:

```bash
# Build graph from papers already stored in PostgreSQL
python scripts/build_graph.py --from-db

# Or build directly from a directory of PDFs:
# python scripts/build_graph.py --papers-dir data/papers/
```

### Step 3: Run Automated Evaluation
Run the evaluation benchmark on the curated dataset:

```bash
python scripts/evaluate.py
```
> Generates a detailed report in `data/eval/evaluation_report.md` and `data/eval/evaluation_report.json`.

### Step 4: Run the Application

In **Terminal 1** — Start the FastAPI backend:
```bash
uvicorn src.api.main:app --reload
```

In **Terminal 2** — Launch the Streamlit frontend:
```bash
streamlit run ui/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Testing

Run the full pytest suite (unit tests for ingestion, embeddings, pgvector, Neo4j live, hybrid router, and API endpoints):

```bash
pytest -v
```

---

## ⚙️ Configuration (.env)

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | Backend provider (`ollama` or `gemini`) | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `OLLAMA_MODEL` | Ollama model name | `gemma4:12b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `POSTGRES_PORT` | PostgreSQL host port | `5433` |
| `NEO4J_URI` | Neo4j Bolt connection URI | `bolt://localhost:7687` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-base-en-v1.5` |
| `CHUNK_SIZE` | Words per chunk | `600` |
| `CHUNK_OVERLAP` | Overlapping words between chunks | `100` |
| `RETRIEVAL_TOP_K` | Default top-k chunks retrieved | `5` |

---

## 📂 Project Structure

```
Research_Intelligence_Assistant/
├── data/
│   ├── papers/                  # Research PDFs
│   └── eval/                    # Benchmark dataset & evaluation reports
├── src/
│   ├── api/                     # FastAPI app, endpoints (/query, /stats, /papers, /graph/*)
│   ├── config.py                # Pydantic Settings
│   ├── db/                      # PostgreSQL + pgvector and Neo4j drivers
│   ├── evaluation/              # Precision@k, Recall@k, MRR, Faithfulness, Relevance
│   ├── graph/                   # LLM entity/relation extraction & Neo4j GraphBuilder
│   ├── ingestion/               # PyMuPDF loader, word chunker, sentence-transformer embedder
│   ├── llm/                     # OllamaClient, GeminiClient, Factory, Prompts
│   └── retrieval/               # Vector search, graph search, and hybrid retriever
├── ui/
│   └── app.py                   # Streamlit UI (Q&A + Knowledge Graph Explorer)
├── scripts/
│   ├── check_connectivity.py    # Health check for all services
│   ├── download_sample_papers.py# arXiv downloader utility
│   ├── ingest.py                # Ingestion pipeline CLI
│   ├── build_graph.py           # Knowledge graph builder CLI
│   └── evaluate.py              # RAG evaluation CLI
├── tests/                       # Pytest test suite (19 passing tests)
├── docker-compose.yml           # Postgres 16 (pgvector) + Neo4j 5.21
└── requirements.txt             # Project dependencies
```

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python >= 3.11 |
| **Backend API** | FastAPI + Uvicorn |
| **Vector Database** | PostgreSQL 16 + `pgvector` |
| **Knowledge Graph** | Neo4j 5.21 (Community) + Cypher |
| **Dense Embeddings** | `sentence-transformers` (`BAAI/bge-base-en-v1.5`, 768 dim) |
| **LLM Providers** | Google Gemini API (`google-genai`) / Ollama (`gemma4:12b`) |
| **PDF Extraction** | PyMuPDF (`pymupdf`) |
| **Frontend UI** | Streamlit |
| **Containers** | Docker & Docker Compose |
