# 📋 Plan de Trabajo — Research Intelligence Assistant

> **Objetivo:** Construir un sistema de inteligencia sobre literatura científica que combine RAG vectorial + GraphRAG + LLM para responder preguntas complejas con evidencia y relaciones entre conceptos, autores y trabajos.

---

## Estructura del Repositorio (objetivo final)

```
Research_Intelligence_Assistant/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
├── PLAN.md
├── idea.md
├── requirements.txt
│
├── data/
│   └── papers/                  # PDFs del corpus
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Settings centralizados (Pydantic)
│   │
│   ├── ingestion/               # Fase 1: Pipeline de ingesta
│   │   ├── __init__.py
│   │   ├── pdf_loader.py        # Extracción de texto desde PDFs
│   │   ├── chunker.py           # Estrategia de chunking
│   │   └── embedder.py          # Generación de embeddings
│   │
│   ├── db/                      # Capa de persistencia
│   │   ├── __init__.py
│   │   ├── postgres.py          # Conexión PG + pgvector
│   │   └── neo4j_client.py      # Conexión Neo4j
│   │
│   ├── retrieval/               # Fase 1–4: Retrieval
│   │   ├── __init__.py
│   │   ├── vector_search.py     # Búsqueda semántica en pgvector
│   │   ├── graph_search.py      # Consultas Cypher en Neo4j
│   │   └── hybrid.py            # Fusión de resultados (Fase 4)
│   │
│   ├── graph/                   # Fase 3: Knowledge Graph
│   │   ├── __init__.py
│   │   ├── extractor.py         # Extracción de entidades/relaciones con LLM
│   │   └── builder.py           # Construcción del grafo en Neo4j
│   │
│   ├── llm/                     # Interfaz con LLM
│   │   ├── __init__.py
│   │   ├── base.py              # Interfaz abstracta LLMClient
│   │   ├── ollama_client.py     # Implementación Ollama (local)
│   │   ├── gemini_client.py     # Implementación Gemini API
│   │   ├── factory.py           # Factory: crea el client según config
│   │   └── prompts.py           # Templates de prompts
│   │
│   ├── evaluation/              # Fase 2: Métricas
│   │   ├── __init__.py
│   │   └── metrics.py           # Precision@k, Recall, faithfulness
│   │
│   └── api/                     # FastAPI endpoints
│       ├── __init__.py
│       ├── main.py
│       └── routes.py
│
├── ui/
│   └── app.py                   # Streamlit frontend
│
├── scripts/
│   ├── ingest.py                # CLI para correr la ingesta
│   ├── build_graph.py           # CLI para construir el grafo
│   └── evaluate.py              # CLI para correr evaluación
│
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_graph.py
```

---

## Fase 0 · Setup del Entorno (~1–2 días)

### Tareas

- [ ] Inicializar repositorio Git + `.gitignore` (Python, .env, data/).
- [ ] Crear `docker-compose.yml` con servicios:
  - **PostgreSQL 16** con extensión `pgvector`.
  - **Neo4j** (imagen oficial).
- [ ] Crear `.env.example` con las variables necesarias (DB hosts, puertos, credenciales, proveedor LLM, API keys).
- [ ] Crear `requirements.txt` con dependencias iniciales:
  ```
  fastapi
  uvicorn
  psycopg[binary]
  pgvector
  neo4j
  ollama
  google-genai           # Gemini API
  sentence-transformers
  pymupdf                # extracción de PDFs
  pydantic-settings
  streamlit
  ```
- [ ] Instalar y verificar **Ollama** con un modelo base (e.g. `llama3.1:8b` o `mistral`).
- [ ] Obtener API key de **Gemini** y verificar conectividad básica.
- [ ] Crear `src/config.py` con `pydantic-settings` para gestionar configuración.
  - Variable `LLM_PROVIDER`: `"ollama"` | `"gemini"` (selecciona el backend).
  - Variables condicionales: `OLLAMA_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`.
- [ ] Verificar conectividad: script mínimo que conecte a PG, Neo4j y al LLM seleccionado.
- [ ] Recopilar corpus inicial: ~20–50 PDFs de singularity theory como dataset de arranque.

### Entregable
`docker-compose up` levanta PG+Neo4j, al menos un proveedor LLM (Ollama o Gemini) responde, y hay un corpus mínimo en `data/papers/`.

---

## Fase 1 · RAG Base (~1–2 semanas)

### 1.1 Ingesta de PDFs

- [ ] `pdf_loader.py` — Extraer texto de PDFs usando PyMuPDF.
  - Manejar PDFs con fórmulas/tablas (extraer texto legible, descartar imágenes por ahora).
  - Extraer metadatos básicos: título, autores (si están accesibles desde el PDF).
- [ ] `chunker.py` — Partir el texto en chunks.
  - Estrategia: chunks de ~500–800 tokens con overlap de ~100 tokens.
  - Cada chunk lleva metadata: `paper_id`, `chunk_index`, `page_number`.
- [ ] `embedder.py` — Generar embeddings con `sentence-transformers`.
  - Modelo recomendado: `all-MiniLM-L6-v2` (rápido) o `BAAI/bge-base-en-v1.5` (mejor calidad).
- [ ] Script `scripts/ingest.py`:
  ```bash
  python scripts/ingest.py --papers-dir data/papers/
  ```

### 1.2 Base de Datos Vectorial

- [ ] `postgres.py` — Esquema en PostgreSQL:
  ```sql
  CREATE TABLE papers (
      id SERIAL PRIMARY KEY,
      title TEXT,
      authors TEXT[],
      filename TEXT UNIQUE,
      abstract TEXT,
      ingested_at TIMESTAMPTZ DEFAULT now()
  );

  CREATE TABLE chunks (
      id SERIAL PRIMARY KEY,
      paper_id INT REFERENCES papers(id),
      chunk_index INT,
      page_number INT,
      content TEXT,
      embedding vector(384),  -- dimensión según modelo
      UNIQUE(paper_id, chunk_index)
  );

  CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
  ```

### 1.3 Retrieval + LLM

- [ ] `vector_search.py` — Búsqueda de los top-k chunks más similares a una query.
- [ ] `llm/base.py` — Interfaz abstracta `LLMClient` con método `chat(messages) -> str`.
- [ ] `llm/ollama_client.py` — Implementación para Ollama (local).
- [ ] `llm/gemini_client.py` — Implementación para Gemini API (`google-genai`).
- [ ] `llm/factory.py` — Factory que instancia el client según `LLM_PROVIDER` en config.
- [ ] `prompts.py` — Template de prompt RAG:
  ```
  Responde la siguiente pregunta basándote ÚNICAMENTE en los fragmentos proporcionados.
  Cita el paper y página de donde proviene cada afirmación.
  Si no encuentras información suficiente, dilo explícitamente.

  Fragmentos:
  {chunks}

  Pregunta: {query}
  ```
- [ ] Endpoint FastAPI `POST /query` que reciba una pregunta y devuelva respuesta + fuentes.

### 1.4 Interfaz Mínima

- [ ] `ui/app.py` — Streamlit con:
  - Input de texto para la pregunta.
  - Área de respuesta con las fuentes/citas expandibles.
  - Sidebar con estadísticas del corpus (# papers, # chunks).

### Entregable
El usuario puede subir PDFs, hacer preguntas, y recibir respuestas con citas exactas (paper + página). Demo funcional end-to-end.

### Criterios de aceptación
- [ ] Ingesta de ≥20 papers sin errores.
- [ ] Query devuelve respuesta coherente con ≥1 cita verificable.
- [ ] Latencia de query < 10s (en local).

---

## Fase 2 · Evaluación del RAG (~1 semana)

### 2.1 Dataset de Evaluación

- [ ] Crear manualmente ~20–30 pares (pregunta, respuesta esperada, paper fuente).
  - Ejemplo: `("¿Qué es Khovanskii non-degeneracy?", "Es una condición sobre...", "paper_X.pdf")`.
- [ ] Almacenar en un JSON/CSV en `data/eval/`.

### 2.2 Métricas de Retrieval

- [ ] `metrics.py` — Implementar:
  - **Precision@k**: de los k chunks recuperados, ¿cuántos son relevantes?
  - **Recall@k**: de los chunks relevantes, ¿cuántos fueron recuperados?
  - **MRR** (Mean Reciprocal Rank): ¿en qué posición aparece el primer chunk relevante?

### 2.3 Métricas de Respuesta

- [ ] **Faithfulness** (fidelidad): ¿La respuesta se basa en los chunks recuperados o alucina?
  - Implementar con LLM-as-judge: pedirle al LLM que verifique si cada afirmación tiene soporte en los chunks.
- [ ] **Relevancia**: ¿La respuesta contesta la pregunta?

### 2.4 Experimentación

- [ ] Comparar configuraciones:
  - Tamaño de chunk (300 vs 500 vs 800 tokens).
  - Modelo de embeddings (`MiniLM` vs `bge-base`).
  - Top-k (3 vs 5 vs 10).
  - Modelo LLM (si hay opciones en Ollama).
- [ ] Script `scripts/evaluate.py` que corra todas las métricas y genere un reporte.

### Entregable
Reporte con métricas cuantitativas que justifican las decisiones de diseño. Configuración óptima documentada.

### Criterios de aceptación
- [ ] Precision@5 ≥ 0.6 en el dataset de evaluación.
- [ ] Faithfulness ≥ 0.8 (las respuestas no inventan información).
- [ ] Reporte reproducible con `scripts/evaluate.py`.

---

## Fase 3 · Knowledge Graph (~1–2 semanas)

### 3.1 Modelo del Grafo

Nodos y relaciones en Neo4j:

```
(:Paper {title, year, doi, filename})
(:Author {name, affiliation?})
(:Concept {name, description?})       — e.g. "Milnor fibration", "Newton polyhedron"
(:Theorem {name, statement?})          — e.g. "Oka's theorem"
(:Method {name})                       — e.g. "resolution of singularities"

(:Author)-[:WROTE]->(:Paper)
(:Paper)-[:CITES]->(:Paper)
(:Paper)-[:DISCUSSES]->(:Concept)
(:Paper)-[:PROVES]->(:Theorem)
(:Paper)-[:USES_METHOD]->(:Method)
(:Concept)-[:RELATED_TO]->(:Concept)
(:Theorem)-[:GENERALIZES]->(:Theorem)
```

### 3.2 Extracción de Entidades/Relaciones

- [ ] `extractor.py` — Pipeline de extracción usando el LLM:
  - Input: texto del paper (o chunks representativos).
  - Prompt estructurado que pida extraer entidades y relaciones en formato JSON.
  - Ejemplo de output esperado:
    ```json
    {
      "entities": [
        {"type": "Concept", "name": "Khovanskii non-degeneracy"},
        {"type": "Theorem", "name": "Oka's theorem"}
      ],
      "relations": [
        {"source": "paper_X", "type": "DISCUSSES", "target": "Khovanskii non-degeneracy"},
        {"source": "paper_X", "type": "PROVES", "target": "Oka's theorem"}
      ]
    }
    ```
- [ ] Validación y deduplicación: normalizar nombres de entidades (e.g. "Milnor fiber" = "Milnor fibre").

### 3.3 Construcción del Grafo

- [ ] `builder.py` — Crear nodos y relaciones en Neo4j.
- [ ] `neo4j_client.py` — Funciones helper para queries Cypher comunes.
- [ ] Script `scripts/build_graph.py`:
  ```bash
  python scripts/build_graph.py --papers-dir data/papers/
  ```

### 3.4 Consultas sobre el Grafo

- [ ] `graph_search.py` — Queries Cypher parametrizadas:
  - Papers de un autor.
  - Conceptos discutidos en un paper.
  - Caminos entre dos conceptos (multi-hop).
  - Autores que comparten temas.
- [ ] Endpoint FastAPI `GET /graph/explore?entity=...`.

### Entregable
Grafo de conocimiento navegable con ≥100 nodos y ≥200 relaciones. Consultas Cypher funcionales.

### Criterios de aceptación
- [ ] Grafo construido automáticamente desde ≥20 papers.
- [ ] ≥5 tipos de consultas Cypher funcionando.
- [ ] Visualización del grafo en Neo4j Browser o Streamlit.

---

## Fase 4 · Hybrid GraphRAG (~1–2 semanas)

### 4.1 Retrieval Híbrido

- [ ] `hybrid.py` — Combinar resultados de:
  1. **Vector search**: chunks semánticamente similares (pgvector).
  2. **Graph search**: entidades/relaciones relevantes (Neo4j).
  - Estrategia de fusión: enriquecer el contexto del LLM con chunks + subgrafo relevante.

### 4.2 Router de Queries

- [ ] Clasificar la pregunta del usuario para decidir la estrategia:
  - **Factual** → RAG vectorial (e.g. "¿Qué dice el paper X sobre Y?").
  - **Relacional** → GraphRAG (e.g. "¿Qué autores conectan X con Y?").
  - **Híbrida** → ambos (e.g. "Explica la relación entre X e Y con evidencia").
- [ ] Implementar con LLM o con reglas heurísticas simples.

### 4.3 Prompt Enriquecido

- [ ] Template de prompt que incluya:
  ```
  Contexto textual (fragmentos relevantes):
  {chunks}

  Contexto estructural (relaciones del grafo):
  {graph_context}

  Pregunta: {query}
  Responde con evidencia. Cita papers específicos.
  ```

### 4.4 UI Final

- [ ] Streamlit mejorado:
  - Modo de búsqueda: Semántica / Grafo / Híbrida.
  - Visualización del subgrafo relevante a la consulta.
  - Panel de fuentes con links a los papers.
  - Historial de consultas.

### Entregable
Sistema completo con retrieval híbrido. El usuario puede hacer preguntas complejas y recibir respuestas que combinan evidencia textual con relaciones del grafo.

### Criterios de aceptación
- [ ] Queries relacionales (multi-hop) devuelven resultados que un RAG puro no puede.
- [ ] La UI muestra claramente de dónde viene cada pieza de información.
- [ ] El sistema funciona end-to-end con `docker-compose up`.

---

## Backlog / Mejoras Futuras (post-MVP)

- [ ] Autenticación básica en la API.
- [ ] Ingesta incremental (detectar papers nuevos/actualizados).
- [ ] Soporte para LaTeX además de PDF.
- [ ] Fine-tuning de embeddings con datos del dominio.
- [ ] Cache de queries frecuentes.
- [ ] CI/CD con GitHub Actions (lint, tests, build de Docker).
- [ ] Despliegue en cloud (e.g. Railway, Fly.io).
- [ ] Exportar el grafo como dataset abierto.

---

## Resumen de Tiempos Estimados

| Fase | Duración Estimada |
| :--- | :--- |
| Fase 0 · Setup | 1–2 días |
| Fase 1 · RAG Base | 1–2 semanas |
| Fase 2 · Evaluación | ~1 semana |
| Fase 3 · Knowledge Graph | 1–2 semanas |
| Fase 4 · Hybrid GraphRAG | 1–2 semanas |
| **Total** | **~5–8 semanas** |

> **Nota:** Los tiempos asumen dedicación parcial (~3–4h/día). Con dedicación completa se puede comprimir significativamente.
