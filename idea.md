Aquí tienes una versión **compacta, directa y estructurada** de la propuesta:

---

# 🚀 Research Intelligence Assistant (RAG + GraphRAG)

> **Tesis:** En lugar de varios proyectos pequeños, construir **un único proyecto de nivel producción** que combine búsqueda semántica, extracción de entidades y grafos de conocimiento, apalancando tu doctorado en matemáticas.

---

### 1. El Núcleo del Proyecto
Un sistema de inteligencia sobre literatura científica que responde preguntas complejas conectando **evidencia textual (RAG)** con **redes de conceptos, autores y teoremas (GraphRAG)**.

* **Ventaja diferencial:** Usar tu propio corpus de investigación (~100–300 papers sobre *singularity theory, Newton polyhedra, topological triviality*). Demuestra aplicación real y dominio experto, no un tutorial con datos de juguete.

---

### 2. Los 3 Pilares Tecnológicos

| Pilar | Rol Principal | Caso de Uso |
| :--- | :--- | :--- |
| **LLM (Ollama)** | Razonamiento, extracción y síntesis. | Interpreta queries, extrae entidades/relaciones y genera respuestas con citas exactas. |
| **Vector RAG** | Recuperación densa por similitud semántica. | *“¿Qué paper define Khovanskii non-degeneracy y en qué párrafo?”* |
| **GraphRAG** | Relaciones estructuradas y multi-hop. | *“¿Qué autores conectan el Teorema X con la propiedad Y a través de citas compartidas?”* |

---

### 3. Stack Tecnológico Minimalista

* **Backend & Lógica:** Python + FastAPI.
* **Base Vectorial:** PostgreSQL + `pgvector` *(aprovecha tu experiencia en SQL)*.
* **Knowledge Graph:** Neo4j *(papers, autores, teoremas, conceptos)*.
* **Modelos:** Ollama *(LLM local)* + `sentence-transformers` *(embeddings)*.
* **Orquestación & UI:** Docker *(reproducibilidad)* + Streamlit *(demo funcional)*.

---

### 4. Roadmap de Desarrollo (Iterativo)

1. **Fase 1 · RAG Base:** Pipeline de ingesta (PDFs → Chunks → Embeddings → `pgvector` → LLM con citas).
2. **Fase 2 · Evaluación:** Métricas de retrieval (Precision@k, Recall) y fidelidad de respuestas (evitar alucinaciones).
3. **Fase 3 · Knowledge Graph:** Pipeline de extracción de entidades/relaciones y modelado en Neo4j.
4. **Fase 4 · Hybrid GraphRAG:** Fusión de búsqueda vectorial + grafo para responder consultas complejas.

---

### 5. Cómo Venderlo en el CV
* **Título:** *Scientific Literature Intelligence System — Hybrid RAG & GraphRAG*.
* **Keywords de impacto:** Knowledge Graphs, Vector Search, Information Extraction, Retrieval Evaluation, pgvector, Neo4j, Docker, LLM Fine-tuning/Serving.
* **Narrativa:** *“Diseño e implementación de un sistema end-to-end de retrieval híbrido para literatura científica de alta complejidad técnica.”*