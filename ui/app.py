"""
Streamlit UI — Research Intelligence Assistant
Hybrid RAG + GraphRAG for Scientific Literature.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Research Intelligence Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS & Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    border: 1px solid #334155;
    padding: 2.2rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.main-header h1 {
    color: #f8fafc;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}

.main-header p {
    color: #cbd5e1;
    margin: 0.6rem 0 0 0;
    font-size: 1.05rem;
}

.answer-box {
    background: #090d16;
    border: 1px solid #1e293b;
    border-left: 5px solid #6366f1;
    border-radius: 10px;
    padding: 1.6rem;
    color: #e2e8f0;
    line-height: 1.75;
    margin: 1.2rem 0;
    font-size: 1.02rem;
}

.source-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
}

.source-card .paper-title {
    color: #38bdf8;
    font-weight: 600;
}

.source-card .score-badge {
    background: #2563eb;
    color: #ffffff;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    float: right;
}

.source-card .page-info {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 2px;
}

.source-card .snippet {
    color: #cbd5e1;
    margin-top: 0.6rem;
    line-height: 1.55;
}

.graph-pill {
    display: inline-block;
    background: #1e1b4b;
    border: 1px solid #4338ca;
    color: #c7d2fe;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.82rem;
    margin: 4px;
}

.graph-pill strong {
    color: #a5b4fc;
}

.stat-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    margin-bottom: 0.6rem;
}

.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}

.stat-card .stat-label {
    color: #94a3b8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.mode-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
}

.mode-hybrid { background: #312e81; color: #a5b4fc; }
.mode-vector { background: #064e3b; color: #6ee7b7; }
.mode-graph { background: #701a75; color: #f0abfc; }

div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=5)
def get_graph_stats():
    try:
        r = requests.get(f"{API_URL}/graph/stats", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=10)
def get_papers():
    try:
        r = requests.get(f"{API_URL}/papers", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def ask_question(question: str, top_k: int, mode: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k, "mode": mode},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Make sure `uvicorn src.api.main:app --reload` is running on port 8000.")
    except Exception as e:
        st.error(f"Error: {e}")
    return None


def explore_entity(entity: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/graph/explore", params={"entity": entity, "limit": 25}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Graph query error: {e}")
    return None


def find_path(source: str, target: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/graph/path", params={"source": source, "target": target}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Path query error: {e}")
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Search Mode")
    mode_options = {
        "auto": "⚡ Auto (Router)",
        "hybrid": "🔬 Hybrid GraphRAG",
        "vector": "📚 Vector RAG",
        "graph": "🕸️ Graph Only",
    }
    selected_mode = st.selectbox(
        "Retrieval Strategy",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=0,
    )

    top_k = st.slider("Passages top-k", min_value=1, max_value=15, value=5)

    st.markdown("---")
    stats = get_stats()
    g_stats = get_graph_stats()
    api_online = stats is not None and g_stats is not None

    if api_online:
        st.markdown(
            '<div style="background:#064e3b;color:#6ee7b7;padding:6px 12px;border-radius:6px;font-size:0.85rem;font-weight:600;margin-bottom:12px;">'
            '● API Backend Online (:8000)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#7f1d1d;color:#fca5a5;padding:8px 12px;border-radius:6px;font-size:0.82rem;font-weight:600;margin-bottom:12px;">'
            '▲ API Backend Offline<br><span style="font-size:0.75rem;font-weight:400;color:#fecaca;">Ejecuta en terminal:<br><code>.venv/bin/uvicorn src.api.main:app --reload</code></span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Knowledge Corpus")

    col1, col2 = st.columns(2)
    with col1:
        papers_display = stats["num_papers"] if (stats and "num_papers" in stats) else "—"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{papers_display}</div>
            <div class="stat-label">Papers (PG)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        chunks_display = stats["num_chunks"] if (stats and "num_chunks" in stats) else "—"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{chunks_display}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        nodes_display = g_stats.get("total_nodes", "—") if g_stats else "—"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{nodes_display}</div>
            <div class="stat-label">Nodes (Neo4j)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        rels_display = g_stats.get("total_relationships", "—") if g_stats else "—"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{rels_display}</div>
            <div class="stat-label">Relations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Ingested Papers")
    papers = get_papers()
    if papers:
        for p in papers:
            with st.expander(f"📝 {p['title'][:42]}", expanded=False):
                st.caption(f"**File:** `{p['filename']}`")
                if p['authors']:
                    st.caption(f"**Authors:** {', '.join(p['authors'][:3])}")
                st.caption(f"**Pages:** {p['num_pages']}")
    elif not api_online:
        st.warning("⚠️ No se puede conectar a la API (puerto 8000). Inicia el backend para listar los artículos.")
    else:
        st.info("No papers in database.\nRun: `python scripts/ingest.py`")


# ---------------------------------------------------------------------------
# Main Content Tabs
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🔬 Research Intelligence Assistant</h1>
    <p>Scientific literature assistant combining <strong>Vector RAG (pgvector)</strong> and <strong>Knowledge Graph (Neo4j)</strong></p>
</div>
""", unsafe_allow_html=True)

tab_qa, tab_graph = st.tabs(["💬 Literature Q&A (GraphRAG)", "🕸️ Knowledge Graph Explorer"])

# ---------------------------------------------------------------------------
# Tab 1: Literature Q&A
# ---------------------------------------------------------------------------
with tab_qa:
    # Example queries
    st.markdown("##### 💡 Suggested scientific queries")
    examples = [
        "What is Khovanskii non-degeneracy and how does it relate to Newton polyhedra?",
        "Which authors and papers connect non-degeneracy with topological triviality?",
        "Explain the formula and invariance of the Milnor number.",
        "What methods are used to prove topological triviality of isolated singularities?",
    ]

    cols = st.columns(2)
    example_clicked = None
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                example_clicked = ex

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    if example_clicked:
        st.session_state.current_question = example_clicked

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    question = st.text_area(
        "Question",
        value=st.session_state.current_question,
        placeholder="Ask questions about theorems, authors, concepts, proofs...",
        height=90,
        label_visibility="collapsed",
    )

    btn_col, clear_col = st.columns([4, 1])
    with btn_col:
        ask_btn = st.button("🚀 Ask Research Assistant", use_container_width=True, type="primary")
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.query_history = []
            st.session_state.current_question = ""
            st.rerun()

    query_to_execute = None
    if example_clicked:
        query_to_execute = example_clicked
    elif ask_btn and question.strip():
        query_to_execute = question.strip()

    if query_to_execute:
        with st.spinner("Traversing knowledge graph, searching vectors, and generating synthesis..."):
            result = ask_question(query_to_execute, top_k=top_k, mode=selected_mode)

        if result:
            st.session_state.query_history.insert(0, {
                "question": query_to_execute,
                "result": result,
            })

    # Results display
    if st.session_state.query_history:
        for entry in st.session_state.query_history:
            st.markdown("---")
            res = entry["result"]
            mode_badge_class = f"mode-{res.get('mode_used', 'hybrid')}"
            st.markdown(
                f'<span class="mode-badge {mode_badge_class}">{res.get("mode_used", "hybrid").upper()} MODE</span> '
                f'<strong>Q: {entry["question"]}</strong>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="answer-box">{res["answer"]}</div>',
                unsafe_allow_html=True,
            )

            # Metadata tags
            model_tag = f"`{res['model']}`"
            st.caption(
                f"Model: {model_tag} · Query Type: `{res.get('query_type', 'hybrid')}` · "
                f"Text Sources: {len(res.get('sources', []))} · Graph Relations: {len(res.get('graph_relations', []))}"
            )

            # Knowledge Graph Relationships
            graph_rels = res.get("graph_relations", [])
            if graph_rels:
                with st.expander(f"🕸️ Knowledge Graph Relations ({len(graph_rels)})", expanded=True):
                    pills_html = "<div>"
                    for rel in graph_rels:
                        pills_html += (
                            f'<span class="graph-pill">'
                            f'({rel.get("source_type")}) <strong>{rel.get("source")}</strong> '
                            f'➔ [{rel.get("relation")}] ➔ '
                            f'({rel.get("target_type")}) <strong>{rel.get("target")}</strong>'
                            f'</span>'
                        )
                    pills_html += "</div>"
                    st.markdown(pills_html, unsafe_allow_html=True)

            # Text Sources
            sources = res.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved Passages ({len(sources)})", expanded=False):
                    for i, src in enumerate(sources, 1):
                        score_pct = int(src["score"] * 100)
                        st.markdown(f"""
<div class="source-card">
    <span class="score-badge">Cosine: {score_pct}%</span>
    <div class="paper-title">[{i}] {src['paper_title']}</div>
    <div class="page-info">📄 {src['paper_filename']} · Page {src['page_number']}</div>
    <div class="snippet">{src['content']}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem; color: #64748b;">
            <div style="font-size:2.5rem">🔬</div>
            <p>Ask a research question or select one of the suggested queries above.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2: Knowledge Graph Explorer
# ---------------------------------------------------------------------------
with tab_graph:
    st.markdown("### 🕸️ Explore Relationships in Neo4j")
    st.caption("Inspect entities (Theorems, Concepts, Methods, Authors) and discover multi-hop connections.")

    sub_col1, sub_col2 = st.columns([1, 1])

    with sub_col1:
        st.markdown("#### 🔍 1-Hop Entity Neighborhood")
        explore_input = st.text_input(
            "Entity to explore",
            value="Newton Polyhedron",
            placeholder="e.g. Khovanskii non-degeneracy, Milnor fibration, Mutsuo Oka",
        )
        if st.button("Explore Entity", key="btn_explore"):
            with st.spinner("Querying Neo4j..."):
                exp_res = explore_entity(explore_input.strip())
            if exp_res and exp_res.get("relations"):
                st.success(f"Found {len(exp_res['relations'])} connected relations for '{explore_input}'")
                for r in exp_res["relations"]:
                    st.markdown(
                        f"- `({r['source_type']})` **{r['source']}** —`[:{r['relation']}]`→ `({r['target_type']})` **{r['target']}**"
                    )
            else:
                st.info(f"No relations found for '{explore_input}'. Ensure the graph has been built.")

    with sub_col2:
        st.markdown("#### 🔗 Multi-hop Path Discovery")
        source_input = st.text_input("Source Entity", value="Mutsuo Oka")
        target_input = st.text_input("Target Entity", value="topological triviality")

        if st.button("Find Connection Path", key="btn_path"):
            with st.spinner("Finding shortest paths in Neo4j..."):
                path_res = find_path(source_input.strip(), target_input.strip())
            if path_res and path_res.get("paths"):
                st.success(f"Path found between '{source_input}' and '{target_input}':")
                for p in path_res["paths"]:
                    nodes = p.get("nodes", [])
                    rels = p.get("relationships", [])
                    chain = []
                    for i in range(len(rels)):
                        chain.append(f"**{nodes[i]}** ➔ `[{rels[i]}]`")
                    chain.append(f"**{nodes[-1]}**")
                    st.markdown(" ➔ ".join(chain))
            else:
                st.info(f"No path found between '{source_input}' and '{target_input}' within 3 hops.")
