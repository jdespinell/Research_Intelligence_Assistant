"""
Streamlit UI — Research Intelligence Assistant
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}

.main-header h1 {
    color: #e2e8f0;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.main-header p {
    color: #94a3b8;
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
}

.answer-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1.5rem;
    color: #e2e8f0;
    line-height: 1.7;
    margin: 1rem 0;
}

.source-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.875rem;
}

.source-card .paper-title {
    color: #60a5fa;
    font-weight: 600;
}

.source-card .score-badge {
    background: #1d4ed8;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    float: right;
}

.source-card .page-info {
    color: #94a3b8;
    font-size: 0.8rem;
}

.source-card .snippet {
    color: #cbd5e1;
    margin-top: 0.5rem;
    line-height: 1.5;
}

.stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
}

.stat-card .stat-label {
    color: #94a3b8;
    font-size: 0.85rem;
}

.stTextArea textarea {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.2s;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_papers():
    try:
        r = requests.get(f"{API_URL}/papers", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def ask_question(question: str, top_k: int) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Make sure `uvicorn src.api.main:app --reload` is running.")
    except Exception as e:
        st.error(f"Error: {e}")
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=15, value=5)

    st.markdown("---")
    st.markdown("## 📊 Corpus Stats")
    stats = get_stats()
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['num_papers']}</div>
                <div class="stat-label">Papers</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['num_chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("API not reachable")

    st.markdown("---")
    st.markdown("## 📄 Papers")
    papers = get_papers()
    if papers:
        for p in papers:
            with st.expander(f"📝 {p['title'][:50]}", expanded=False):
                st.caption(f"**File:** {p['filename']}")
                if p['authors']:
                    st.caption(f"**Authors:** {', '.join(p['authors'][:3])}")
                st.caption(f"**Pages:** {p['num_pages']}")
    else:
        st.info("No papers ingested yet.\nRun: `python scripts/ingest.py`")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🔬 Research Intelligence Assistant</h1>
    <p>Ask complex questions about your scientific literature — powered by RAG + GraphRAG</p>
</div>
""", unsafe_allow_html=True)

# Example queries
st.markdown("#### 💡 Example queries")
examples = [
    "What is Khovanskii non-degeneracy and how is it related to Newton polyhedra?",
    "Which papers discuss the Milnor fibration theorem?",
    "What methods are used to prove topological triviality?",
    "Explain the connection between Newton non-degeneracy and the Milnor number.",
]

cols = st.columns(2)
example_clicked = None
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}"):
            example_clicked = ex

# Query input
if "query_history" not in st.session_state:
    st.session_state.query_history = []

question = st.text_area(
    "Your question",
    value=example_clicked or "",
    placeholder="Ask anything about the ingested papers...",
    height=100,
    label_visibility="collapsed",
)

ask_col, clear_col = st.columns([4, 1])
with ask_col:
    ask_btn = st.button("🔍 Ask", use_container_width=True)
with clear_col:
    if st.button("Clear", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

# Run query
if ask_btn and question.strip():
    with st.spinner("Retrieving relevant passages and generating answer..."):
        result = ask_question(question.strip(), top_k)

    if result:
        st.session_state.query_history.insert(0, {
            "question": question.strip(),
            "result": result,
        })

# Display results
if st.session_state.query_history:
    for entry in st.session_state.query_history:
        st.markdown("---")
        st.markdown(f"**Q:** {entry['question']}")

        res = entry["result"]
        st.markdown(
            f'<div class="answer-box">{res["answer"]}</div>',
            unsafe_allow_html=True,
        )

        model_tag = f"`{res['model']}`"
        st.caption(f"Model: {model_tag} · {len(res['sources'])} sources retrieved")

        with st.expander("📚 View sources", expanded=False):
            for i, src in enumerate(res["sources"], 1):
                score_pct = int(src["score"] * 100)
                st.markdown(f"""
<div class="source-card">
    <span class="score-badge">Score: {score_pct}%</span>
    <div class="paper-title">[{i}] {src['paper_title']}</div>
    <div class="page-info">📄 {src['paper_filename']} · Page {src['page_number']}</div>
    <div class="snippet">{src['content']}</div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #475569;">
        <div style="font-size:3rem">🔬</div>
        <p>Ask a question above to get started.</p>
        <p style="font-size:0.85rem">Make sure papers are ingested and the API is running.</p>
    </div>
    """, unsafe_allow_html=True)
