"""Streamlit UI for RAG Vector Search."""

import streamlit as st
import requests

API_URL = "http://localhost:8000"


def check_api():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

    
def search_documents(query: str, limit: int = 5):
    try:
        resp = requests.post(
            f"{API_URL}/search",
            json={"query": query, "limit": limit},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Request failed"}


def rag_query(query: str, limit: int = 5, use_faiss: bool = False):
    try:
        resp = requests.post(
            f"{API_URL}/rag",
            json={"query": query, "limit": limit, "use_hybrid": True, "use_faiss": use_faiss, "stream": False},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Request failed"}


def get_stats():
    try:
        resp = requests.get(f"{API_URL}/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


st.set_page_config(
    page_title="RAG Vector Search",
    page_icon="🔍",
    layout="wide",
)

st.title("RAG Vector Search")
st.markdown("Search arXiv papers using semantic vector search with LLM-powered answers")

api_status = check_api()
if api_status:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Database", api_status.get("database", "unknown"))
    with col2:
        st.metric("FAISS Index", "Loaded" if api_status.get("faiss_loaded") else "Not loaded")
    with col3:
        st.metric("Embedding Model", "Ready" if api_status.get("model_loaded") else "Loading")
else:
    st.error("API is not running. Please start the backend with: `.venv/Scripts/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`")

st.divider()

tab1, tab2 = st.tabs(["Search", "RAG Query"])

with tab1:
    st.header("Vector Search")
    
    search_col, results_col = st.columns([1, 2])
    
    with search_col:
        search_query = st.text_input("Search query:", placeholder="e.g., deep learning neural networks")
        top_k = st.slider("Number of results:", 1, 20, 5)
        search_btn = st.button("Search", type="primary")
    
    with results_col:
        if search_btn and search_query:
            with st.spinner("Searching..."):
                results = search_documents(search_query, top_k)
            
            if "error" in results:
                st.error(results["error"])
            elif results:
                st.success(f"Found {len(results)} documents")
                for i, doc in enumerate(results, 1):
                    with st.expander(f"Document {i} (ID: {doc['id']}, Distance: {doc['distance']:.4f})"):
                        st.text(doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"])
            else:
                st.warning("No results found")
        else:
            st.info("Enter a search query and click Search")

with tab2:
    st.header("RAG Query")
    st.caption("Ask questions and get answers powered by LLM with retrieved context")
    
    query_col, answer_col = st.columns([1, 2])
    
    with query_col:
        rag_query_text = st.text_input("Your question:", placeholder="e.g., What is deep learning?")
        rag_top_k = st.slider("Documents to retrieve:", 1, 10, 3)
        use_faiss = st.checkbox("Use FAISS (faster)", value=False)
        rag_btn = st.button("Ask", type="primary")
    
    with answer_col:
        if rag_btn and rag_query_text:
            with st.spinner("Generating answer... (this may take a moment)"):
                result = rag_query(rag_query_text, rag_top_k, use_faiss=use_faiss)
            
            if "error" in result:
                st.error(result["error"])
            elif result:
                engine = result.get("retrieval_engine", "pgvector").upper()
                st.info(f"Retrieved using: {engine}")
                st.subheader("Answer:")
                st.write(result.get("answer", "No answer generated"))
                
                if result.get("documents"):
                    st.divider()
                    st.subheader("Retrieved Documents:")
                    for i, doc in enumerate(result["documents"], 1):
                        with st.expander(f"Doc {i}: {doc['content'][:80]}..."):
                            st.text(doc["content"])
            else:
                st.warning("No results")
        else:
            st.info("Enter a question and click Ask")

st.divider()

with st.expander("API Stats"):
    stats = get_stats()
    if stats:
        st.json(stats)
    else:
        st.warning("Could not fetch stats")
