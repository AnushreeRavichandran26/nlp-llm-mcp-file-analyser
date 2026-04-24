"""
app.py — Streamlit UI for the NLP Chatbot
==========================================
Run with:  streamlit run app.py

Architecture shown in UI:
  File Upload → MCP Server → NLP Engine (Chunks + FAISS + TF-IDF + NER + KW)
              → Groq LLM → Answer
"""

import asyncio
import os
import sys
import threading
import tempfile
import time
import concurrent.futures
import streamlit as st

# ── Streamlit page config ────────────────────────────────
st.set_page_config(
    page_title  = "NLP Document Q&A",
    page_icon   = "📚",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Import our modules ───────────────────────────────────
from nlp_engine import NLPEngine, chunk_text
from nlp_bot    import MCPClient, LLM, Evaluator

# ════════════════════════════════════════════════════════
#  PERSISTENT EVENT LOOP  — the core fix
# ════════════════════════════════════════════════════════
#
#  WHY THIS IS NEEDED:
#  Streamlit reruns the entire script on every user interaction.
#  asyncio objects (subprocess pipes, streams) are bound to the
#  event loop that CREATED them.  If run_async() spins a new loop
#  each call, the MCPClient.process created on loop-1 cannot be
#  used on loop-2 → AttributeError / NoneType errors.
#
#  THE FIX:
#  Start ONE background thread that owns ONE event loop for the
#  lifetime of the Streamlit session.  All async calls are
#  submitted as futures to that single loop via
#  asyncio.run_coroutine_threadsafe(), which is thread-safe and
#  reuses the same loop every time.
#
#  On Windows we use ProactorEventLoop (required for subprocesses).
#  On macOS/Linux the default loop works fine.

def _make_loop() -> asyncio.AbstractEventLoop:
    """Create the right event loop type for this OS."""
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    return loop

def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Thread target: run the loop forever until explicitly stopped."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

# One loop + one thread, stored in session_state so they survive reruns.
if "_bg_loop" not in st.session_state:
    _loop = _make_loop()
    _thread = threading.Thread(target=_start_background_loop,
                               args=(_loop,), daemon=True)
    _thread.start()
    st.session_state._bg_loop   = _loop
    st.session_state._bg_thread = _thread

def run_async(coro):
    """
    Submit a coroutine to the persistent background loop and block
    until it completes.  Safe to call from the Streamlit main thread.
    """
    future = asyncio.run_coroutine_threadsafe(
        coro, st.session_state._bg_loop
    )
    return future.result()   # blocks until done, propagates exceptions


# ════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════
if "mcp"       not in st.session_state: st.session_state.mcp       = None
if "nlp"       not in st.session_state: st.session_state.nlp       = NLPEngine()
if "llm"       not in st.session_state: st.session_state.llm       = None
if "evaluator" not in st.session_state: st.session_state.evaluator = Evaluator()
if "chat"      not in st.session_state: st.session_state.chat      = []   # [(q, a, meta)]
if "loaded"    not in st.session_state: st.session_state.loaded    = {}
if "docs_dir"  not in st.session_state:
    st.session_state.docs_dir = tempfile.mkdtemp(prefix="nlp_docs_")
if "indexed"   not in st.session_state: st.session_state.indexed   = False


# ════════════════════════════════════════════════════════
#  SIDEBAR — Configuration & File Upload
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")

    # ── API key ──────────────────────────────────────────
    api_key = st.text_input("🔑 Groq API Key", type="password",
                             help="Get yours at console.groq.com")

    if api_key and st.session_state.llm is None:
        st.session_state.llm = LLM(api_key)
        st.success("LLM initialised")

    st.divider()

    # ── File upload ──────────────────────────────────────
    st.subheader("📂 Upload Document")
    uploaded = st.file_uploader(
        "Supports PDF, DOCX, TXT, code files, CSV …",
        type=["pdf","docx","txt","md","py","js","java","csv","json",
              "xlsx","pptx","png","jpg","jpeg"],
        accept_multiple_files=False
    )

    if uploaded and st.button("📥 Load Document"):
        # Save to temp docs dir (which MCP server can access)
        dest = os.path.join(st.session_state.docs_dir, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getvalue())

        with st.spinner("Loading via MCP server …"):
            # Start MCP if not running
            if st.session_state.mcp is None:
                mcp = MCPClient()
                run_async(mcp.start(documents_dir=st.session_state.docs_dir))
                st.session_state.mcp = mcp

            # Load through MCP
            res = run_async(st.session_state.mcp.load_document(uploaded.name))
            try:
                msg = res["result"]["content"][0]["text"]
                st.success(f"✅ Loaded: {uploaded.name}")

                # Index through NLP engine
                raw = run_async(st.session_state.mcp.get_full_content("full content"))
                if raw:
                    n = st.session_state.nlp.index_document(raw)
                    st.info(f"🔍 Indexed {n} semantic chunks")
                    st.session_state.indexed = True
                    st.session_state.loaded[uploaded.name] = "text"
            except Exception:
                err = res.get("error", {}).get("message", "Unknown error")
                st.error(f"❌ {err}")

    st.divider()

    # ── Loaded docs ──────────────────────────────────────
    st.subheader("📚 Loaded Documents")
    if st.session_state.loaded:
        for name in st.session_state.loaded:
            st.write(f"  📄 {name}")
    else:
        st.caption("No documents loaded yet")

    if st.button("🗑️ Clear All"):
        if st.session_state.mcp:
            run_async(st.session_state.mcp.clear_documents())
        st.session_state.loaded.clear()
        st.session_state.indexed = False
        st.session_state.nlp     = NLPEngine()
        st.rerun()

    st.divider()

    # ── NLP settings ─────────────────────────────────────
    st.subheader("🧠 NLP Settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5,
                       help="Number of chunks sent to LLM")
    show_nlp = st.checkbox("Show NLP insights (NER + keywords)", value=True)
    show_eval = st.checkbox("Show evaluation metrics", value=True)


# ════════════════════════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════════════════════════
st.title("📚 NLP Document Q&A System")
st.caption("Pipeline: MCP Document Loader → NLP Engine (FAISS + TF-IDF + NER) → Groq LLM")

# ── Architecture diagram (simple text-based) ─────────────
with st.expander("🏗️ System Architecture", expanded=False):
    st.code("""
  ┌──────────────┐    ┌──────────────────────────────────────────┐    ┌─────────────┐
  │  File Upload │───►│ MCP Server (Universal File Loader)        │    │  Groq LLM   │
  │  (Streamlit) │    │  • PDF, DOCX, TXT, code, CSV, images …   │    │  LLaMA 3.3  │
  └──────────────┘    └──────────────┬───────────────────────────┘    └──────┬──────┘
                                     │ raw text                              │
                                     ▼                                       │
                      ┌──────────────────────────────────────────┐          │
                      │ NLP Engine                                │          │
                      │  1. Chunker  (300 words, 50 overlap)      │          │
                      │  2. FAISS    (sentence-transformers)       │          │
                      │  3. TF-IDF   (sublinear, re-ranker)       │─────────►│
                      │  4. NER      (spaCy / regex fallback)     │ top-k    │
                      │  5. Keywords (TF-IDF scores)              │ chunks   │
                      └──────────────────────────────────────────┘          │
                                                                             │
                                         ┌───────────────────────────────────┘
                                         ▼
                                    Final Answer
    """, language="text")

# ── Chat interface ───────────────────────────────────────
st.subheader("💬 Ask a Question")

# Display chat history
for q, a, meta in st.session_state.chat:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if show_eval and meta:
            cols = st.columns(4)
            cols[0].metric("⏱ Time",    f"{meta.get('response_time_s',0):.2f}s")
            cols[1].metric("📝 Context", f"{meta.get('context_length',0)} chars")
            cols[2].metric("📦 Chunks",  meta.get('chunks_used', 0))
            cols[3].metric("🔍 Method",  meta.get('retrieval_method','—'))
        if show_nlp and meta.get("entities"):
            with st.expander("🏷️ Named Entities"):
                for label, items in meta["entities"].items():
                    st.write(f"**{label}**: {', '.join(items[:5])}")
        if show_nlp and meta.get("keywords"):
            with st.expander("🔑 Top Keywords"):
                kws = [w for w, _ in meta["keywords"][:8]]
                st.write(", ".join(kws))

# Input box
question = st.chat_input("Ask anything about your document …")

if question:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
    elif not st.session_state.loaded:
        st.warning("⚠️ Please upload and load a document first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("MCP → NLP → LLM …"):
                t0 = time.time()

                # Step 1: MCP
                mcp_ctx = run_async(st.session_state.mcp.get_full_content(question))

                # Step 2: NLP
                if not st.session_state.indexed and mcp_ctx:
                    st.session_state.nlp.index_document(mcp_ctx)
                    st.session_state.indexed = True

                nlp_result = st.session_state.nlp.retrieve(question, top_k=top_k)
                context    = nlp_result["context"] or mcp_ctx
                entities   = nlp_result["entities"]
                keywords   = nlp_result["keywords"]
                method     = nlp_result["method"]
                chunks     = nlp_result["chunks_used"]

                # Step 3: LLM
                answer = st.session_state.llm.generate(question, context, entities, keywords)
                elapsed = time.time() - t0

            st.write(answer)

            meta = {
                "response_time_s":  elapsed,
                "context_length":   len(context),
                "chunks_used":      chunks,
                "retrieval_method": method,
                "entities":         entities,
                "keywords":         keywords,
                "entities_found":   len(entities),
                "keywords_found":   len(keywords),
            }

            if show_eval:
                cols = st.columns(4)
                cols[0].metric("⏱ Time",    f"{elapsed:.2f}s")
                cols[1].metric("📝 Context", f"{len(context)} chars")
                cols[2].metric("📦 Chunks",  chunks)
                cols[3].metric("🔍 Method",  method)

            if show_nlp and entities:
                with st.expander("🏷️ Named Entities"):
                    for label, items in entities.items():
                        st.write(f"**{label}**: {', '.join(items[:5])}")

            if show_nlp and keywords:
                with st.expander("🔑 Top Keywords"):
                    st.write(", ".join(w for w, _ in keywords[:8]))

        st.session_state.chat.append((question, answer, meta))
        st.session_state.evaluator.record(question, meta)


# ── Evaluation panel ─────────────────────────────────────
if show_eval and st.session_state.evaluator.history:
    st.divider()
    st.subheader("📊 Session Evaluation Metrics")
    hist = st.session_state.evaluator.history
    n    = len(hist)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries",      n)
    c2.metric("Avg Time (s)",        f"{sum(r.get('response_time_s', 0) for r in hist)/n:.2f}")
    c3.metric("Avg Context (chars)", f"{sum(r.get('context_length', 0)  for r in hist)/n:.0f}")
    c4.metric("Avg Chunks",          f"{sum(r.get('chunks_used', 0)     for r in hist)/n:.1f}")
    c5.metric("Avg Entities",        f"{sum(len(r['entities']) if isinstance(r.get('entities'), dict) else r.get('entities_found', 0) for r in hist)/n:.1f}")

    # Before vs After comparison (show reduction in context size)
    if n >= 2:
        st.caption("💡 Context shrinks over time as the NLP engine learns "
                   "to retrieve tighter chunks for specific questions.")