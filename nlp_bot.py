"""
nlp_bot.py — Production NLP Chatbot  (Upgraded from original)
==============================================================
Pipeline remains: MCP → NLP Refinement → LLM   (unchanged as required)

Upgrades in this file:
  • NLPEngine (semantic + TF-IDF + NER + keywords) replaces inline refiner
  • Evaluation module: response time, context precision, chunk stats
  • Cleaner async flow; image branch preserved
"""

import asyncio
import json
import os
import base64
import time
import logging
from groq import Groq

# Import our new NLP engine (replaces the inline NLPRefiner)
from nlp_engine import NLPEngine

logging.basicConfig(level=logging.WARNING)   # keep output clean in chat
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  STEP 1 — MCP CLIENT  (unchanged interface from original)
#  MCP is still the PRIMARY document loader.
# ══════════════════════════════════════════════════════════

class MCPClient:
    """
    Thin async wrapper around the MCP subprocess (stdio transport).
    No changes to the public interface — the pipeline contract is intact.
    """

    def __init__(self):
        self.process = None
        self._id = 0

    def _next_id(self):
        self._id += 1
        return self._id

    async def start(self, documents_dir: str):
        import sys
        import os

        server_path = os.path.join(os.path.dirname(__file__), "mcp_universal_file_server.py")

        self.process = await asyncio.create_subprocess_exec(
            sys.executable,          # ✅ correct python
            server_path,             # ✅ absolute path
            "--directory", documents_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE   # ✅ show errors
        )
        await asyncio.sleep(1)

        # 🔥 IMPORTANT CHECK
        if self.process is None:
            raise RuntimeError("❌ MCP process failed to start")

        print("✅ MCP started at:", server_path)
        # Required handshake
        await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method":  "initialize",
            "params":  {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "clientInfo":      {"name": "nlp-bot", "version": "2.0.0"}
            }
        })

    async def _send(self, request: dict) -> dict:
        # 🔥 safety check (prevents your current error)
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("❌ MCP process not running")

        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        return json.loads(line.decode().strip())

    async def load_document(self, path: str) -> dict:
        return await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method":  "tools/call",
            "params":  {"name": "load_any_document", "arguments": {"path": path}}
        })

    async def get_full_content(self, question: str) -> str:
        """
        Retrieve raw document content from MCP (Step 1).
        MCP still filters by question context internally.
        """
        res = await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method":  "tools/call",
            "params":  {
                "name":      "query_document_context",
                "arguments": {"query": question}
            }
        })
        try:
            raw    = res["result"]["content"][0]["text"]
            marker = "📖 Document Content for Analysis:"
            return raw.split(marker, 1)[1].strip() if marker in raw else raw
        except Exception:
            return ""

    async def list_documents(self) -> dict:
        return await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method":  "tools/call",
            "params":  {"name": "list_loaded_documents", "arguments": {}}
        })

    async def clear_documents(self) -> dict:
        return await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method":  "tools/call",
            "params":  {"name": "clear_document_context", "arguments": {}}
        })


# ══════════════════════════════════════════════════════════
#  STEP 2 — NLP ENGINE  (replaces inline NLPRefiner)
#  Chunking → FAISS semantic search → TF-IDF re-rank
#  → NER → keyword extraction
#  (imported from nlp_engine.py)
# ══════════════════════════════════════════════════════════
# (NLPEngine is imported above)


# ══════════════════════════════════════════════════════════
#  STEP 3 — LLM  (Groq / Llama — unchanged from original)
# ══════════════════════════════════════════════════════════

class LLM:
    """Groq LLM wrapper — kept identical to original."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate(self, question: str, context: str,
                 entities: dict = None, keywords: list = None) -> str:
        """
        UPGRADE: Optionally inject NER + keyword hints into the prompt
        so the LLM can use them as structured signals.
        """
        ner_hint = ""
        if entities:
            ner_lines = []
            for label, items in list(entities.items())[:4]:
                ner_lines.append(f"  {label}: {', '.join(items[:3])}")
            if ner_lines:
                ner_hint = "\n\nKey Entities in Context:\n" + "\n".join(ner_lines)

        kw_hint = ""
        if keywords:
            kw_words = [w for w, _ in keywords[:6]]
            kw_hint  = f"\n\nTop Keywords: {', '.join(kw_words)}"

        prompt = f"""You are a document analysis assistant.
Answer the question using ONLY the context provided below.
If the context doesn't contain enough information, say so clearly.
{ner_hint}{kw_hint}

Context:
{context}

Question: {question}

Answer:"""
        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"❌ LLM Error: {e}"

    def vision(self, image_path: str, question: str) -> str:
        """Vision model for image files — bypasses NLP pipeline (unchanged)."""
        try:
            ext  = os.path.splitext(image_path)[1].lower()
            mime = {".jpg":"image/jpeg",".jpeg":"image/jpeg",
                    ".png":"image/png",".webp":"image/webp"}.get(ext, "image/jpeg")
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            res = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": [
                    {"type": "text",      "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]}],
                max_tokens=1024
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"❌ Vision Error: {e}"


# ══════════════════════════════════════════════════════════
#  EVALUATION MODULE
#  Academic requirement: show measurable improvement metrics.
# ══════════════════════════════════════════════════════════

class Evaluator:
    """
    Lightweight evaluator for academic comparison.

    Metrics tracked per query:
      • response_time_s    — wall-clock time for the full pipeline
      • context_length     — characters sent to the LLM
      • chunks_used        — how many chunks were retrieved
      • retrieval_method   — 'semantic+tfidf' or 'tfidf_only'
      • entities_found     — number of NE types detected
      • keywords_found     — number of keywords extracted

    Academic use: compare 'before' (full doc → LLM) vs 'after'
    (chunked retrieval → LLM) on context length and response time.
    """

    def __init__(self):
        self.history = []    # list of metric dicts

    def record(self, query: str, metrics: dict):
        metrics["query"] = query[:60]
        self.history.append(metrics)

    def summary(self) -> str:
        if not self.history:
            return "No queries evaluated yet."
        n   = len(self.history)
        avg = lambda key: sum(r.get(key, 0) for r in self.history) / n
        lines = [
            f"📊 Evaluation Summary ({n} queries)",
            f"{'─'*40}",
            f"  Avg response time  : {avg('response_time_s'):.2f}s",
            f"  Avg context length : {avg('context_length'):.0f} chars",
            f"  Avg chunks used    : {avg('chunks_used'):.1f}",
            f"  Avg entities found : {avg('entities_found'):.1f} types",
            f"  Avg keywords found : {avg('keywords_found'):.1f}",
        ]
        methods = [r.get("retrieval_method","?") for r in self.history]
        lines.append(f"  Retrieval methods  : {set(methods)}")
        return "\n".join(lines)

    def last(self) -> dict:
        return self.history[-1] if self.history else {}


# ══════════════════════════════════════════════════════════
#  CHATBOT — orchestrates MCP → NLP → LLM  (upgraded)
# ══════════════════════════════════════════════════════════

IMAGE_EXTS   = {".jpg",".jpeg",".png",".gif",".bmp",".webp",".tiff"}
VISUAL_WORDS = {"image","photo","picture","describe","show","see",
                "color","colour","person","face","look","what is in","what's in"}


class Chatbot:
    def __init__(self, api_key: str, documents_dir: str = "./documents"):
        self.mcp      = MCPClient()
        self.nlp      = NLPEngine()     # UPGRADE: full NLP engine
        self.llm      = LLM(api_key)
        self.evaluator = Evaluator()
        self.loaded: dict[str, str] = {}          # path → "image" | "text"
        self.documents_dir = documents_dir
        self._nlp_indexed  = False

    def _is_image(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in IMAGE_EXTS

    # ── load ─────────────────────────────────────────────
    async def _load(self, path: str):
        print(f"  [MCP] Loading: {path} …")
        res = await self.mcp.load_document(path)
        try:
            msg = res["result"]["content"][0]["text"]
            print(msg)
            kind = "image" if self._is_image(path) else "text"
            self.loaded[path] = kind
            # If it's a text file, fetch its content and build NLP index
            if kind == "text":
                print("  [NLP] Indexing document (chunking + embedding) …")
                raw_content = await self.mcp.get_full_content("full document content")
                if raw_content:
                    n_chunks = self.nlp.index_document(raw_content)
                    print(f"  [NLP] Indexed {n_chunks} chunks")
                    self._nlp_indexed = True
        except Exception:
            err = res.get("error", {}).get("message", "Unknown error")
            print(f"❌ Load failed: {err}")
            if path.lower().endswith(".pdf"):
                print("   Fix: pip install PyPDF2")

    # ── summarise ────────────────────────────────────────
    async def _summarize(self):
        print("  [MCP] Fetching document content …")
        raw = await self.mcp.get_full_content("full document summary")
        if not raw:
            print("❌ No document loaded."); return
        print("  [NLP] Extractive summarisation (TF-IDF) …")
        summary = self.nlp.summarize(raw, top_k=5)
        print(f"\n📄 Summary:\n{summary}")

    # ── main pipeline ────────────────────────────────────
    async def _ask(self, question: str):
        t0 = time.time()

        # Image branch — unchanged
        images = [p for p, t in self.loaded.items() if t == "image"]
        if images and any(w in question.lower() for w in VISUAL_WORDS):
            print("  [IMG] Routing to vision model …")
            for img in images:
                print(f"\n🤖 Vision Answer:\n{self.llm.vision(img, question)}")
            return

        # ── Step 1: MCP — primary document fetch ──────────────────
        print("  [MCP] Fetching document context …")
        mcp_context = await self.mcp.get_full_content(question)
        if not mcp_context:
            print("❌ No document loaded. Use: load <path>"); return
        print(f"        → {len(mcp_context)} chars from MCP")

        # ── Step 2: NLP — chunk retrieval + enrichment ────────────
        if self._nlp_indexed:
            print("  [NLP] Semantic+TF-IDF chunk retrieval …")
            nlp_result = self.nlp.retrieve(question, top_k=5)
        else:
            # First query before explicit load: index on-the-fly
            self.nlp.index_document(mcp_context)
            nlp_result = self.nlp.retrieve(question, top_k=5)

        context = nlp_result["context"] or mcp_context
        method  = nlp_result["method"]
        chunks  = nlp_result["chunks_used"]
        entities = nlp_result["entities"]
        keywords = nlp_result["keywords"]

        print(f"        → {len(context)} chars after retrieval "
              f"({chunks} chunks, method: {method})")

        # Print NLP insights
        if entities:
            print(f"  [NER] Entities: {self.nlp.ner.format_entities(entities)}")
        if keywords:
            print(f"  [KW]  Keywords: {self.nlp.keywords.format_keywords(keywords)}")

        # ── Step 3: LLM — generate answer ─────────────────────────
        print("  [LLM] Generating answer …")
        answer = self.llm.generate(question, context, entities, keywords)
        elapsed = time.time() - t0

        print(f"\n🤖 Answer:\n{answer}")
        print(f"\n⏱  {elapsed:.2f}s  |  {len(context)} chars to LLM  |  {chunks} chunks")

        # Record for evaluation
        self.evaluator.record(question, {
            "response_time_s":  elapsed,
            "context_length":   len(context),
            "chunks_used":      chunks,
            "retrieval_method": method,
            "entities_found":   len(entities),
            "keywords_found":   len(keywords),
        })

    # ── chat loop ────────────────────────────────────────
    async def run(self):
        await self.mcp.start(documents_dir=self.documents_dir)
        print("🤖 NLP Chatbot v2  |  MCP → NLP (semantic+TF-IDF+NER) → LLM")
        print("━" * 60)
        print("  load <path>   — load a file from ./documents/")
        print("  summarize     — extractive TF-IDF summary")
        print("  list          — show loaded documents")
        print("  clear         — unload all documents")
        print("  eval          — show evaluation metrics")
        print("  <question>    — run the full pipeline")
        print("  exit          — quit")
        print("━" * 60)
        print("  Deps: pip install sentence-transformers faiss-cpu spacy groq")
        print("        python -m spacy download en_core_web_sm")
        print("━" * 60 + "\n")

        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if not user: continue

            low = user.lower()

            if low in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            elif low.startswith("load "):
                await self._load(user[5:].strip())

            elif low in ("summarize", "summary"):
                await self._summarize()

            elif low == "list":
                res = await self.mcp.list_documents()
                try: print(res["result"]["content"][0]["text"])
                except Exception: print("❌ Could not list.")

            elif low == "clear":
                res = await self.mcp.clear_documents()
                try: print(res["result"]["content"][0]["text"])
                except Exception: print("❌ Could not clear.")
                self.loaded.clear()
                self._nlp_indexed = False

            elif low in ("eval", "evaluate", "metrics"):
                print(self.evaluator.summary())

            else:
                await self._ask(user)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

async def main():
    api_key       = input("🔑 Enter your Groq API key: ").strip()
    docs_dir      = input("📁 Documents directory [./documents]: ").strip() or "./documents"
    os.makedirs(docs_dir, exist_ok=True)
    bot = Chatbot(api_key, documents_dir=docs_dir)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
