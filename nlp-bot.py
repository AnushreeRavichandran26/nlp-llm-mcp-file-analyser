import asyncio
import json
import math
import re
import os
import base64
from collections import Counter
from groq import Groq

# ══════════════════════════════════════════════════════════
#  STEP 1 — MCP CLIENT  (smart query_context filtering)
# ══════════════════════════════════════════════════════════

class MCPClient:
    def __init__(self):
        self.process = None
        self._id = 0

    def _next_id(self):
        self._id += 1
        return self._id

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            "python", "mcp_universal_file_server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        # Required MCP handshake
        await self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "chatbot", "version": "1.0.0"}
            }
        })

    async def _send(self, request):
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        return json.loads(line.decode().strip())

    async def load_document(self, path):
        return await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": "load_any_document", "arguments": {"path": path}}
        })

    async def smart_query(self, question: str) -> str:
        """
        Step 1 — Smart filtering via MCP.
        MCP's query_document_context already uses the question as a filter key.
        We pass the actual user question so the server returns only
        the relevant portion of the loaded document — not the full text.
        This is the primary context retrieval step.
        """
        res = await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "query_document_context",
                "arguments": {"query": question}   # ← question drives the filter
            }
        })
        try:
            raw = res["result"]["content"][0]["text"]
            # Extract just the document content block
            marker = "📖 Document Content for Analysis:"
            if marker in raw:
                return raw.split(marker, 1)[1].strip()
            return raw
        except Exception:
            return ""

    async def list_documents(self):
        return await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": "list_loaded_documents", "arguments": {}}
        })

    async def clear_documents(self):
        return await self._send({
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": "clear_document_context", "arguments": {}}
        })


# ══════════════════════════════════════════════════════════
#  STEP 2 — NLP REFINER  (optional TF-IDF + Cosine polish)
#
#  MCP already does smart filtering by question.
#  NLP kicks in only when the MCP context is still large
#  (>= 800 chars) — it re-ranks sentences by cosine similarity
#  to the question and trims to the top-k most relevant ones.
#  If MCP already returned a tight, focused chunk, NLP is skipped.
# ══════════════════════════════════════════════════════════

STOP_WORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but",
    "for","with","this","that","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","not","no","by","from","as","if","so",
    "its","their","they","we","you","he","she","i","my","your","our",
    "what","which","who","when","where","how","why","about","up","out",
    "into","than","then","there","these","those","also","just","only",
    "more","all","any","each","some","such","other","over","after","new"
}

NLP_THRESHOLD = 800   # chars — only refine if MCP context is larger than this

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in STOP_WORDS and len(w) > 1]

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]

def build_tfidf(sentences):
    tokenized = [tokenize(s) for s in sentences]
    N = len(tokenized)
    df = {}
    for tokens in tokenized:
        for w in set(tokens):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (f + 1)) + 1 for w, f in df.items()}
    vecs = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vecs.append({w: (c / total) * idf.get(w, 1) for w, c in tf.items()})
    return vecs, idf

def cosine(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[w] * b[w] for w in common)
    mag = lambda v: math.sqrt(sum(x**2 for x in v.values()))
    ma, mb = mag(a), mag(b)
    return dot / (ma * mb) if ma and mb else 0.0


class NLPRefiner:

    def refine(self, question: str, mcp_context: str, top_k: int = 5) -> tuple[str, bool]:
        """
        Returns (refined_context, was_refined).
        Skips refinement if MCP context is already short enough.
        """
        if len(mcp_context) < NLP_THRESHOLD:
            return mcp_context, False     # MCP context is tight — skip NLP

        sentences = split_sentences(mcp_context)
        if len(sentences) <= top_k:
            return mcp_context, False     # too few sentences to need filtering

        # Build TF-IDF for sentences + question
        all_texts  = sentences + [question]
        all_vecs, _  = build_tfidf(all_texts)
        sent_vecs  = all_vecs[:-1]
        q_vec      = all_vecs[-1]

        # Score and pick top-k, restore document order
        scored = [(cosine(q_vec, sv), i) for i, sv in enumerate(sent_vecs)]
        scored.sort(reverse=True)
        top_idx = sorted(i for _, i in scored[:top_k])
        refined = "\n".join(f"- {sentences[i]}" for i in top_idx)
        return refined, True

    def summarize(self, text: str, top_k: int = 5) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return "Nothing to summarize."
        if len(sentences) <= top_k:
            return " ".join(sentences)
        vecs, _ = build_tfidf(sentences)
        scores  = [sum(v.values()) / max(len(v), 1) for v in vecs]
        top_idx = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        )
        return " ".join(sentences[i] for i in top_idx)


# ══════════════════════════════════════════════════════════
#  STEP 3 — LLM  (Groq / Llama)
# ══════════════════════════════════════════════════════════

class LLM:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, question: str, context: str) -> str:
        """Step 3 — Answer using the refined context from NLP."""
        prompt = f"""You are a document analysis assistant.
Answer the question using ONLY the context provided below.
If the context doesn't contain enough information, say so clearly.

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
        """Vision model for image files — bypasses NLP pipeline."""
        try:
            ext  = os.path.splitext(image_path)[1].lower()
            mime = {".jpg":"image/jpeg",".jpeg":"image/jpeg",
                    ".png":"image/png",".webp":"image/webp"}.get(ext,"image/jpeg")
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
#  CHATBOT  — orchestrates all 3 steps
# ══════════════════════════════════════════════════════════

IMAGE_EXTS   = {".jpg",".jpeg",".png",".gif",".bmp",".webp",".tiff"}
VISUAL_WORDS = {"image","photo","picture","describe","show","see",
                "color","colour","person","face","look","what is in","what's in"}

class Chatbot:
    def __init__(self, api_key):
        self.mcp = MCPClient()
        self.nlp = NLPRefiner()
        self.llm = LLM(api_key)
        self.loaded: dict[str, str] = {}   # path → "image" | "text"

    def _is_image(self, path):
        return os.path.splitext(path)[1].lower() in IMAGE_EXTS

    # ── load ─────────────────────────────────────────────
    async def _load(self, path):
        res = await self.mcp.load_document(path)
        try:
            print(res["result"]["content"][0]["text"])
            self.loaded[path] = "image" if self._is_image(path) else "text"
        except Exception:
            err = res.get("error", {}).get("message", "Unknown error")
            print(f"❌ Load failed: {err}")
            if path.lower().endswith(".pdf"):
                print("   Fix: pip install PyPDF2")

    # ── summarize ────────────────────────────────────────
    async def _summarize(self):
        print("  [MCP] Fetching context with query 'full document summary'...")
        ctx = await self.mcp.smart_query("full document summary overview")
        if not ctx:
            print("❌ No document loaded. Use: load <path>")
            return
        print("  [NLP] Extractive summarization via TF-IDF...")
        summary = self.nlp.summarize(ctx, top_k=5)
        print(f"\n📄 Summary:\n{summary}")

    # ── main pipeline ────────────────────────────────────
    async def _ask(self, question):
        # Image branch — bypass text pipeline
        images = [p for p, t in self.loaded.items() if t == "image"]
        if images and any(w in question.lower() for w in VISUAL_WORDS):
            print("  [IMG] Routing to vision model...")
            for img in images:
                print(f"\n🤖 Vision Answer:\n{self.llm.vision(img, question)}")
            return

        # ── Step 1: MCP smart_query ───────────────────
        print("  [MCP] Running smart query_context with your question...")
        mcp_context = await self.mcp.smart_query(question)
        if not mcp_context:
            print("❌ No document loaded. Use: load <path>")
            return
        print(f"        → {len(mcp_context)} chars returned from MCP")

        # ── Step 2: NLP optional refine ───────────────
        context, was_refined = self.nlp.refine(question, mcp_context, top_k=5)
        if was_refined:
            print(f"  [NLP] Context was large — TF-IDF re-ranked to top 5 sentences")
        else:
            print(f"  [NLP] Context already focused — skipping refinement")

        # ── Step 3: LLM answer ────────────────────────
        print("  [LLM] Generating answer...")
        answer = self.llm.generate(question, context)
        print(f"\n🤖 Answer:\n{answer}")

    # ── chat loop ────────────────────────────────────────
    async def run(self):
        await self.mcp.start()

        print("🤖 Chatbot  |  MCP (smart filter) → NLP (refine) → LLM")
        print("━" * 55)
        print("  load <path>   — load any file (txt, pdf, docx …)")
        print("  summarize     — extractive TF-IDF summary")
        print("  list          — show loaded documents")
        print("  clear         — unload all documents")
        print("  <question>    — run the full pipeline")
        print("  exit          — quit")
        print("━" * 55)
        print("  PDF:  pip install PyPDF2")
        print("  DOCX: pip install python-docx")
        print("━" * 55 + "\n")

        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if not user:
                continue

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
                try:
                    print(res["result"]["content"][0]["text"])
                except Exception:
                    print("❌ Could not list documents.")

            elif low == "clear":
                res = await self.mcp.clear_documents()
                try:
                    print(res["result"]["content"][0]["text"])
                except Exception:
                    print("❌ Could not clear documents.")
                self.loaded.clear()

            else:
                await self._ask(user)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

async def main():
    api_key = input("🔑 Enter your Groq API key: ").strip()
    bot = Chatbot(api_key)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
