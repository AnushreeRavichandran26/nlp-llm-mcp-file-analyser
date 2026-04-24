"""
nlp_engine.py — Production NLP Engine (Upgrade for Academic NLP Project)
=========================================================================
Pipeline: raw text → chunk → embed (sentence-transformers) → FAISS index
          + TF-IDF re-ranking → NER → keyword extraction → refined context

Why each part matters (for your NLP report):
  • Chunking      — prevents context window overflow; keeps semantics local
  • Embeddings    — dense vectors capture meaning, not just keyword overlap
  • FAISS         — fast approximate nearest-neighbour search at scale
  • TF-IDF        — lightweight lexical scorer, works without GPU
  • NER           — shows *who/what/where* the answer involves
  • Keywords      — highlights the most informative terms in context
"""

import math
import re
import logging
from collections import Counter
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  SECTION 1 — CHUNKER
#  Break a long document into overlapping windows so each
#  chunk is small enough to embed and retrieve meaningfully.
# ─────────────────────────────────────────────────────────────

def chunk_text(text: str,
               chunk_size: int = 300,
               overlap: int = 50) -> List[str]:
    """
    Split text into word-level chunks with overlap.

    Example: chunk_size=300 words, overlap=50 words means consecutive
    chunks share 50 words → smoother semantic boundaries.

    Academic note: Overlapping chunks improve recall for answers that
    span a sentence boundary between two non-overlapping windows.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap          # stride between chunk starts
    for start in range(0, len(words), step):
        chunk = " ".join(words[start: start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


# ─────────────────────────────────────────────────────────────
#  SECTION 2 — SEMANTIC RETRIEVER  (FAISS + sentence-transformers)
#  Builds a dense vector index so we can find the top-k chunks
#  most *semantically* similar to a question — not just lexically.
# ─────────────────────────────────────────────────────────────

class SemanticRetriever:
    """
    Encodes chunks with sentence-transformers and stores them in FAISS.
    Falls back gracefully to TF-IDF if dependencies are missing.

    Academic relevance:
      • Dense retrieval (bi-encoder) vs sparse retrieval (TF-IDF) is
        one of the central comparisons in modern IR / NLP research.
      • FAISS (Facebook AI Similarity Search) provides sub-linear
        nearest-neighbour lookup via Hierarchical NSW or IVF indexes.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model      = None
        self.index      = None       # FAISS index
        self.chunks: List[str] = []
        self._available = False
        self._try_load()

    def _try_load(self):
        """Load sentence-transformers + FAISS; set flag if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss                           # noqa: F401
            self.model      = SentenceTransformer(self.model_name)
            self._available = True
            logger.info("✅ Semantic retriever ready (sentence-transformers + FAISS)")
        except ImportError:
            logger.warning("⚠️  sentence-transformers or faiss not found — "
                           "falling back to TF-IDF only.\n"
                           "   Install: pip install sentence-transformers faiss-cpu")

    def build_index(self, chunks: List[str]) -> None:
        """Embed all chunks and insert into a flat FAISS L2 index."""
        if not self._available or not chunks:
            return
        import faiss
        import numpy as np

        self.chunks = chunks
        logger.info(f"  Building FAISS index for {len(chunks)} chunks …")
        embeddings = self.model.encode(chunks,
                                       batch_size=32,
                                       show_progress_bar=False,
                                       convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        # Normalise → cosine similarity becomes dot product
        faiss.normalize_L2(embeddings)

        dim         = embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)   # Inner Product ≈ cosine after L2 norm
        self.index.add(embeddings)
        logger.info(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return [(chunk_text, similarity_score), …] for the top-k chunks.
        Falls back to empty list if FAISS is unavailable.
        """
        if not self._available or self.index is None or not self.chunks:
            return []
        import faiss
        import numpy as np

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    @property
    def available(self):
        return self._available


# ─────────────────────────────────────────────────────────────
#  SECTION 3 — TF-IDF RE-RANKER  (improved from original)
#  Original used sentence-level TF-IDF.  Upgrade: chunk-level
#  TF-IDF that also uses sublinear TF scaling (log(1+tf))
#  for better discrimination.
# ─────────────────────────────────────────────────────────────

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

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in STOP_WORDS and len(w) > 1]


class TFIDFRanker:
    """
    UPGRADE: Uses sublinear TF scaling (log(1 + tf)) to reduce the
    dominance of very frequent terms — a standard improvement in IR.

    Academic note: This is the BM25-lite approach; full BM25 adds a
    document-length normalisation term we can also add for bonus marks.
    """

    def fit(self, documents: List[str]) -> None:
        """Compute IDF weights from a corpus of documents (chunks)."""
        self._docs  = documents
        N = len(documents)
        df: Dict[str, int] = {}
        self._tokenized = []
        for doc in documents:
            toks = tokenize(doc)
            self._tokenized.append(toks)
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        # Smoothed IDF to avoid zero weights
        self._idf = {w: math.log((N + 1) / (f + 1)) + 1 for w, f in df.items()}

    def _vec(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        # Sublinear TF: replace raw count with log(1+tf)
        return {w: (1 + math.log(c)) * self._idf.get(w, 1)
                for w, c in tf.items()}

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common: return 0.0
        dot = sum(a[w] * b[w] for w in common)
        mag = lambda v: math.sqrt(sum(x*x for x in v.values()))
        ma, mb = mag(a), mag(b)
        return dot / (ma * mb) if ma and mb else 0.0

    def rank(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (doc, score) pairs ranked by cosine to query."""
        if not hasattr(self, "_docs"):
            return []
        q_vec = self._vec(tokenize(query))
        scored = []
        for i, toks in enumerate(self._tokenized):
            d_vec = self._vec(toks)
            scored.append((self._cosine(q_vec, d_vec), i))
        scored.sort(reverse=True)
        return [(self._docs[i], score) for score, i in scored[:top_k]]


# ─────────────────────────────────────────────────────────────
#  SECTION 4 — NAMED ENTITY RECOGNISER (NER)
#  Academic NLP requirement: identify PERSON, ORG, DATE, LOC, etc.
#  Uses spaCy if available; falls back to regex-based heuristics.
# ─────────────────────────────────────────────────────────────

class NERExtractor:
    """
    Extract named entities from context to enrich the answer with
    structured metadata (who, what, where, when).

    Academic relevance: NER is a core NLP task demonstrating
    sequence labelling (CRF / BiLSTM-CRF / transformer approaches).
    """

    def __init__(self):
        self._nlp = None
        self._try_load()

    def _try_load(self):
        try:
            import spacy
            # Try to load the small English model
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("✅ NER ready (spaCy en_core_web_sm)")
            except OSError:
                logger.warning("⚠️  spaCy model not found.\n"
                               "   Install: python -m spacy download en_core_web_sm")
        except ImportError:
            logger.warning("⚠️  spaCy not installed — using regex NER fallback.\n"
                           "   Install: pip install spacy")

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Return a dict of entity_type → [entity_text, …].
        e.g. {"PERSON": ["Alice"], "ORG": ["Google"], "DATE": ["2024"]}
        """
        if self._nlp:
            return self._spacy_extract(text)
        return self._regex_fallback(text)

    def _spacy_extract(self, text: str) -> Dict[str, List[str]]:
        doc = self._nlp(text[:5000])          # cap at 5k chars for speed
        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, [])
            val = ent.text.strip()
            if val and val not in entities[ent.label_]:
                entities[ent.label_].append(val)
        return entities

    def _regex_fallback(self, text: str) -> Dict[str, List[str]]:
        """
        Very simple heuristic NER — capitalised runs of words are
        likely proper nouns (PERSON or ORG).  Years are DATEs.
        Not production-grade, but shows the concept clearly.
        """
        entities: Dict[str, List[str]] = {}

        # Dates — four-digit years and common date patterns
        dates = re.findall(r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        if dates: entities["DATE"] = list(set(dates))

        # Capitalised phrases (likely names / organisations)
        caps = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b", text)
        if caps: entities["PROPER_NOUN"] = list(set(caps[:10]))   # top 10

        # Percentage / money — useful for financial docs
        money = re.findall(r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%", text)
        if money: entities["MONEY/PERCENT"] = list(set(money[:10]))

        return entities

    def format_entities(self, entities: Dict[str, List[str]]) -> str:
        """Human-readable entity summary for display in the UI."""
        if not entities:
            return "No named entities detected."
        lines = []
        for label, items in sorted(entities.items()):
            lines.append(f"  {label}: {', '.join(items[:5])}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  SECTION 5 — KEYWORD EXTRACTOR
#  Show which terms are most informative (high TF-IDF weight).
#  Useful for the NLP report: demonstrates vocabulary analysis.
# ─────────────────────────────────────────────────────────────

class KeywordExtractor:
    """
    Extract the most informative keywords from a piece of text using
    TF-IDF scores relative to a background corpus (the full document).

    Academic note: This is related to RAKE, YAKE, and KeyBERT —
    progressively more powerful keyword extraction algorithms.
    """

    def extract(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return [(keyword, score), …] sorted by importance."""
        tokens = tokenize(text)
        if not tokens:
            return []
        tf = Counter(tokens)
        total = len(tokens)
        # Simple within-document TF * log(unique_terms / doc_frequency)
        unique = len(set(tokens))
        scored = []
        for word, count in tf.items():
            tf_score  = count / total
            # Treat unique-word-ratio as a stand-in for IDF
            idf_proxy = math.log(unique / count + 1)
            scored.append((word, tf_score * idf_proxy))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def format_keywords(self, keywords: List[Tuple[str, float]]) -> str:
        if not keywords:
            return "No keywords found."
        return ", ".join(f"{w} ({s:.3f})" for w, s in keywords)


# ─────────────────────────────────────────────────────────────
#  SECTION 6 — MAIN NLP ENGINE  (orchestrates all components)
#  This is what nlp-bot.py will call.
# ─────────────────────────────────────────────────────────────

class NLPEngine:
    """
    Unified NLP pipeline:
      1. Chunk the document from MCP
      2. Build FAISS (semantic) + TF-IDF (lexical) indexes
      3. On query: semantic top-k → TF-IDF re-rank → NER → keywords
      4. Return enriched context + metadata for the LLM
    """

    def __init__(self):
        self.semantic  = SemanticRetriever()
        self.tfidf     = TFIDFRanker()
        self.ner       = NERExtractor()
        self.keywords  = KeywordExtractor()
        self._chunks: List[str] = []
        self._indexed  = False

    def index_document(self, text: str,
                       chunk_size: int = 300,
                       overlap: int = 50) -> int:
        """
        Chunk → embed → index.
        Returns the number of chunks created.
        """
        self._chunks = chunk_text(text, chunk_size, overlap)
        if not self._chunks:
            logger.warning("No chunks created — document may be empty.")
            return 0

        logger.info(f"  Chunked into {len(self._chunks)} chunks "
                    f"(size={chunk_size}, overlap={overlap})")

        # Build both indexes so we can compare or combine them
        if self.semantic.available:
            self.semantic.build_index(self._chunks)

        self.tfidf.fit(self._chunks)
        self._indexed = True
        return len(self._chunks)

    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve the top-k most relevant chunks for the query.

        Strategy:
          • If FAISS available: semantic search → TF-IDF re-rank
          • Otherwise: TF-IDF only
        Returns a dict with: context, entities, keywords, method_used
        """
        if not self._indexed or not self._chunks:
            return {"context": "", "entities": {}, "keywords": [],
                    "method": "no_index", "chunks_used": 0}

        if self.semantic.available:
            # Dense retrieval: get top 2×k candidates
            candidates = self.semantic.search(query, top_k=top_k * 2)
            candidate_texts = [c for c, _ in candidates]

            # Lexical re-rank within candidates
            if candidate_texts:
                self.tfidf.fit(candidate_texts)   # refit on candidates only
                reranked = self.tfidf.rank(query, top_k=top_k)
                top_chunks = [c for c, _ in reranked]
            else:
                top_chunks = candidate_texts[:top_k]

            method = "semantic+tfidf"
        else:
            # Fallback: lexical only
            reranked  = self.tfidf.rank(query, top_k=top_k)
            top_chunks = [c for c, _ in reranked]
            method    = "tfidf_only"

        context = "\n\n".join(top_chunks)

        # NLP enrichments
        entities = self.ner.extract(context)
        kw_list  = self.keywords.extract(context, top_k=8)

        return {
            "context":     context,
            "entities":    entities,
            "keywords":    kw_list,
            "method":      method,
            "chunks_used": len(top_chunks)
        }

    def summarize(self, text: str, top_k: int = 5) -> str:
        """Extractive summary using TF-IDF sentence scores."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
        if not sentences: return "Nothing to summarise."
        if len(sentences) <= top_k: return " ".join(sentences)
        ranker = TFIDFRanker()
        ranker.fit(sentences)
        # Score each sentence by its average TF-IDF weight
        scored = []
        for i, sent in enumerate(sentences):
            toks = tokenize(sent)
            vec  = ranker._vec(toks)
            score = sum(vec.values()) / max(len(vec), 1)
            scored.append((score, i))
        scored.sort(reverse=True)
        top_idx = sorted(i for _, i in scored[:top_k])
        return " ".join(sentences[i] for i in top_idx)
