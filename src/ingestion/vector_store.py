# ══════════════════════════════════════════════════════════
#  src/ingestion/vector_store.py
#
#  WHAT THIS FILE DOES:
#  Manages our Vector Database — the "Vector DB" box in
#  the architecture diagram.
#
#  NEW: Hybrid Search = Semantic + BM25 keyword search
#  combined via Reciprocal Rank Fusion (RRF)
#
#  WHY HYBRID?
#  - Semantic alone: great at meaning, misses exact terms
#  - BM25 alone: great at exact words, misses synonyms
#  - Hybrid: best of both worlds
#
#  ARCHITECTURE:
#  Query → [ChromaDB semantic + BM25 keyword] → RRF → Top results
# ══════════════════════════════════════════════════════════

import os
import math
from pathlib import Path
from dotenv import load_dotenv

# LangChain + ChromaDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

# BM25 keyword search
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Configuration ──────────────────────────────────────────
VECTOR_DB_PATH  = Path(os.getenv("VECTOR_DB_PATH", "data/vectors/chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
COLLECTION_NAME = "sec_10k_filings"

VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  EMBEDDING FUNCTION
# ══════════════════════════════════════════════════════════

def get_embeddings() -> OllamaEmbeddings:
    """
    Ollama embedding function.
    Converts text → 768-dimensional vectors using nomic-embed-text.
    Free, local, runs on your M2 chip.
    """
    return OllamaEmbeddings(
        model    = EMBEDDING_MODEL,
        base_url = OLLAMA_BASE_URL,
    )


# ══════════════════════════════════════════════════════════
#  VECTOR STORE
# ══════════════════════════════════════════════════════════

def get_vectorstore() -> Chroma:
    """Connect to (or create) the local ChromaDB vector store."""
    return Chroma(
        collection_name    = COLLECTION_NAME,
        embedding_function = get_embeddings(),
        persist_directory  = str(VECTOR_DB_PATH),
    )


def add_documents(documents: list[Document], batch_size: int = 50) -> int:
    """
    Embed and store LangChain Documents in ChromaDB.
    Skips documents already stored (safe to re-run).
    """
    if not documents:
        print("  No documents to add")
        return 0

    vectorstore = get_vectorstore()

    try:
        existing_ids = set(vectorstore.get()["ids"])
    except Exception:
        existing_ids = set()

    new_docs = [
        doc for doc in documents
        if doc.metadata.get("chunk_id") not in existing_ids
    ]

    if not new_docs:
        print("  All documents already in vector store — skipping")
        return 0

    print(f"  Adding {len(new_docs)} new documents...")
    total_added   = 0
    total_batches = (len(new_docs) - 1) // batch_size + 1

    for batch_num, i in enumerate(range(0, len(new_docs), batch_size), 1):
        batch = new_docs[i: i + batch_size]
        ids   = [doc.metadata["chunk_id"] for doc in batch]
        vectorstore.add_documents(documents=batch, ids=ids)
        total_added += len(batch)
        print(f"  Batch {batch_num}/{total_batches} — {total_added}/{len(new_docs)}")

    print(f"  Added {total_added} documents to ChromaDB")
    return total_added


# ══════════════════════════════════════════════════════════
#  RECIPROCAL RANK FUSION
#
#  RRF merges two ranked lists into one combined ranking.
#
#  Formula: score(doc) = 1 / (rank + k)
#  where k=60 is a smoothing constant (standard value)
#
#  Example:
#    Doc A: rank 1 in semantic, rank 3 in BM25
#           score = 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = 0.0323
#
#    Doc B: rank 2 in semantic, rank 1 in BM25
#           score = 1/(2+60) + 1/(1+60) = 0.0161 + 0.0164 = 0.0325
#
#    Doc B wins — it ranked highly in BOTH searches
# ══════════════════════════════════════════════════════════

def _reciprocal_rank_fusion(
    semantic_docs: list[Document],
    keyword_docs:  list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Merge semantic and keyword results using Reciprocal Rank Fusion.

    Args:
        semantic_docs: Documents from ChromaDB, ordered by similarity
        keyword_docs:  Documents from BM25, ordered by keyword score
        k:             Smoothing constant (60 is standard)

    Returns:
        Merged list of unique Documents, ordered by combined RRF score
    """
    # Build a score dictionary keyed by chunk_id
    # Each doc gets a score contribution from each list it appears in
    scores  = {}   # chunk_id → combined RRF score
    doc_map = {}   # chunk_id → Document object

    # Score from semantic search results
    for rank, doc in enumerate(semantic_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[chunk_id]  = scores.get(chunk_id, 0) + 1 / (rank + k)
        doc_map[chunk_id] = doc

    # Score from BM25 keyword results
    for rank, doc in enumerate(keyword_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[chunk_id]  = scores.get(chunk_id, 0) + 1 / (rank + k)
        doc_map[chunk_id] = doc

    # Sort by combined score — highest first
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_map[chunk_id] for chunk_id in ranked_ids]


# ══════════════════════════════════════════════════════════
#  BM25 KEYWORD SEARCH
#
#  BM25 (Best Match 25) is the gold standard keyword search
#  algorithm. It's what Google and Elasticsearch use.
#
#  It scores documents based on:
#  - Term frequency (how often the word appears)
#  - Inverse document frequency (how rare the word is)
#  - Document length normalization
#
#  We build a BM25 index from ALL documents in ChromaDB
#  on every search call. This is fine for our size (~1000 docs).
# ══════════════════════════════════════════════════════════

def _bm25_search(
    query:          str,
    all_docs:       list[Document],
    k:              int = 10,
    company_filter: str = None,
    year_filter:    int = None,
) -> list[Document]:
    """
    BM25 keyword search over a list of Documents.

    Process:
    1. Apply metadata filters (company, year) to narrow candidates
    2. Tokenize all document texts → build BM25 index
    3. Tokenize query → score all documents
    4. Return top k documents by BM25 score

    Args:
        query:          The search query
        all_docs:       All documents to search through
        k:              Number of results to return
        company_filter: Only search this company's documents
        year_filter:    Only search this year's documents

    Returns:
        Top k Documents ranked by BM25 score
    """
    # Step 1: Apply metadata filters
    candidates = []
    for doc in all_docs:
        meta = doc.metadata
        if company_filter and meta.get("company") != company_filter:
            continue
        if year_filter and meta.get("year") != year_filter:
            continue
        candidates.append(doc)

    if not candidates:
        return []

    # Step 2: Tokenize documents
    # Simple whitespace tokenization + lowercase
    # In production you'd use NLTK for stemming/stopwords
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in candidates
    ]

    # Step 3: Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)

    # Step 4: Score query against all documents
    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)

    # Step 5: Get top k indices by score
    # argsort gives ascending order → reverse for descending
    top_indices = sorted(
        range(len(scores)),
        key     = lambda i: scores[i],
        reverse = True
    )[:k]

    # Filter out zero-score results (no keyword match at all)
    return [
        candidates[i] for i in top_indices
        if scores[i] > 0
    ]


# ══════════════════════════════════════════════════════════
#  HYBRID SEARCH  ← main function used by rag_chain.py
#
#  This is the "Vector DB" box in our architecture:
#
#  Query
#    ↓
#  ┌──────────────┬───────────────┐
#  │              │               │
#  Semantic       BM25            │
#  Search         Keyword         │
#  (ChromaDB)     Search          │
#  │              │               │
#  └──────┬───────┘               │
#         ↓                       │
#    RRF Fusion                   │
#         ↓                       │
#    Top Results ─────────────────┘
# ══════════════════════════════════════════════════════════

def hybrid_search(
    query:          str,
    k:              int = 5,
    company_filter: str = None,
    year_filter:    int = None,
    section_filter: str = None,
) -> list[Document]:
    """
    Hybrid search: ChromaDB semantic + BM25 keyword → RRF fusion.

    This replaces the old similarity_search() function.
    It returns better results by combining two complementary
    search methods.

    Args:
        query:          The user's question
        k:              Number of final results to return
        company_filter: Restrict to one company's filings
        year_filter:    Restrict to one year's filings
        section_filter: Restrict to one 10-K section

    Returns:
        Top k Documents ranked by hybrid RRF score
    """
    vectorstore = get_vectorstore()

    # ── Step 1: Get ALL stored documents for BM25 ─────────
    # We need all docs to build the BM25 index
    try:
        stored      = vectorstore.get()
        all_docs    = [
            Document(
                page_content = stored["documents"][i],
                metadata     = stored["metadatas"][i] or {},
            )
            for i in range(len(stored["ids"]))
        ]
    except Exception as e:
        print(f"  Could not load documents for BM25: {e}")
        all_docs = []

    # ── Step 2: Semantic search via ChromaDB ───────────────
    where = {}
    if company_filter:
        where["company"] = {"$eq": company_filter}
    if year_filter:
        where["year"]    = {"$eq": year_filter}
    if section_filter:
        where["section"] = {"$eq": section_filter}

    try:
        if where:
            semantic_results = vectorstore.similarity_search(
                query, k=k*2, filter=where
            )
        else:
            semantic_results = vectorstore.similarity_search(
                query, k=k*2
            )
    except Exception as e:
        print(f"  Semantic search failed: {e}")
        semantic_results = []

    # ── Step 3: BM25 keyword search ────────────────────────
    keyword_results = _bm25_search(
        query          = query,
        all_docs       = all_docs,
        k              = k * 2,
        company_filter = company_filter,
        year_filter    = year_filter,
    )

    # ── Step 4: Reciprocal Rank Fusion ─────────────────────
    if not semantic_results and not keyword_results:
        return []

    if not semantic_results:
        return keyword_results[:k]

    if not keyword_results:
        return semantic_results[:k]

    # Merge both result lists using RRF
    fused = _reciprocal_rank_fusion(semantic_results, keyword_results)

    print(
        f"  Hybrid search: {len(semantic_results)} semantic + "
        f"{len(keyword_results)} keyword → "
        f"{len(fused)} fused → top {k} returned"
    )

    # Filter out XBRL/numeric garbage chunks before returning
    # These contain raw SEC XBRL tags and pollute RAG results
    def is_useful_chunk(doc):
        text = doc.page_content.strip()
        # Skip chunks that are mostly XBRL tags or numeric codes
        xbrl_indicators = [
            "0000320193", "0001318605", "0000789019",
            "us-gaap:", "aapl:", "msft:", "tsla:",
            "Member 20", "Member 2021", "Member 2022",
        ]
        xbrl_count = sum(1 for x in xbrl_indicators if x in text)
        if xbrl_count >= 2:
            return False
        # Skip chunks shorter than 100 chars
        if len(text) < 100:
            return False
        return True

    filtered = [doc for doc in fused if is_useful_chunk(doc)]
    print(f"  After filtering garbage: {len(filtered)} useful chunks")
    return filtered[:k]


# ── Keep old function name as alias for compatibility ──────
def similarity_search(query, k=5, company_filter=None,
                      year_filter=None, section_filter=None):
    """
    Backwards-compatible wrapper — now calls hybrid_search().
    Any code that called similarity_search() automatically
    gets hybrid search without any other changes needed.
    """
    return hybrid_search(query, k, company_filter, year_filter, section_filter)


# ══════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════

def get_store_stats() -> dict:
    """Return statistics about what's stored in ChromaDB."""
    try:
        vectorstore = get_vectorstore()
        data        = vectorstore.get()
        total       = len(data["ids"])
        metadatas   = data.get("metadatas", [])

        companies = set(m.get("company") for m in metadatas if m)
        years     = set(m.get("year")    for m in metadatas if m)
        sections  = set(m.get("section") for m in metadatas if m)

        return {
            "total_chunks": total,
            "companies":    sorted(c for c in companies if c),
            "years":        sorted(y for y in years if y),
            "sections":     sorted(s for s in sections if s),
        }
    except Exception:
        return {
            "total_chunks": 0,
            "companies":    [],
            "years":        [],
            "sections":     [],
        }


# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    stats = get_store_stats()
    print(f"Vector store: {stats['total_chunks']} chunks")
    print(f"Companies: {stats['companies']}")
    print(f"Years: {stats['years']}")

    if stats["total_chunks"] > 0:
        print("\nTesting hybrid search...")
        results = hybrid_search("revenue growth risk factors", k=3)
        for i, doc in enumerate(results, 1):
            m = doc.metadata
            print(f"\n  Result {i}: {m.get('company')} {m.get('year')} — {m.get('section')}")
            print(f"  Preview: {doc.page_content[:100]}...")
