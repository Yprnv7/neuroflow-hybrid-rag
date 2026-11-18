import time
from typing import Any, Dict, List

from fastapi import FastAPI, status
from pydantic import BaseModel, Field

from neuroflow.chunking import chunk_text
from neuroflow.dense_encoder import DenseEncoder
from neuroflow.fusion_rrf import reciprocal_rank_fusion
from neuroflow.reranker import CrossEncoderReranker
from neuroflow.semantic_cache import SemanticCache
from neuroflow.sparse_bm25 import BM25Index

app = FastAPI(
    title="NeuroFlow Hybrid RAG Engine",
    description="Enterprise Hybrid Retrieval, RRF Fusion, Cross-Encoder Reranking, and Tenant-Isolated Semantic Cache",
    version="0.2.0"
)

# Core singletons
bm25 = BM25Index()
encoder = DenseEncoder(fallback=True)
reranker = CrossEncoderReranker(fallback=True)
cache = SemanticCache(similarity_threshold=0.92)

all_chunks: List[Dict[str, Any]] = []
chunk_embeddings: List[Any] = []


class IngestDoc(BaseModel):
    doc_id: str
    tenant_id: str = "default"
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    top_k: int = 5
    enable_cache: bool = True
    enable_rerank: bool = True


class QueryResponse(BaseModel):
    query: str
    tenant_id: str
    cached: bool
    latency_ms: float
    timing_breakdown: Dict[str, float] = Field(default_factory=dict)
    results: List[Dict[str, Any]]
    answer_preview: str


@app.get("/health")
def health():
    return {"status": "healthy", "service": "neuroflow-hybrid-rag", "version": "0.2.0"}


@app.post("/v1/ingest", status_code=status.HTTP_201_CREATED)
def ingest_documents(docs: List[IngestDoc]):
    global all_chunks, chunk_embeddings
    new_chunks = []
    for d in docs:
        chunks = chunk_text(d.text, doc_id=d.doc_id, tenant_id=d.tenant_id, metadata=d.metadata)
        new_chunks.extend(chunks)

    all_chunks.extend(new_chunks)
    bm25.index(all_chunks)

    texts = [c["text"] for c in all_chunks]
    chunk_embeddings = encoder.encode(texts)
    return {"status": "success", "new_chunks_indexed": len(new_chunks), "total_chunks": len(all_chunks)}


@app.post("/v1/query", response_model=QueryResponse)
def query_pipeline(req: QueryRequest):
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    t_embed_start = time.perf_counter()
    q_vec = encoder.encode_one(req.query)
    timings["embed_ms"] = round((time.perf_counter() - t_embed_start) * 1000, 2)

    # 1. Tenant-isolated semantic cache check
    if req.enable_cache:
        t_cache_start = time.perf_counter()
        cached_result = cache.get(req.tenant_id, q_vec)
        timings["cache_ms"] = round((time.perf_counter() - t_cache_start) * 1000, 2)
        if cached_result is not None:
            cached_resp, sim = cached_result
            cached_resp["cached"] = True
            cached_resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            cached_resp["timing_breakdown"] = timings
            return cached_resp
    else:
        timings["cache_ms"] = 0.0

    # 2. Parallel BM25 + Dense retrieval
    t_bm25_start = time.perf_counter()
    bm25_hits = bm25.search(req.query, tenant_id=req.tenant_id, top_k=50)
    timings["bm25_ms"] = round((time.perf_counter() - t_bm25_start) * 1000, 2)

    # Dense vector cosine similarity
    t_dense_start = time.perf_counter()
    dense_hits = []
    if len(all_chunks) > 0 and len(chunk_embeddings) > 0:
        sims = chunk_embeddings @ q_vec
        indices = [i for i, c in enumerate(all_chunks) if c["tenant_id"] == req.tenant_id]
        if indices:
            sub_sims = sims[indices]
            sub_ranked = sorted(zip(indices, sub_sims), key=lambda x: x[1], reverse=True)[:50]
            dense_hits = [(all_chunks[idx], float(score)) for idx, score in sub_ranked]
    timings["dense_ms"] = round((time.perf_counter() - t_dense_start) * 1000, 2)

    # 3. Reciprocal Rank Fusion (RRF)
    t_fusion_start = time.perf_counter()
    fused_candidates = reciprocal_rank_fusion(bm25_hits, dense_hits, k=60)
    top_candidates = [c for c, _ in fused_candidates[:25]]
    timings["fusion_ms"] = round((time.perf_counter() - t_fusion_start) * 1000, 2)

    # 4. Cross-Encoder Reranking
    t_rerank_start = time.perf_counter()
    if req.enable_rerank and top_candidates:
        final_ranked = reranker.rerank(req.query, top_candidates, top_k=req.top_k)
        final_results = [{"chunk": c, "rerank_score": s} for c, s in final_ranked]
    else:
        final_results = [{"chunk": c, "rerank_score": None} for c in top_candidates[:req.top_k]]
    timings["rerank_ms"] = round((time.perf_counter() - t_rerank_start) * 1000, 2)

    # Generate citation summary
    citations = [f"[{res['chunk']['doc_id']}:{res['chunk']['chunk_id']}]" for res in final_results]
    preview = f"Grounded response for query '{req.query}' supported by sources: {' '.join(citations)}"

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    response_data = {
        "query": req.query,
        "tenant_id": req.tenant_id,
        "cached": False,
        "latency_ms": total_ms,
        "timing_breakdown": timings,
        "results": final_results,
        "answer_preview": preview
    }

    if req.enable_cache:
        cache.put(req.tenant_id, req.query, q_vec, response_data)

    return response_data


@app.delete("/v1/tenants/{tenant_id}/forget")
def gdpr_forget_tenant(tenant_id: str):
    """GDPR Article 17 Right to Erasure."""
    global all_chunks, chunk_embeddings
    removed_cache = cache.forget_tenant(tenant_id)
    all_chunks = [c for c in all_chunks if c["tenant_id"] != tenant_id]
    if all_chunks:
        bm25.index(all_chunks)
        chunk_embeddings = encoder.encode([c["text"] for c in all_chunks])
    else:
        bm25.index([])
        chunk_embeddings = []

    return {"status": "erased", "tenant_id": tenant_id, "cache_entries_removed": removed_cache}
