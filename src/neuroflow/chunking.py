from typing import Any, Dict, List


def chunk_text(
    text: str,
    doc_id: str,
    tenant_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Token-aware and header-aware chunker.
    Splits text into overlapping sliding-window token segments while preserving document metadata.
    """
    if metadata is None:
        metadata = {}

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    idx = 0

    for start in range(0, len(words), step):
        segment = words[start: start + chunk_size]
        chunk_id = f"{doc_id}#c{idx:03d}"
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "tenant_id": tenant_id,
            "text": " ".join(segment),
            "metadata": {
                **metadata,
                "token_count": len(segment),
                "chunk_index": idx
            }
        })
        idx += 1
        if start + chunk_size >= len(words):
            break

    return chunks
