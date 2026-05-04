"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # 1. Split text into sentences:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast
    embeddings = model.encode(sentences)

    # 3. Compare consecutive sentences:
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b): return dot(a, b) / (norm(a) * norm(b))

    # 4. Group sentences:
    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))
            current_group = []
        current_group.append(sentences[i])
        
    # Don't forget last group
    if current_group:
        chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))

    # 5. Return chunks with metadata
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents_list = []
    children_list = []

    # 1. Split text into parents:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    current_parent_text = ""
    p_index = 0
    
    def process_parent(parent_text: str, pid: str):
        # 2. Split each parent into children:
        # Simple window approach by words
        words = parent_text.split()
        child_text = []
        child_len = 0
        for w in words:
            if child_len + len(w) + 1 > child_size and child_text:
                children_list.append(Chunk(text=" ".join(child_text), metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
                child_text = []
                child_len = 0
            child_text.append(w)
            child_len += len(w) + 1
        if child_text:
            children_list.append(Chunk(text=" ".join(child_text), metadata={**metadata, "chunk_type": "child"}, parent_id=pid))

    for para in paragraphs:
        if len(current_parent_text) + len(para) > parent_size and current_parent_text:
            pid = f"parent_{p_index}"
            parent = Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
            parents_list.append(parent)
            process_parent(current_parent_text.strip(), pid)
            
            current_parent_text = ""
            p_index += 1
        current_parent_text += para + "\n\n"
        
    if current_parent_text.strip():
        pid = f"parent_{p_index}"
        parent = Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
        parents_list.append(parent)
        process_parent(current_parent_text.strip(), pid)

    # 3. Return (parents_list, children_list)
    return parents_list, children_list


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # 1. Split by markdown headers:
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    # 2. Pair headers with their content:
    chunks = []
    current_header = ""
    current_content = ""
    
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
            
    # Don't forget last section
    if current_content.strip() or current_header.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
        ))

    # 3. Return chunks — mỗi chunk = 1 section hoàn chỉnh
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {"basic": {}, "semantic": {}, "hierarchical": {}, "structure": {}}
    stats = {k: {"num_chunks": 0, "lengths": []} for k in results.keys()}
    stats["hierarchical"]["parent_chunks"] = 0
    stats["hierarchical"]["child_chunks"] = 0

    # 1. For each doc, run all strategies
    for doc in documents:
        text = doc["text"]
        
        c_basic = chunk_basic(text)
        stats["basic"]["num_chunks"] += len(c_basic)
        stats["basic"]["lengths"].extend([len(c.text) for c in c_basic])
        
        c_sem = chunk_semantic(text)
        stats["semantic"]["num_chunks"] += len(c_sem)
        stats["semantic"]["lengths"].extend([len(c.text) for c in c_sem])
        
        p_hier, c_hier = chunk_hierarchical(text)
        stats["hierarchical"]["parent_chunks"] += len(p_hier)
        stats["hierarchical"]["child_chunks"] += len(c_hier)
        stats["hierarchical"]["lengths"].extend([len(c.text) for c in c_hier])
        
        c_struct = chunk_structure_aware(text)
        stats["structure"]["num_chunks"] += len(c_struct)
        stats["structure"]["lengths"].extend([len(c.text) for c in c_struct])

    # 2. Collect stats and Print comparison table
    print(f"{'Strategy':<14} | {'Chunks':<10} | {'Avg Len':<7} | {'Min':<5} | {'Max':<5}")
    print("-" * 50)
    for k in stats:
        lengths = stats[k]["lengths"]
        avg_l = sum(lengths) // len(lengths) if lengths else 0
        min_l = min(lengths) if lengths else 0
        max_l = max(lengths) if lengths else 0
        
        if k == "hierarchical":
            chunks_str = f"{stats[k]['parent_chunks']}p/{stats[k]['child_chunks']}c"
            results[k] = {"parents": stats[k]["parent_chunks"], "children": stats[k]["child_chunks"], "avg_len": avg_l, "min_len": min_l, "max_len": max_l}
        else:
            chunks_str = str(stats[k]["num_chunks"])
            results[k] = {"chunks": stats[k]["num_chunks"], "avg_len": avg_l, "min_len": min_l, "max_len": max_l}
            
        print(f"{k:<14} | {chunks_str:<10} | {avg_l:<7} | {min_l:<5} | {max_l:<5}")

    # 4. Return results dict
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
