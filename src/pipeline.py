"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4.

KEY FIXES:
  1. Enriched text dùng CHỈ để INDEX (tăng recall khi search).
  2. Original text (trước enrichment) đưa cho LLM → faithfulness cao hơn.
  3. System prompt cứng 5 quy tắc + temperature=0.0 → giảm hallucination.
"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # ── Step 1: Load & Chunk (M1) ──────────────────────────────────────────────
    print("\n[1/4] Chunking documents (hierarchical)...")
    docs = load_documents()
    all_children_raw: list[dict] = []

    for doc in docs:
        _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_children_raw.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id or ""},
            })

    print(f"  {len(all_children_raw)} child chunks from {len(docs)} docs")

    # ── Step 2: Enrichment (M5) ────────────────────────────────────────────────
    # Enriched text → dùng để INDEX (semantic tốt hơn, tăng recall)
    # Original text  → lưu vào metadata["_original_text"], đưa cho LLM sau
    print("\n[2/4] Enriching child chunks for indexing (M5)...")
    enriched = enrich_chunks(all_children_raw, methods=["contextual", "hyqa", "metadata"])

    if enriched:
        index_chunks: list[dict] = []
        for e in enriched:
            meta = {**e.auto_metadata}
            meta["_original_text"] = e.original_text   # ← LLM sẽ đọc cái này
            index_chunks.append({
                "text": e.enriched_text,               # ← Index/search dùng cái này
                "metadata": meta,
            })
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  ⚠️  M5 not implemented — using raw child chunks (fallback)")
        index_chunks = all_children_raw

    # ── Step 3: Index (M2) ────────────────────────────────────────────────────
    print("\n[3/4] Indexing with Hybrid Search (BM25 + Dense)...")
    search = HybridSearch()
    search.index(index_chunks)

    # ── Step 4: Reranker (M3) ─────────────────────────────────────────────────
    print("\n[4/4] Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""

    # ── Retrieve (Hybrid BM25 + Dense + RRF) ──────────────────────────────────
    results = search.search(query)
    docs_for_rerank = [
        {"text": r.text, "score": r.score, "metadata": r.metadata}
        for r in results
    ]

    # ── Rerank (Cross-encoder) ─────────────────────────────────────────────────
    reranked = reranker.rerank(query, docs_for_rerank, top_k=RERANK_TOP_K)
    top_items = reranked if reranked else results[:RERANK_TOP_K]

    # ── Build contexts: dùng original_text (trước enrichment) cho LLM ─────────
    # Lý do: enriched_text có câu AI-generated prepend → LLM "bịa" theo → faithfulness thấp
    # Original_text = tài liệu gốc → LLM trả lời trung thực hơn → faithfulness cao
    contexts: list[str] = []
    for item in top_items:
        meta = item.metadata if hasattr(item, "metadata") else {}
        original = meta.get("_original_text", "")
        contexts.append(original if original else item.text)

    if not contexts:
        contexts = [r.text for r in results[:RERANK_TOP_K]]

    # ── Generate answer (prompt cứng → faithfulness cao) ──────────────────────
    try:
        from openai import OpenAI
        client = OpenAI()
        context_str = "\n\n---\n\n".join(contexts)

        system_prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi CHỈ DỰA VÀO tài liệu được cung cấp.\n"
            "NGUYÊN TẮC BẮT BUỘC:\n"
            "1. Chỉ dùng thông tin có trong [CONTEXT]. Không thêm kiến thức bên ngoài.\n"
            "2. Trích xuất thông tin chính xác từ context. Không diễn giải tự do.\n"
            "3. Nếu context không có thông tin: trả lời 'Không tìm thấy thông tin trong tài liệu.'\n"
            "4. Câu trả lời ngắn gọn, đúng trọng tâm, bằng tiếng Việt."
        )

        user_prompt = (
            f"[CONTEXT]\n{context_str}\n\n"
            f"[CÂU HỎI]\n{query}\n\n"
            "[TRẢ LỜI] (chỉ dựa vào context trên, không thêm thông tin ngoài):"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,   # deterministic → giảm hallucination tối đa
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM error: {e}")
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."

    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries on test set...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:60]}...")

    print("\n[Eval] Running RAGAS evaluation...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
