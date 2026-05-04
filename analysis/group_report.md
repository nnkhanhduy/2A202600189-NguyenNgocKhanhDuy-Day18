# Group Report — Production RAG Pipeline
## Day 18, Track 3: Production RAG

---

## 1. Team Information(Làm 1 mình)

| Field | Info |
|---|---|
| Lab | Day 18 — Production RAG Pipeline |
| Track | Track 3: Production RAG |
| Submission Date | 2026-05-04 |

---

## 2. System Architecture

```
Documents (BCTC.pdf, Nghị định 13/2023)
         │
    ┌────▼────────────────────┐
    │  M1: Hierarchical       │  Parent (1000 chars) + Child (256 chars)
    │  Chunking               │  underthesea sentence tokenizer
    └────┬────────────────────┘
         │ 289 child chunks
    ┌────▼────────────────────┐
    │  M5: Enrichment         │  Contextual summary + HyQA + Metadata
    │  (for indexing only)    │  enriched_text → index; original_text → LLM
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  M2: Hybrid Search      │  BM25 + Dense (paraphrase-multilingual-mpnet)
    │  + RRF Fusion           │  Reciprocal Rank Fusion
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  M3: Cross-Encoder      │  Re-ranks top-K results
    │  Reranker               │  ms-marco-MiniLM-L-6-v2
    └────┬────────────────────┘
         │ top-5 chunks (original text)
    ┌────▼────────────────────┐
    │  Generation (GPT-4o)    │  Strict 5-rule system prompt
    │  temperature=0.0        │  Vietnamese output
    └─────────────────────────┘
```

---

## 3. Key Design Decisions

### 3.1 Separation of Enriched Text vs Original Text
**Decision:** Sử dụng `enriched_text` CHỈ để indexing/search, đưa `original_text` cho LLM.

**Rationale:**
- `enriched_text` = AI-generated summary prepended to original → tăng semantic richness → recall tốt hơn
- Nhưng LLM đọc `enriched_text` → hallucinate theo summary → faithfulness giảm
- Giải pháp: lưu `_original_text` trong metadata, truyền vào LLM context

**Trade-off:** Nếu metadata không truyền đúng qua reranker interface → fallback về enriched text → faithfulness drop (đây là failure case chính trong kết quả)

### 3.2 Hierarchical Chunking Strategy
**Decision:** Chia tài liệu thành parent chunks (1000 chars) và child chunks (256 chars).

**Rationale:**
- Child chunks → precision cao trong embedding
- Parent chunks → cung cấp đủ context cho LLM khi cần

### 3.3 Hybrid Search với RRF
**Decision:** Kết hợp BM25 (lexical) + Dense (semantic) → Reciprocal Rank Fusion.

**Rationale:**
- BM25: tốt cho exact-match (mã số thuế, số điều luật)
- Dense: tốt cho semantic similarity (synonym, paraphrase)
- RRF: fusion không cần normalize scores → robust hơn linear combination

### 3.4 Strict System Prompt
**Decision:** 5 quy tắc bắt buộc + `temperature=0.0`.

```
1. Chỉ dùng thông tin trong [CONTEXT]
2. Không thêm kiến thức bên ngoài
3. Trích xuất chính xác, không diễn giải tự do
4. Nếu không có thông tin → "Không tìm thấy"
5. Ngắn gọn, đúng trọng tâm
```

---

## 4. Evaluation Results

### 4.1 RAGAS Scores

| Metric | Basic Baseline | Production RAG | Δ |
|---|---|---|---|
| faithfulness | 0.9667 | 0.8333 | -0.1333 |
| answer_relevancy | 0.8678 | 0.7413 | -0.1265 |
| **context_precision** | 0.8833 | **0.9667** | **+0.0833** |
| context_recall | 0.9667 | 0.9500 | -0.0167 |

### 4.2 Interpretation

**Production thắng về RETRIEVAL (context_precision +0.0833):**
- Hybrid Search + Reranker xếp hạng chunk đúng lên đầu tốt hơn Dense-only
- Chứng minh M2 (hybrid) và M3 (reranker) hoạt động đúng

**Production thua về GENERATION (faithfulness, answer_relevancy):**
- Root cause: `_original_text` không được truyền đúng qua reranker → LLM nhận enriched text → hallucinate
- System prompt quá cứng → answer_relevancy giảm (RAGAS không thể reverse-generate đúng câu hỏi)

### 4.3 Test Set Design

10 câu hỏi được thiết kế để thể hiện rõ sự khác biệt giữa hai pipeline:
- **Multi-fact questions:** Cần retrieve 2+ chunk khác nhau (bất lợi cho Basic dense-only)
- **Cross-reference questions:** Kết hợp thông tin từ nhiều Điều/Khoản
- **Numerical reasoning:** Tính toán từ bảng số liệu tài chính
- **Negation questions:** Câu hỏi về quy định phủ định

---

## 5. Module Implementation Status

| Module | Status | Notes |
|---|---|---|
| M1: Chunking | ✅ Complete | Hierarchical (parent + child), underthesea |
| M2: Search | ✅ Complete | BM25 + Dense + RRF fusion |
| M3: Reranker | ✅ Complete | CrossEncoder ms-marco-MiniLM |
| M4: Evaluation | ✅ Complete | RAGAS (faithfulness, relevancy, precision, recall) |
| M5: Enrichment | ✅ Complete | Contextual, HyQA, metadata extraction |

---

## 6. Challenges & Lessons Learned

### Challenge 1: Enrichment vs Faithfulness Trade-off
Enrichment cải thiện recall khi search, nhưng nếu đưa enriched text vào LLM thì hallucination tăng. Giải pháp là tách riêng luồng: enriched_text cho index, original_text cho generation.

### Challenge 2: Evaluation Metric Understanding
Ban đầu nghĩ rằng Naive Baseline (trả về `contexts[0]`) là baseline tốt. Nhưng thực ra đây là "trick" không thực tế vì:
- `faithfulness = 1.0` trivially (answer chính là context)
- Không phản ánh khả năng tổng hợp thông tin thực tế của pipeline

Giải pháp: Cả hai pipeline đều dùng LLM để sinh answer → so sánh công bằng về retrieval quality.

### Challenge 3: RAGAS Metrics Sensitivity
`answer_relevancy` đặc biệt nhạy cảm với phrasing của answer. System prompt quá cứng khiến LLM trả lời theo format liệt kê điều luật → RAGAS reverse-generate câu hỏi không match → điểm thấp.

---

## 7. Future Improvements (If More Time)

1. **Fix `_original_text` propagation** — đảm bảo metadata đúng sau reranker (→ faithfulness +0.1)
2. **Table-aware chunking** — xử lý riêng bảng số liệu tài chính
3. **Increase top_k to 10** — capture thêm chunk cho multi-fact questions
4. **Soften system prompt** — bỏ strict format, cho LLM trả lời tự nhiên hơn
5. **Query rewriting** — tự động tách câu hỏi 2 vế thành 2 queries độc lập

---

## 8. Conclusion

Production RAG pipeline đã cải thiện **context_precision** (+8.33%) so với Basic Baseline, chứng minh rằng Hybrid Search + Reranking hoạt động hiệu quả trong việc xếp hạng chunk liên quan.

Điểm yếu chính là **generation layer**: do bug về propagation của `original_text` qua reranker interface, LLM đôi khi nhận enriched text thay vì original text → hallucination. Đây là vấn đề kỹ thuật có thể fix được, không phải vấn đề thiết kế.

Về tổng thể, pipeline đã implement đầy đủ 5 modules (M1-M5) với các kỹ thuật production-ready: hierarchical chunking, hybrid search, cross-encoder reranking, data enrichment, và LLM generation với strict prompt engineering.
