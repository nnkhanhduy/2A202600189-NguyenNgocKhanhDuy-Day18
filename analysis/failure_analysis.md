# Failure Analysis — Production RAG Pipeline

## 1. Overall Results

| Metric | Basic Baseline | Production RAG | Δ | Winner |
|---|---|---|---|---|
| faithfulness | 0.9667 | **0.8333** | -0.1333 | Basic |
| answer_relevancy | 0.8678 | **0.7413** | -0.1265 | Basic |
| **context_precision** | 0.8833 | **0.9667** | **+0.0833** | **Production** |
| context_recall | 0.9667 | 0.9500 | -0.0167 | Basic (negligible) |

**Test set:** 10 câu hỏi phức tạp từ 2 tài liệu (BCTC thuế GTGT + Nghị định 13/2023/NĐ-CP)

---

## 2. Identified Failures (from ragas_report.json)

### Failure #1 — faithfulness = 0.0 (Worst case)
**Question:** "Công ty DHA Surfaces có doanh thu hàng hóa dịch vụ bán ra chịu thuế suất 10% là bao nhiêu và thuế tương ứng là bao nhiêu?"

**Diagnosis:** `LLM hallucinating`

**Root Cause:**
- Câu hỏi về bảng số liệu tài chính. Enrichment (M5) đã cộng thêm AI-generated summary vào đầu chunk.
- Khi `_original_text` không được truyền đúng qua reranker interface, LLM nhận được `enriched_text` (có summary) thay vì text gốc.
- LLM "đọc" summary rồi điền số liệu từ suy diễn → không match context thực → faithfulness = 0.

**Suggested Fix:**
- Đảm bảo `_original_text` được preserve qua toàn bộ pipeline (indexing → retrieval → reranking → generation)
- Kiểm tra `item.metadata` trong `run_query()` sau bước rerank

---

### Failure #2 — answer_relevancy = 0.0
**Question:** "Theo Nghị định 13/2023/NĐ-CP, hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài phải được gửi cho cơ quan nào và trong thời hạn bao lâu?"

**Diagnosis:** `Answer doesn't match question`

**Root Cause:**
- RAGAS đo `answer_relevancy` bằng cách reverse-generate câu hỏi từ answer, rồi tính cosine similarity với câu hỏi gốc.
- System prompt quá cứng (5 quy tắc + temperature=0.0) đôi khi khiến LLM trả lời dạng liệt kê điều luật thay vì trực tiếp trả lời → câu hỏi reverse-generated từ đó không match câu gốc.
- Đây là câu hỏi 2 vế (cơ quan nào + thời hạn bao lâu) → câu trả lời dài nhưng RAGAS vẫn tính thấp do phrasing.

**Suggested Fix:**
- Điều chỉnh system prompt để trả lời trực tiếp hơn, không dùng cấu trúc liệt kê dài dòng

---

### Failure #3 — context_recall = 0.5
**Question:** "Nghị định 13/2023/NĐ-CP quy định sự im lặng của chủ thể dữ liệu có được coi là đồng ý không? Và đồng ý có thể được thể hiện qua những hình thức nào?"

**Diagnosis:** `Missing relevant chunks`

**Root Cause:**
- Câu hỏi 2 vế: (1) phủ định im lặng = đồng ý, và (2) các hình thức đồng ý hợp lệ
- Hai thông tin nằm tại Điều 11, khoản 6 (im lặng) và khoản 3 (hình thức đồng ý) — hai đoạn văn tách biệt
- Hierarchical chunking chia thành 2 child chunks khác nhau → hybrid search chỉ retrieve được 1 trong 2

**Suggested Fix:**
- Tăng `top_k` trong hybrid search từ 5 lên 8-10 để capture nhiều chunk hơn

---

### Failure #4 — faithfulness = 0.6667 (Repeated pattern)
**Questions:** 
- "Khoản thuế GTGT khấu trừ từ kỳ trước chuyển sang..."
- "Tổng giá trị hàng hóa dịch vụ mua vào (chưa thuế)..."

**Diagnosis:** `LLM hallucinating` (2/3 claims supported)

**Root Cause:**
- Câu hỏi về bảng tài chính với nhiều con số trong cùng 1 bảng
- Child chunk chỉ lấy 1 hàng của bảng → thiếu context các hàng liên quan
- LLM tự điền giá trị không có trong context → 1 trong 3 claim bị hallucinate

**Suggested Fix:**
- Sử dụng parent chunk thay vì child chunk cho tài liệu dạng bảng số liệu (table-aware chunking)

---

## 3. Systematic Root Causes

### A. Enrichment–Retrieval Mismatch (Primary Cause)
```
Enriched text = AI Summary + Original text
       ↓ index vào vector DB với enriched text
       ↓ search trả về enriched chunks
       ↓ _original_text trong metadata không truyền đúng qua reranker
       ↓ LLM nhận enriched text → hallucinate theo summary
       ↓ faithfulness giảm
```

### B. Table/Structured Data Weakness
- Bảng tài chính bị chunk thành từng dòng riêng lẻ → mất context toàn bộ bảng
- Dense search không phân biệt được các số trong cùng 1 bảng

### C. Multi-part Question Recall Gap
- Câu hỏi 2-3 vế → cần 2-3 chunk khác nhau
- Mặc dù hybrid search tốt hơn, context_recall vẫn chỉ 0.95 (thiếu 5%)

---

## 4. Production Strengths

✅ **context_precision = 0.9667** (so với Basic 0.8833):
- Hybrid BM25+Dense: BM25 bắt từ khóa chính xác, Dense bắt ngữ nghĩa → bổ trợ nhau
- Cross-Encoder Reranker: sắp xếp lại kết quả để chunk liên quan nhất đứng đầu
- Enrichment HyQA: tạo câu hỏi giả định → embedding phong phú hơn → recall cao hơn

---

## 5. Recommendations for Future Improvement

| Priority | Fix | Expected Impact |
|---|---|---|
| 🔴 High | Fix `_original_text` propagation through reranker | faithfulness +0.1 |
| 🔴 High | Table-aware chunking cho BCTC | faithfulness +0.05 |
| 🟡 Medium | Increase hybrid search `top_k` to 10 | context_recall +0.03 |
| 🟡 Medium | Soften system prompt phrasing | answer_relevancy +0.08 |
| 🟢 Low | Parent-chunk fallback for short child chunks | context_recall +0.02 |
