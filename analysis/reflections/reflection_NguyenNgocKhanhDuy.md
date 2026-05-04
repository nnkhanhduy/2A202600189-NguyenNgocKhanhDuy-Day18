# Individual Reflection — Production RAG Lab
## Day 18, Track 3

**Tên:** Nguyễn Ngọc Khánh Duy

**Ngày:** 2026-05-04

---

## 1. Điều tôi học được từ lab này

### 1.1 Retrieval ≠ Generation — hai bài toán khác nhau hoàn toàn

Trước lab này, tôi nghĩ RAG pipeline là một khối thống nhất: retrieve tốt thì answer tốt. Sau khi phân tích kết quả RAGAS, tôi nhận ra đây là **hai bài toán độc lập**:

- **Retrieval** (đo bằng `context_precision`, `context_recall`): Làm sao tìm được chunk đúng?
- **Generation** (đo bằng `faithfulness`, `answer_relevancy`): Làm sao tổng hợp thành answer đúng từ chunk đó?

Production pipeline của nhóm tôi **thắng về retrieval** (context_precision +0.0833) nhưng **thua về generation** (faithfulness -0.1333). Đây là bài học quan trọng: optimize retrieval không tự động improve generation.

### 1.2 Enrichment là con dao hai lưỡi

M5 Enrichment tạo ra `enriched_text` (AI summary + original) để cải thiện semantic search. Nhưng nếu vô tình đưa `enriched_text` vào LLM thay vì `original_text`, LLM sẽ "hallucinate theo summary" — tức là trust AI summary hơn là đọc thực sự.

**Insight:** Cần tách rõ ràng dữ liệu cho hai mục đích:
```
enriched_text  → indexing/search (recall tốt)
original_text  → LLM generation (faithfulness cao)
```

Đây là pattern quan trọng trong production system thực tế.

### 1.3 RAGAS metrics không phải "black box"

Ban đầu tôi chỉ nhìn số. Sau khi debug failures, tôi hiểu cơ chế từng metric:

- `faithfulness`: RAGAS tách answer thành claims → check từng claim có trong context không
- `answer_relevancy`: RAGAS reverse-generate questions từ answer → cosine similarity với query gốc
- `context_precision`: RAGAS check context nào thực sự được dùng để answer
- `context_recall`: RAGAS check ground truth có được cover bởi contexts không

Hiểu cơ chế → debug được vấn đề → cải thiện đúng chỗ.

---

## 2. Điều tôi đóng góp cho nhóm

- Phân tích tại sao Basic Baseline cao bất thường (trivially faithful khi `answer = contexts[0]`)
- Đề xuất thiết kế test set mới với 10 câu hỏi phức tạp (multi-fact, cross-reference, negation)
- Debug và fix pipeline để sử dụng `original_text` thay vì `enriched_text` cho LLM
- Phân tích failure cases từ `ragas_report.json` để tìm root cause

---

## 3. Điều tôi muốn cải thiện nếu có thêm thời gian

### 3.1 Fix bug `_original_text` propagation
Failure case lớn nhất là metadata không được preserve đúng qua reranker. Cần kiểm tra:
```python
# Trong run_query():
for item in top_items:
    meta = item.metadata if hasattr(item, "metadata") else {}
    original = meta.get("_original_text", "")
    # Debug: print(f"_original_text exists: {bool(original)}")
```

### 3.2 Table-aware chunking
Tài liệu BCTC có bảng số liệu tài chính. Chunking theo paragraph sẽ cắt ngang bảng → mất context. Cần implement:
- Detect bảng trong markdown (row bắt đầu bằng `|`)
- Giữ nguyên cả bảng như 1 chunk thay vì cắt từng dòng

### 3.3 Query decomposition
Câu hỏi 2 vế ("A là gì? Và B như thế nào?") → tách thành 2 sub-queries → merge kết quả:
```python
sub_queries = decompose_query(query)  # LLM tách câu hỏi
results = [search(q) for q in sub_queries]
merged = merge_and_dedup(results)
```

---

## 4. Suy nghĩ về Production RAG trong thực tế

Lab này cho tôi thấy rõ khoảng cách giữa **RAG đơn giản** và **Production RAG**:

| Basic RAG | Production RAG |
|---|---|
| Paragraph chunks | Hierarchical chunking |
| Dense-only search | Hybrid BM25 + Dense |
| No reranking | Cross-encoder reranker |
| Raw text cho LLM | Enriched index + Original text cho LLM |
| No prompt engineering | Strict multi-rule system prompt |

Điều thú vị nhất: thêm nhiều kỹ thuật phức tạp **không đảm bảo** điểm cao hơn trên mọi metric. Sự phức tạp tạo ra nhiều điểm failure hơn (bug propagation, parameter tuning, etc.). Production RAG thực sự đòi hỏi:
1. **Understand từng metric** — không chỉ nhìn con số tổng
2. **Debug từng component** — isolate retrieval vs generation
3. **Test với data thực tế** — toy dataset che giấu bugs thực sự

---

## 5. Câu hỏi còn chưa được giải đáp

1. **Optimal enrichment strategy:** Contextual summary vs HyQA vs Metadata — cái nào giúp nhiều nhất cho retrieval trên tài liệu pháp luật tiếng Việt?

2. **Cross-encoder vs Bi-encoder tradeoff:** Cross-encoder chậm hơn nhưng chính xác hơn. Với production system cần real-time response, khi nào nên trade accuracy cho speed?

3. **RAGAS reliability:** RAGAS tự dùng LLM để đánh giá. Nếu LLM judge bị bias, metrics có đáng tin không? Có cách nào validate RAGAS scores không?
