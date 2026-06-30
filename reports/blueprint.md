# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Trần Nhất Huy
**Mã học viên:** 2A202600731
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~3ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~400ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 1.5 | 3.2 | 5.0 | <10ms |
| NeMo Input Rail | 250.0 | 410.0 | 600.0 | <300ms |
| RAG Pipeline | 850.0 | 1200.0 | 1500.0 | <2000ms |
| NeMo Output Rail | 250.0 | 410.0 | 600.0 | <300ms |
| **Total Guard** | 501.5 | **823.2** | 1205.0 | **<500ms** |

**Budget OK?** [ ] Yes / [x] No  
**Comment:** Total Guard vượt budget do NeMo Input/Output Rail phụ thuộc vào tốc độ phản hồi của OpenAI API. Cần đổi sang mô hình open-source siêu nhỏ gọn lưu trữ trên RAM (Local Llama-3-8B hoặc SLM) và áp dụng Streaming để tối ưu.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.95 |
| Worst metric | None |
| Dominant failure distribution | N/A |
| Cohen's κ | 1.00 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 823.2 ms |

---

## Nhận xét & Cải tiến

> 1. RAGAS Pipeline hoạt động xuất sắc về mặt chức năng nhưng thư viện Pydantic V2 khá nhạy cảm gây khó khăn tương thích, phải xử lý fallback cẩn thận.
> 2. Việc tìm kiếm lai kết hợp (Hybrid Search) và Reranker đem đến kết quả tuyệt đối nhưng đánh đổi là thời gian tính toán của mô hình khá lâu, đặc biệt khi không có Caching.
> 3. Trong Production, tôi sẽ implement Redis Caching cho các Embeddings và chuyển đổi NeMo Guardrails chạy local model để tối ưu toàn diện phần Latency bị vượt quá (hiện đang ~800ms).
