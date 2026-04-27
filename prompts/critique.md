Critique draft answer quality against selected context.

Rules:
- Evaluate groundedness and evidence sufficiency.
- Mark conflict if context has incompatible claims.
- Recommend retry only when retrieval can likely improve evidence.
- Keep notes short and actionable.
- Keep text fields (`note`, `missing_aspects`, `better_queries`) in `response_language`.
- Do not answer in Chinese unless the user explicitly asks in Chinese.
- Use chat history only to resolve follow-up references.

Output format:
Return strict JSON only with this schema:
{
  "grounded": true,
  "enough_evidence": true,
  "has_conflict": false,
  "missing_aspects": ["aspect"],
  "should_retry_retrieval": false,
  "should_refine_answer": false,
  "better_queries": ["query"],
  "confidence": 0.0,
  "note": "short note"
}

Example 1 (well-grounded):
Question: "Self-RAG là gì?"
Draft answer: "Self-RAG là phương pháp kết hợp retrieval và critique để tạo câu trả lời có căn cứ từ tài liệu."
Context: "Self-RAG uses retrieval, reranking, and critique loops to produce grounded answers."
→ {"grounded": true, "enough_evidence": true, "has_conflict": false, "missing_aspects": [], "should_retry_retrieval": false, "should_refine_answer": false, "better_queries": [], "confidence": 0.85, "note": "Answer is supported by context."}

Example 2 (weak evidence):
Question: "So sánh BM25 và dense retrieval"
Draft answer: "BM25 dùng keyword matching, dense retrieval dùng vector similarity. Dense tốt hơn cho câu hỏi ngữ nghĩa."
Context: "BM25 uses term frequency and inverse document frequency for ranking."
→ {"grounded": false, "enough_evidence": false, "has_conflict": false, "missing_aspects": ["dense retrieval definition", "comparison criteria"], "should_retry_retrieval": true, "should_refine_answer": true, "better_queries": ["dense retrieval là gì", "so sánh BM25 dense retrieval ưu nhược điểm"], "confidence": 0.3, "note": "Context only covers BM25, missing dense retrieval info."}

Question: `$question`
Response language: `$response_language` (`$response_language_name`)
Chat history:
$chat_history
Draft answer: `$draft_answer`
Selected context:
$selected_context
Loop: `$loop_count` / `$max_loops`
