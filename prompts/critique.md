Critique draft answer quality against selected context.

Rules:
- Evaluate groundedness and evidence sufficiency.
- Mark conflict if context has incompatible claims.
- Recommend retry only when retrieval can likely improve evidence.
- Keep notes short and actionable.
- Keep text fields (`note`, `missing_aspects`, `better_queries`) in `response_language`.
- Do not answer in Chinese unless the user explicitly asks in Chinese.

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

Question: `$question`
Response language: `$response_language` (`$response_language_name`)
Draft answer: `$draft_answer`
Selected context:
$selected_context
Loop: `$loop_count` / `$max_loops`
