Rewrite the query for better retrieval coverage.

Rules:
- Keep the original intent.
- Use chat history to resolve follow-up references if needed.
- Return rewrites in `response_language`.
- If `response_language` is `vi`, write rewrites fully in Vietnamese.
- Do not return Chinese unless the user explicitly asks in Chinese.
- Return up to 3 rewrites, concise and retrieval-oriented.
- Avoid duplicates.

Output format:
Return strict JSON only:
`{"rewrites": ["query 1", "query 2", "query 3"]}`

Question: `$question`
Response language: `$response_language` (`$response_language_name`)
Chat history:
$chat_history
Loop count: `$loop_count`
Critique:
$critique
