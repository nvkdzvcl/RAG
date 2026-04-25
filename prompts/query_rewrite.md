Rewrite the query for better retrieval coverage.

Rules:
- Keep the original intent.
- Keep language compatible with original query (Vietnamese-compatible).
- Return up to 3 rewrites, concise and retrieval-oriented.
- Avoid duplicates.

Output format:
Return strict JSON only:
`{"rewrites": ["query 1", "query 2", "query 3"]}`

Question: `$question`
Loop count: `$loop_count`
Critique:
$critique
