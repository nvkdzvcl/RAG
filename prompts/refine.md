Refine the draft answer using critique and context.

Rules:
- Keep only grounded claims.
- Address missing aspects when evidence exists.
- Return in `response_language` only.
- If `response_language` is `vi`, answer fully in Vietnamese.
- Do not answer in Chinese unless the user explicitly asks in Chinese.
- Keep technical terms when useful, optionally with English in parentheses.

Output format:
Return strict JSON only:
`{"refined_answer": "string"}`

Question: `$question`
Response language: `$response_language` (`$response_language_name`)
Draft answer: `$draft_answer`
Critique:
$critique
Selected context:
$selected_context
