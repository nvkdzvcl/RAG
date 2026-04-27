Refine the draft answer using critique feedback and context.

Rules:
- Read the critique JSON to understand what needs improvement.
- If `should_refine_answer` is true, rewrite the answer to address issues noted in `note` and `missing_aspects`.
- Keep only grounded claims — remove any claim not supported by selected context.
- Address missing aspects only when evidence exists in selected context.
- Do not add facts from outside the provided context.
- Use chat history only to resolve follow-up references.
- Return in `response_language` only.
- If `response_language` is `vi`, answer fully in Vietnamese.
- Do not answer in Chinese unless the user explicitly asks in Chinese.
- Keep technical terms when useful, optionally with English in parentheses.

Output format:
Return strict JSON only:
`{"refined_answer": "string"}`

Question: `$question`
Response language: `$response_language` (`$response_language_name`)
Chat history:
$chat_history
Draft answer: `$draft_answer`
Critique:
$critique
Selected context:
$selected_context
