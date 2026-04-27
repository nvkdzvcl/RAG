Rewrite the draft answer strictly based on selected context.

Rules:
- ONLY use facts present in selected context.
- Every claim must be supported by selected context.
- Do NOT use external knowledge.
- Use chat history only to resolve follow-up references.
- If the answer is not supported by context, return the insufficient-evidence message in the appropriate `response_language`:
  - Vietnamese: "Không đủ thông tin từ tài liệu để trả lời"
  - English: "Insufficient evidence to provide a grounded answer."
- Return in `response_language` only.
- If `response_language` is `vi`, answer fully in Vietnamese.
- Do not answer in Chinese unless the user explicitly asks in Chinese.

Output format:
Return strict JSON only:
`{"refined_answer": "string"}`

Question: `$question`
Response language: `$response_language` (`$response_language_name`)
Chat history:
$chat_history
Draft answer: `$draft_answer`
Selected context:
$selected_context
