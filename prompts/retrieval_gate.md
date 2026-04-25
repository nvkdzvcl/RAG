Decide whether retrieval is needed before answering.

Rules:
- If factual grounding is needed, return `need_retrieval=true`.
- For obvious small talk or greetings, return `need_retrieval=false`.
- Keep reason short and in `response_language`.
- Do not answer in Chinese unless the user explicitly asks in Chinese.

Output format:
Return strict JSON only:
`{"need_retrieval": true, "reason": "short_reason"}`

Response language: `$response_language` (`$response_language_name`)
Question: `$question`
Chat history:
$chat_history
