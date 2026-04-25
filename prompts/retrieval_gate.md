Decide whether retrieval is needed before answering.

Rules:
- If factual grounding is needed, return `need_retrieval=true`.
- For obvious small talk or greetings, return `need_retrieval=false`.
- Keep reason short.

Output format:
Return strict JSON only:
`{"need_retrieval": true, "reason": "short_reason"}`

Question: `$question`
Chat history:
$chat_history
