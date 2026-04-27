Decide whether retrieval is needed before answering.

Rules:
- If factual grounding is needed to answer accurately, return `need_retrieval=true`.
- For obvious small talk, greetings, or simple acknowledgements (e.g. "cảm ơn", "ok", "hi"), return `need_retrieval=false`.
- For follow-up questions that refer to previous conversation (e.g. "còn cái kia thì sao?"), return `need_retrieval=true` — the system needs fresh context.
- For confirmation or agreement (e.g. "đúng rồi", "ok hiểu rồi"), return `need_retrieval=false`.
- For meta-questions about the system itself (e.g. "bạn là ai?"), return `need_retrieval=false`.
- Keep reason short and in `response_language`.
- Do not answer in Chinese unless the user explicitly asks in Chinese.

Output format:
Return strict JSON only:
`{"need_retrieval": true, "reason": "short_reason"}`

Response language: `$response_language` (`$response_language_name`)
Question: `$question`
Chat history:
$chat_history
