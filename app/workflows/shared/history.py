"""Chat history trimming and rendering helpers."""

from __future__ import annotations


def trim_chat_history(
    chat_history: list[dict[str, str]] | None,
    *,
    memory_window: int,
) -> list[dict[str, str]]:
    """Return the latest bounded conversation window for prompt context."""
    if not chat_history:
        return []
    if memory_window <= 0:
        return []

    # Memory window is measured in turns; one turn roughly equals user + assistant.
    max_messages = max(1, memory_window * 2)
    normalized: list[dict[str, str]] = []
    for item in chat_history[-max_messages:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def build_chat_history_context(
    chat_history: list[dict[str, str]] | None,
    *,
    memory_window: int,
) -> str:
    """Render bounded chat history into concise prompt text."""
    window = trim_chat_history(chat_history, memory_window=memory_window)
    if not window:
        return "(empty)"

    lines: list[str] = []
    for idx, message in enumerate(window, start=1):
        role_label = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{idx}. {role_label}: {message['content']}")
    return "\n".join(lines)
