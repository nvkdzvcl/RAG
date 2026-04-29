import type { Mode, QueryResult } from "@/types/chat";
import type { ChatSession, ChatSessionMessage } from "@/types/chat-session";

const CHAT_SESSIONS_STORAGE_KEY = "smartdocai.chat_sessions.v1";

const DEFAULT_SESSION_TITLE = "Cuộc trò chuyện mới";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nowIso(): string {
  return new Date().toISOString();
}

export function createChatSession({ mode, model }: { mode: Mode; model: string }): ChatSession {
  const now = nowIso();
  return {
    id: `chat-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: DEFAULT_SESSION_TITLE,
    createdAt: now,
    updatedAt: now,
    pinned: false,
    mode,
    model,
    messages: [],
    lastResultSummary: null,
    lastResult: null,
    lastSubmittedQuery: null,
  };
}

export function deriveSessionTitle(messages: ChatSessionMessage[]): string {
  const firstUserMessage = messages.find((item) => item.role === "user" && item.content.trim().length > 0);
  if (!firstUserMessage) {
    return DEFAULT_SESSION_TITLE;
  }
  const compact = firstUserMessage.content.replace(/\s+/g, " ").trim();
  return compact.length > 72 ? `${compact.slice(0, 69)}...` : compact;
}

function normalizeMode(value: unknown): Mode {
  if (value === "standard" || value === "advanced" || value === "compare") {
    return value;
  }
  return "standard";
}

function normalizeMessages(value: unknown): ChatSessionMessage[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const items: ChatSessionMessage[] = [];
  for (const entry of value) {
    if (!isObject(entry)) {
      continue;
    }
    const role = entry.role === "assistant" ? "assistant" : entry.role === "user" ? "user" : null;
    const content = typeof entry.content === "string" ? entry.content : "";
    if (!role || !content.trim()) {
      continue;
    }
    items.push({
      role,
      content,
      mode: normalizeMode(entry.mode),
      model: typeof entry.model === "string" && entry.model.trim().length > 0 ? entry.model : "qwen2.5:3b",
      timestamp: typeof entry.timestamp === "string" && entry.timestamp.trim().length > 0 ? entry.timestamp : nowIso(),
      metadata: isObject(entry.metadata) ? entry.metadata : null,
    });
  }
  return items;
}

function normalizeSession(payload: unknown): ChatSession | null {
  if (!isObject(payload)) {
    return null;
  }

  const id = typeof payload.id === "string" && payload.id.trim().length > 0 ? payload.id : null;
  if (!id) {
    return null;
  }

  const messages = normalizeMessages(payload.messages);
  const mode = normalizeMode(payload.mode);
  const model =
    typeof payload.model === "string" && payload.model.trim().length > 0 ? payload.model.trim() : "qwen2.5:3b";

  const createdAt =
    typeof payload.createdAt === "string" && payload.createdAt.trim().length > 0 ? payload.createdAt : nowIso();
  const updatedAt =
    typeof payload.updatedAt === "string" && payload.updatedAt.trim().length > 0 ? payload.updatedAt : createdAt;

  const titleFromPayload =
    typeof payload.title === "string" && payload.title.trim().length > 0 ? payload.title.trim() : deriveSessionTitle(messages);

  return {
    id,
    title: titleFromPayload || DEFAULT_SESSION_TITLE,
    createdAt,
    updatedAt,
    pinned: payload.pinned === true,
    mode,
    model,
    messages,
    lastResultSummary: typeof payload.lastResultSummary === "string" ? payload.lastResultSummary : null,
    lastResult: isObject(payload.lastResult) ? (payload.lastResult as QueryResult) : null,
    lastSubmittedQuery: typeof payload.lastSubmittedQuery === "string" ? payload.lastSubmittedQuery : null,
  };
}

export function loadChatSessions(): ChatSession[] {
  if (typeof window === "undefined") {
    return [];
  }

  const raw = window.localStorage.getItem(CHAT_SESSIONS_STORAGE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .map(normalizeSession)
      .filter((item): item is ChatSession => item !== null)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  } catch {
    return [];
  }
}

export function persistChatSessions(sessions: ChatSession[]): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(CHAT_SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
}

export function clearChatSessionsStorage(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(CHAT_SESSIONS_STORAGE_KEY);
}

export function formatRelativeTime(value: string): string {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return value;
  }

  const now = Date.now();
  const deltaMs = now - timestamp.getTime();
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;

  if (deltaMs < minute) {
    return "vừa xong";
  }
  if (deltaMs < hour) {
    return `${Math.max(1, Math.floor(deltaMs / minute))} phút trước`;
  }
  if (deltaMs < day) {
    return `${Math.max(1, Math.floor(deltaMs / hour))} giờ trước`;
  }
  if (deltaMs < 7 * day) {
    return `${Math.max(1, Math.floor(deltaMs / day))} ngày trước`;
  }

  return timestamp.toLocaleString();
}

export function defaultSessionTitle(): string {
  return DEFAULT_SESSION_TITLE;
}
