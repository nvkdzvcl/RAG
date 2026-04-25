import type { Mode, QueryResult } from "@/types/chat";

export type ChatMessageRole = "user" | "assistant";

export type ChatSessionMessage = {
  role: ChatMessageRole;
  content: string;
  mode: Mode;
  model: string;
  timestamp: string;
  metadata?: Record<string, unknown> | null;
};

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  mode: Mode;
  model: string;
  messages: ChatSessionMessage[];
  lastResultSummary?: string | null;
  lastResult?: QueryResult | null;
  lastSubmittedQuery?: string | null;
};
