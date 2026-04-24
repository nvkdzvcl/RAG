import type { Mode } from "@/types/chat";

export type ApiCitation = {
  chunk_id: string;
  doc_id: string;
  source: string;
  title?: string | null;
  section?: string | null;
  page?: number | null;
};

export type ApiModeResponse = {
  mode: "standard" | "advanced";
  answer: string;
  citations: ApiCitation[];
  confidence: number | null;
  stop_reason?: string | null;
  status: string;
  latency_ms?: number | null;
  loop_count?: number | null;
  trace: Array<Record<string, unknown>>;
};

export type ApiComparison = {
  confidence_delta?: number | null;
  latency_delta_ms?: number | null;
  citation_delta?: number | null;
  note?: string | null;
};

export type ApiCompareResponse = {
  mode: "compare";
  standard: ApiModeResponse & { mode: "standard" };
  advanced: ApiModeResponse & { mode: "advanced" };
  comparison: ApiComparison;
};

export type ApiQueryResponse = ApiModeResponse | ApiCompareResponse;

export type ApiQueryRequest = {
  query: string;
  mode: Mode;
  chat_history: Array<Record<string, string>>;
};
