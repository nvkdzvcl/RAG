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
  response_language?: string;
  language_mismatch?: boolean;
  llm_fallback_used?: boolean;
  grounded_score?: number;
  citation_count?: number;
  hallucination_detected?: boolean;
  trace: Array<Record<string, unknown>>;
};

export type ApiComparison = {
  confidence_delta?: number | null;
  latency_delta_ms?: number | null;
  citation_delta?: number | null;
  grounded_score_delta?: number | null;
  preferred_mode?: string | null;
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
  model?: string | null;
};

export type ApiDocument = {
  id: string;
  filename: string;
  status: string;
  stage?: string | null;
  chunk_count?: number | null;
  created_at?: string | null;
  message?: string | null;
};

export type ApiUploadDocumentResponse = ApiDocument;

export type ApiDocumentStatusResponse = ApiDocument;

export type ApiListDocumentsResponse = ApiDocument[] | { documents: ApiDocument[] };

export type ApiHealthResponse = {
  status: string;
  llm_model?: string | null;
};
