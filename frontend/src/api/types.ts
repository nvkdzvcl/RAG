import type { Mode } from "@/types/chat";

export type ApiCitation = {
  chunk_id: string;
  doc_id: string;
  source: string;
  title?: string | null;
  section?: string | null;
  page?: number | null;
  block_type?: string | null;
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

export type ApiDeleteAllDocumentsResponse = {
  status: "deleted";
  deleted_documents: number;
  deleted_files: number;
};

export type ApiDeleteDocumentResponse = {
  status: "deleted";
  document_id: string;
  remaining_documents: number;
  deleted_files: number;
};

export type ApiReindexDocumentsRequest = {
  chunk_size: number;
  chunk_overlap: number;
};

export type ApiReindexDocumentsResponse = {
  status: "reindexed";
  chunk_size: number;
  chunk_overlap: number;
  reindexed_documents: number;
  active_chunks: number;
};

export type ApiChunkingMode = "small" | "medium" | "large" | "custom";
export type ApiChunkConfigMode = "preset" | "custom";
export type ApiRetrievalMode = "low" | "balanced" | "high" | "custom";
export type ApiRetrievalConfigMode = "preset" | "custom";

export type ApiUpdateChunkingSettingsRequest = {
  mode: ApiChunkingMode;
  chunk_size?: number;
  chunk_overlap?: number;
};

export type ApiUpdateChunkingSettingsResponse = {
  status: "reindexed";
  mode: ApiChunkingMode;
  chunk_mode: ApiChunkConfigMode;
  chunk_size: number;
  chunk_overlap: number;
  reindexed_documents: number;
  active_chunks: number;
};

export type ApiUpdateRetrievalSettingsRequest = {
  mode: ApiRetrievalMode;
  top_k?: number;
};

export type ApiUpdateRetrievalSettingsResponse = {
  status: "updated";
  mode: ApiRetrievalMode;
  retrieval_mode: ApiRetrievalConfigMode;
  top_k: number;
  rerank_top_n: number;
  context_top_k: number;
};

export type ApiHealthResponse = {
  status: string;
  llm_model?: string | null;
};
