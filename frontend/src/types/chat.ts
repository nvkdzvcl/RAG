export type Mode = "standard" | "advanced" | "compare";
export type SingleMode = Exclude<Mode, "compare">;

export type Citation = {
  id: string;
  chunkId: string;
  docId: string;
  source: string;
  title?: string | null;
  section?: string | null;
  page?: number | null;
};

export type SourceReference = {
  id: string;
  chunkId: string;
  docId: string;
  source: string;
  title?: string | null;
  section?: string | null;
  page?: number | null;
  rerankScore?: number | null;
};

export type TraceStatus = "info" | "success" | "warning";

export type TraceEntry = {
  step: string;
  detail: string;
  status: TraceStatus;
  meta?: Record<string, unknown>;
};

export type ModeResult = {
  mode: SingleMode;
  answer: string;
  citations: Citation[];
  citationCount: number;
  confidence: number | null;
  groundedScore: number;
  status: string;
  stopReason: string | null;
  latencyMs: number | null;
  loopCount: number | null;
  responseLanguage: string;
  languageMismatch: boolean;
  hallucinationDetected: boolean;
  llmFallbackUsed: boolean;
  sources: SourceReference[];
  trace: TraceEntry[];
};

export type ComparisonSummary = {
  confidenceDelta: number | null;
  latencyDeltaMs: number | null;
  citationDelta: number | null;
  groundedScoreDelta: number | null;
  preferredMode: string | null;
  note: string | null;
};

export type CompareResult = {
  mode: "compare";
  standard: ModeResult;
  advanced: ModeResult;
  comparison: ComparisonSummary;
};

export type QueryResult = ModeResult | CompareResult;
