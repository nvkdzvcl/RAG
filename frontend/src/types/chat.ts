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
};

export type TraceStatus = "info" | "success" | "warning";

export type TraceEntry = {
  step: string;
  detail: string;
  status: TraceStatus;
};

export type ModeResult = {
  mode: SingleMode;
  answer: string;
  citations: Citation[];
  confidence: number | null;
  status: string;
  stopReason: string | null;
  latencyMs: number | null;
  loopCount: number | null;
  sources: SourceReference[];
  trace: TraceEntry[];
};

export type ComparisonSummary = {
  confidenceDelta: number | null;
  latencyDeltaMs: number | null;
  citationDelta: number | null;
  note: string | null;
};

export type CompareResult = {
  mode: "compare";
  standard: ModeResult;
  advanced: ModeResult;
  comparison: ComparisonSummary;
};

export type QueryResult = ModeResult | CompareResult;
