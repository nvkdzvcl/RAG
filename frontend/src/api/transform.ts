import type { ApiCitation, ApiModeResponse, ApiQueryResponse } from "@/api/types";
import type { Citation, CompareResult, ModeResult, QueryResult, SourceReference, TraceEntry, TraceStatus } from "@/types/chat";

function citationToUi(citation: ApiCitation, index: number): Citation {
  return {
    id: `${citation.chunk_id}-${index}`,
    chunkId: citation.chunk_id,
    docId: citation.doc_id,
    source: citation.source,
    title: citation.title ?? null,
    section: citation.section ?? null,
    page: citation.page ?? null,
  };
}

function citationsToSources(citations: Citation[]): SourceReference[] {
  const map = new Map<string, SourceReference>();
  for (const citation of citations) {
    const key = `${citation.docId}:${citation.chunkId}:${citation.source}`;
    if (!map.has(key)) {
      map.set(key, {
        id: key,
        chunkId: citation.chunkId,
        docId: citation.docId,
        source: citation.source,
        title: citation.title,
        section: citation.section,
        page: citation.page,
      });
    }
  }
  return Array.from(map.values());
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function inferTraceStatus(trace: Record<string, unknown>): TraceStatus {
  if (typeof trace.status === "string") {
    if (trace.status === "success" || trace.status === "warning" || trace.status === "info") {
      return trace.status;
    }
  }

  if (trace.step === "generate" && trace.status === "insufficient_evidence") {
    return "warning";
  }
  if (trace.step === "retrieval_gate" && trace.need_retrieval === false) {
    return "warning";
  }
  return "info";
}

function traceToUi(item: unknown): TraceEntry {
  if (!isObject(item)) {
    return {
      step: "unknown",
      detail: String(item),
      status: "info",
    };
  }

  const step = typeof item.step === "string" ? item.step : "step";
  const keys = Object.keys(item).filter((key) => key !== "step");
  const detail = keys.length
    ? keys
        .map((key) => `${key}=${typeof item[key] === "object" ? JSON.stringify(item[key]) : String(item[key])}`)
        .join(" | ")
    : "no details";

  return {
    step,
    detail,
    status: inferTraceStatus(item),
    meta: item,
  };
}

function modeToUi(result: ApiModeResponse): ModeResult {
  const citations = result.citations.map(citationToUi);
  return {
    mode: result.mode,
    answer: result.answer,
    citations,
    confidence: result.confidence,
    status: result.status,
    stopReason: result.stop_reason ?? null,
    latencyMs: result.latency_ms ?? null,
    loopCount: result.loop_count ?? null,
    sources: citationsToSources(citations),
    trace: result.trace.map(traceToUi),
  };
}

export function apiToUi(result: ApiQueryResponse): QueryResult {
  if (result.mode === "compare") {
    const mapped: CompareResult = {
      mode: "compare",
      standard: modeToUi(result.standard),
      advanced: modeToUi(result.advanced),
      comparison: {
        confidenceDelta: result.comparison.confidence_delta ?? null,
        latencyDeltaMs: result.comparison.latency_delta_ms ?? null,
        citationDelta: result.comparison.citation_delta ?? null,
        note: result.comparison.note ?? null,
      },
    };
    return mapped;
  }

  return modeToUi(result);
}
