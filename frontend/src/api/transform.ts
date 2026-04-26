import type { ApiCitation, ApiModeResponse, ApiQueryResponse } from "@/api/types";
import type { Citation, CompareResult, ModeResult, QueryResult, SourceReference, TraceEntry, TraceStatus } from "@/types/chat";

function filenameFromSourcePath(path: string): string | null {
  const trimmed = path.trim();
  if (!trimmed) {
    return null;
  }
  const normalized = trimmed.replace(/\\/g, "/").replace(/\/+$/, "");
  const candidate = normalized.split("/").filter(Boolean).pop();
  return candidate && candidate.length > 0 ? candidate : trimmed;
}

function citationToUi(citation: ApiCitation, index: number): Citation {
  const derivedFilename = filenameFromSourcePath(citation.source);
  return {
    id: `${citation.chunk_id}-${index}`,
    chunkId: citation.chunk_id,
    docId: citation.doc_id,
    source: citation.source,
    fileName: citation.file_name ?? derivedFilename ?? null,
    fileType: citation.file_type ?? null,
    title: citation.title ?? null,
    section: citation.section ?? null,
    page: citation.page ?? null,
    blockType: citation.block_type ?? null,
  };
}

function extractRerankScoreMap(traceItems: Array<Record<string, unknown>>): Map<string, number> {
  const scores = new Map<string, number>();

  for (const item of traceItems) {
    if (!isObject(item) || typeof item.step !== "string") {
      continue;
    }

    let docs: unknown = null;
    if (item.step === "rerank") {
      docs = item.docs;
    } else if (item.step === "loop") {
      docs = item.reranked_docs;
    }

    if (!Array.isArray(docs)) {
      continue;
    }

    for (const doc of docs) {
      if (!isObject(doc) || typeof doc.chunk_id !== "string") {
        continue;
      }
      const value = doc.rerank_score;
      if (typeof value !== "number") {
        continue;
      }
      scores.set(doc.chunk_id, value);
    }
  }

  return scores;
}

function citationsToSources(citations: Citation[], rerankScores: Map<string, number>): SourceReference[] {
  const map = new Map<string, SourceReference>();
  for (const citation of citations) {
    const key = `${citation.docId}:${citation.chunkId}:${citation.source}`;
    if (!map.has(key)) {
      map.set(key, {
        id: key,
        chunkId: citation.chunkId,
        docId: citation.docId,
        source: citation.source,
        fileName: citation.fileName ?? null,
        fileType: citation.fileType ?? null,
        title: citation.title,
        section: citation.section,
        page: citation.page,
        blockType: citation.blockType,
        rerankScore: rerankScores.get(citation.chunkId) ?? null,
      });
    }
  }
  return Array.from(map.values());
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type ParsedEmbeddedAnswer = {
  answer: string;
  confidence: number | null;
  status: string | null;
};

function parseAnswerObjectCandidate(raw: string): ParsedEmbeddedAnswer | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!isObject(parsed) || typeof parsed.answer !== "string" || parsed.answer.trim().length === 0) {
      return null;
    }

    const confidence =
      typeof parsed.confidence === "number" && Number.isFinite(parsed.confidence) ? parsed.confidence : null;
    const status = typeof parsed.status === "string" && parsed.status.trim().length > 0 ? parsed.status : null;
    return {
      answer: parsed.answer.trim(),
      confidence,
      status,
    };
  } catch {
    return null;
  }
}

function parseEmbeddedAnswer(rawAnswer: string): ParsedEmbeddedAnswer {
  const trimmed = rawAnswer.trim();
  if (!trimmed) {
    return {
      answer: rawAnswer,
      confidence: null,
      status: null,
    };
  }

  const candidates: string[] = [trimmed];
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fencedMatch?.[1]) {
    candidates.push(fencedMatch[1].trim());
  }
  const objectStart = trimmed.indexOf("{");
  const objectEnd = trimmed.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    candidates.push(trimmed.slice(objectStart, objectEnd + 1).trim());
  }

  for (const candidate of candidates) {
    const parsed = parseAnswerObjectCandidate(candidate);
    if (parsed) {
      // Handle nested answer payloads like {"answer":"{\"answer\":\"...\"}"}.
      const nested = parseAnswerObjectCandidate(parsed.answer);
      if (nested) {
        return {
          answer: nested.answer,
          confidence: parsed.confidence ?? nested.confidence,
          status: parsed.status ?? nested.status,
        };
      }
      return parsed;
    }
  }

  return {
    answer: rawAnswer,
    confidence: null,
    status: null,
  };
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
  if (trace.step === "grounding_check") {
    if (trace.hallucination_detected === true || trace.citation_count === 0 || trace.llm_fallback_used === true) {
      return "warning";
    }
    return "success";
  }
  if (trace.step === "hallucination_guard") {
    if (trace.refined_hallucination_detected === true) {
      return "warning";
    }
    return "success";
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
  const rerankScores = extractRerankScoreMap(result.trace);
  const embeddedAnswer = parseEmbeddedAnswer(result.answer);
  const confidence =
    typeof result.confidence === "number" && Number.isFinite(result.confidence)
      ? result.confidence
      : embeddedAnswer.confidence;
  const status =
    typeof result.status === "string" && result.status.trim().length > 0
      ? result.status
      : embeddedAnswer.status ?? "answered";
  return {
    mode: result.mode,
    answer: embeddedAnswer.answer,
    citations,
    citationCount:
      typeof result.citation_count === "number" && Number.isFinite(result.citation_count)
        ? result.citation_count
        : citations.length,
    confidence,
    groundedScore:
      typeof result.grounded_score === "number" && Number.isFinite(result.grounded_score)
        ? result.grounded_score
        : 0,
    status,
    stopReason: result.stop_reason ?? null,
    latencyMs: result.latency_ms ?? null,
    loopCount: result.loop_count ?? null,
    responseLanguage: result.response_language ?? "en",
    languageMismatch: result.language_mismatch ?? false,
    hallucinationDetected: result.hallucination_detected ?? false,
    llmFallbackUsed: result.llm_fallback_used ?? false,
    sources: citationsToSources(citations, rerankScores),
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
        winner: result.comparison.winner ?? null,
        reasons: Array.isArray(result.comparison.reasons) ? result.comparison.reasons : [],
        standardScore:
          typeof result.comparison.standard_score === "number" && Number.isFinite(result.comparison.standard_score)
            ? result.comparison.standard_score
            : null,
        advancedScore:
          typeof result.comparison.advanced_score === "number" && Number.isFinite(result.comparison.advanced_score)
            ? result.comparison.advanced_score
            : null,
        confidenceDelta: result.comparison.confidence_delta ?? null,
        latencyDeltaMs: result.comparison.latency_delta_ms ?? null,
        citationDelta: result.comparison.citation_delta ?? null,
        groundedScoreDelta: result.comparison.grounded_score_delta ?? null,
        preferredMode: result.comparison.preferred_mode ?? null,
        note: result.comparison.note ?? null,
      },
    };
    return mapped;
  }

  return modeToUi(result);
}
