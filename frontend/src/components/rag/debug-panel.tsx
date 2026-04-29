import { Activity, CheckCircle2, CircleDashed, Timer } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { ModeResult, QueryResult, TraceEntry } from "@/types/chat";

type DebugPanelProps = {
  result: QueryResult | null;
};

type DebugMetrics = {
  totalMs: number | null;
  retrievalMs: number | null;
  rerankMs: number | null;
  llmMs: number | null;
  groundingMs: number | null;
  fastPathUsed: boolean | null;
  queryComplexity: string | null;
  rerankerUsed: string | null;
  groundingSemanticUsed: boolean | null;
  cacheHits: Array<{ label: string; value: boolean | null }>;
  llmCallCountEstimate: number | null;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function boolValue(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function lastMeta(trace: TraceEntry[], step: string): Record<string, unknown> | null {
  for (let index = trace.length - 1; index >= 0; index -= 1) {
    const item = trace[index];
    if (item.step === step && isObject(item.meta)) {
      return item.meta;
    }
  }
  return null;
}

function firstPresent<T>(...values: Array<T | null | undefined>): T | null {
  for (const value of values) {
    if (value !== null && value !== undefined) {
      return value;
    }
  }
  return null;
}

function nestedNumber(meta: Record<string, unknown> | null, key: string): number | null {
  const timings = isObject(meta?.pipeline_step_timings_ms) ? meta?.pipeline_step_timings_ms : null;
  return firstPresent(numberValue(meta?.[key]), numberValue(timings?.[key]));
}

function metricsForResult(result: ModeResult): DebugMetrics {
  const summary = lastMeta(result.trace, "timing_summary");
  const retrieve = lastMeta(result.trace, "retrieve");
  const rerank = lastMeta(result.trace, "rerank");
  const generate = lastMeta(result.trace, "generate");
  const grounding = lastMeta(result.trace, "grounding_check");
  const loop = lastMeta(result.trace, "loop");
  const evidence = lastMeta(result.trace, "evidence_decision");

  const groundingMs = firstPresent(
    nestedNumber(summary, "grounding_ms"),
    nestedNumber(summary, "final_grounding_ms"),
    nestedNumber(grounding, "grounding_ms"),
    nestedNumber(evidence, "grounding_ms"),
  );

  return {
    totalMs: firstPresent(result.latencyMs, numberValue(summary?.total_ms)),
    retrievalMs: firstPresent(nestedNumber(summary, "retrieval_total_ms"), numberValue(retrieve?.retrieval_total_ms)),
    rerankMs: firstPresent(nestedNumber(summary, "rerank_ms"), numberValue(rerank?.rerank_ms)),
    llmMs: firstPresent(nestedNumber(summary, "llm_generate_ms"), numberValue(generate?.llm_generate_ms)),
    groundingMs,
    fastPathUsed: firstPresent(
      boolValue(summary?.fast_path_used),
      boolValue(generate?.fast_path_used),
      boolValue(loop?.fast_path_used),
    ),
    queryComplexity: firstPresent(
      stringValue(summary?.query_complexity),
      stringValue(retrieve?.query_complexity),
      stringValue(loop?.query_complexity),
    ),
    rerankerUsed: firstPresent(stringValue(summary?.reranker_used), stringValue(rerank?.reranker_used)),
    groundingSemanticUsed: firstPresent(
      boolValue(summary?.grounding_semantic_used),
      boolValue(grounding?.grounding_semantic_used),
      boolValue(evidence?.grounding_semantic_used),
    ),
    cacheHits: [
      { label: "embedding", value: firstPresent(boolValue(summary?.embedding_cache_hit), boolValue(retrieve?.embedding_cache_hit)) },
      { label: "retrieval", value: firstPresent(boolValue(summary?.retrieval_cache_hit), boolValue(retrieve?.retrieval_cache_hit)) },
      { label: "rerank", value: firstPresent(boolValue(summary?.rerank_cache_hit), boolValue(rerank?.rerank_cache_hit)) },
      { label: "llm", value: firstPresent(boolValue(summary?.llm_cache_hit), boolValue(generate?.llm_cache_hit), boolValue(grounding?.llm_cache_hit)) },
      { label: "grounding", value: firstPresent(boolValue(summary?.grounding_cache_hit), boolValue(grounding?.grounding_cache_hit)) },
    ],
    llmCallCountEstimate: firstPresent(
      numberValue(summary?.llm_call_count_estimate),
      numberValue(grounding?.llm_call_count_estimate),
      numberValue(evidence?.llm_call_count_estimate),
    ),
  };
}

function msLabel(value: number | null): string {
  return value === null ? "n/a" : `${value}ms`;
}

function flagLabel(value: boolean | null): string {
  if (value === null) {
    return "n/a";
  }
  return value ? "yes" : "no";
}

function MetricRow({
  label,
  value,
  total,
}: {
  label: string;
  value: number | null;
  total: number | null;
}) {
  const width = value !== null && total && total > 0 ? `${Math.min(100, Math.round((value / total) * 100))}%` : "0%";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 font-mono text-[11px] text-muted-foreground">
        <span>{label}</span>
        <span>{msLabel(value)}</span>
      </div>
      <div className="h-2 rounded-full bg-muted">
        <div className="h-2 rounded-full bg-gradient-to-r from-primary to-accent" style={{ width }} />
      </div>
    </div>
  );
}

function CacheBadge({ label, value }: { label: string; value: boolean | null }) {
  return (
    <Badge
      variant="outline"
      className={
        value
          ? "border-success/25 bg-success/10 text-success"
          : "border-border bg-muted/50 text-muted-foreground"
      }
    >
      {label}: {flagLabel(value)}
    </Badge>
  );
}

function ModeDebugBlock({ title, result }: { title: string; result: ModeResult }) {
  const metrics = metricsForResult(result);
  return (
    <section className="rounded-xl border border-border bg-card/80 p-3">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Timer className="h-4 w-4 text-primary" />
          <p className="text-sm font-semibold text-foreground">{title}</p>
        </div>
        <span className="font-mono text-xs text-muted-foreground">{msLabel(metrics.totalMs)}</span>
      </div>

      <div className="space-y-3">
        <MetricRow label="retrieval" value={metrics.retrievalMs} total={metrics.totalMs} />
        <MetricRow label="rerank" value={metrics.rerankMs} total={metrics.totalMs} />
        <MetricRow label="LLM" value={metrics.llmMs} total={metrics.totalMs} />
        <MetricRow label="grounding" value={metrics.groundingMs} total={metrics.totalMs} />
      </div>

      <div className="mt-4 grid gap-2 text-xs">
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">fast_path_used</span>
          <span className="font-mono text-foreground">{flagLabel(metrics.fastPathUsed)}</span>
        </div>
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">query_complexity</span>
          <span className="font-mono text-foreground">{metrics.queryComplexity ?? "n/a"}</span>
        </div>
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">reranker_used</span>
          <span className="font-mono text-foreground">{metrics.rerankerUsed ?? "n/a"}</span>
        </div>
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">grounding_semantic_used</span>
          <span className="font-mono text-foreground">{flagLabel(metrics.groundingSemanticUsed)}</span>
        </div>
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">llm_call_count_estimate</span>
          <span className="font-mono text-foreground">{metrics.llmCallCountEstimate ?? "n/a"}</span>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {metrics.cacheHits.map((item) => (
          <CacheBadge key={item.label} label={item.label} value={item.value} />
        ))}
      </div>
    </section>
  );
}

export function DebugPanel({ result }: DebugPanelProps) {
  if (!result) {
    return (
      <div className="rounded-xl border border-dashed border-border px-4 py-5 text-sm leading-6 text-muted-foreground">
        Run a query to inspect latency and optimization flags.
      </div>
    );
  }

  if (result.mode === "compare") {
    return (
      <div className="space-y-3">
        <ModeDebugBlock title="Standard" result={result.standard} />
        <ModeDebugBlock title="Advanced" result={result.advanced} />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-border bg-card/80 p-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <Activity className="h-4 w-4 text-primary" />
          Latest Run
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-lg bg-muted/50 px-3 py-2">
            <p className="text-muted-foreground">status</p>
            <p className="mt-1 font-mono text-foreground">{result.status}</p>
          </div>
          <div className="rounded-lg bg-muted/50 px-3 py-2">
            <p className="text-muted-foreground">stop_reason</p>
            <p className="mt-1 font-mono text-foreground">{result.stopReason ?? "n/a"}</p>
          </div>
          <div className="rounded-lg bg-muted/50 px-3 py-2">
            <p className="text-muted-foreground">grounded</p>
            <p className="mt-1 flex items-center gap-1 font-mono text-foreground">
              {result.groundedScore >= 0.1 ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-success" />
              ) : (
                <CircleDashed className="h-3.5 w-3.5 text-warning" />
              )}
              {result.groundedScore.toFixed(3)}
            </p>
          </div>
          <div className="rounded-lg bg-muted/50 px-3 py-2">
            <p className="text-muted-foreground">citations</p>
            <p className="mt-1 font-mono text-foreground">{result.citationCount}</p>
          </div>
        </div>
      </div>
      <ModeDebugBlock title={result.mode === "advanced" ? "Advanced" : "Standard"} result={result} />
    </div>
  );
}
