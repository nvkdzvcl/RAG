import type { CompareResult, ModeResult, QueryResult, TraceEntry, TraceStatus } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type WorkflowTraceProps = {
  result: QueryResult | null;
};

type TimelineItem = {
  label: string;
  detail: string;
  status: TraceStatus;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toTitleCase(step: string): string {
  return step
    .replaceAll("_", " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function timelineFromStandard(trace: TraceEntry[]): TimelineItem[] {
  if (trace.length === 0) {
    return [{ label: "Trace", detail: "No trace entries were returned.", status: "info" }];
  }

  return trace.map((entry) => {
    const meta = entry.meta;
    if (!isObject(meta)) {
      return {
        label: toTitleCase(entry.step),
        detail: entry.detail,
        status: entry.status,
      };
    }

    const parts: string[] = [];
    if (typeof meta.query === "string") parts.push(`query=${meta.query}`);
    if (typeof meta.count === "number") parts.push(`count=${meta.count}`);
    if (typeof meta.status === "string") parts.push(`status=${meta.status}`);
    if (typeof meta.stop_reason === "string") parts.push(`stop=${meta.stop_reason}`);

    return {
      label: toTitleCase(String(meta.step ?? entry.step)),
      detail: parts.length > 0 ? parts.join(" | ") : entry.detail,
      status: entry.status,
    };
  });
}

function timelineFromAdvanced(trace: TraceEntry[]): TimelineItem[] {
  if (trace.length === 0) {
    return [{ label: "Trace", detail: "No advanced trace entries were returned.", status: "info" }];
  }

  const timeline: TimelineItem[] = [];

  for (const entry of trace) {
    const meta = entry.meta;
    if (!isObject(meta)) {
      timeline.push({ label: toTitleCase(entry.step), detail: entry.detail, status: entry.status });
      continue;
    }

    if (meta.step === "retrieval_gate") {
      const needRetrieval = meta.need_retrieval === false ? "skip retrieval" : "retrieve";
      const reason = typeof meta.reason === "string" ? meta.reason : "no reason provided";
      timeline.push({
        label: "Retrieval Gate",
        detail: `${needRetrieval} | ${reason}`,
        status: meta.need_retrieval === false ? "warning" : "success",
      });
      continue;
    }

    if (meta.step === "loop") {
      const loop = typeof meta.loop === "number" ? meta.loop : 1;
      const query = typeof meta.query === "string" ? meta.query : "n/a";
      const retrieved = typeof meta.retrieved_count === "number" ? meta.retrieved_count : 0;
      const reranked = typeof meta.reranked_count === "number" ? meta.reranked_count : 0;

      timeline.push({
        label: `Query Rewrite (Loop ${loop})`,
        detail: loop === 1 ? `initial query=${query}` : `rewritten query=${query}`,
        status: loop === 1 ? "info" : "success",
      });

      timeline.push({
        label: `Retrieval (Loop ${loop})`,
        detail: `retrieved_count=${retrieved}`,
        status: retrieved > 0 ? "success" : "warning",
      });

      timeline.push({
        label: `Rerank (Loop ${loop})`,
        detail: `reranked_count=${reranked}`,
        status: reranked > 0 ? "success" : "warning",
      });

      const critique = isObject(meta.critique) ? meta.critique : undefined;
      const confidence = critique && typeof critique.confidence === "number" ? critique.confidence : null;
      const note = critique && typeof critique.note === "string" ? critique.note : "no critique note";
      const retry = critique && critique.should_retry_retrieval === true;
      const refine = critique && critique.should_refine_answer === true;

      timeline.push({
        label: `Critique (Loop ${loop})`,
        detail: `${note}${confidence === null ? "" : ` | confidence=${confidence.toFixed(2)}`}`,
        status: retry || refine ? "warning" : "success",
      });

      let finalDecision = "finalize answer";
      if (retry) finalDecision = "retry retrieval";
      if (refine) finalDecision = "refine answer";
      if (critique && critique.enough_evidence === false && !retry && !refine) finalDecision = "abstain path";

      timeline.push({
        label: `Final Decision (Loop ${loop})`,
        detail: finalDecision,
        status: retry || finalDecision === "abstain path" ? "warning" : "success",
      });
    }
  }

  if (timeline.length === 0) {
    return [{ label: "Trace", detail: "Unable to parse advanced trace payload.", status: "warning" }];
  }
  return timeline;
}

function isCompare(result: QueryResult): result is CompareResult {
  return result.mode === "compare";
}

function TimelineList({ items }: { items: TimelineItem[] }) {
  return (
    <ul className="space-y-2">
      {items.map((item, index) => (
        <li key={`${item.label}-${index}`} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium text-slate-700">{item.label}</p>
            <Badge variant="outline" className="capitalize">
              {item.status}
            </Badge>
          </div>
          <p className="text-xs text-slate-500">{item.detail}</p>
        </li>
      ))}
    </ul>
  );
}

function ModeTrace({ label, modeResult }: { label: string; modeResult: ModeResult }) {
  const items = modeResult.mode === "advanced" ? timelineFromAdvanced(modeResult.trace) : timelineFromStandard(modeResult.trace);
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <TimelineList items={items} />
    </div>
  );
}

export function WorkflowTrace({ result }: WorkflowTraceProps) {
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Workflow Trace</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {!result ? <p className="text-sm text-slate-500">Run a query to inspect workflow trace steps.</p> : null}

        {result && !isCompare(result) ? (
          <TimelineList items={result.mode === "advanced" ? timelineFromAdvanced(result.trace) : timelineFromStandard(result.trace)} />
        ) : null}

        {result && isCompare(result) ? (
          <div className="grid gap-3 xl:grid-cols-2">
            <ModeTrace label="Standard" modeResult={result.standard} />
            <ModeTrace label="Advanced" modeResult={result.advanced} />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
