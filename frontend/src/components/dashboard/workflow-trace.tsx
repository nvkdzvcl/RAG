import type { CompareResult, ModeResult, QueryResult, TraceEntry, TraceStatus } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { translations } from "@/lib/translations";

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
    return [{ label: "Luồng xử lý", detail: "Không có dữ liệu luồng xử lý.", status: "info" }];
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
    return [{ label: "Luồng xử lý", detail: "Không có dữ liệu luồng xử lý nâng cao.", status: "info" }];
  }

  const timeline: TimelineItem[] = [];

  for (const entry of trace) {
    const meta = entry.meta;
    if (!isObject(meta)) {
      timeline.push({ label: toTitleCase(entry.step), detail: entry.detail, status: entry.status });
      continue;
    }

    if (meta.step === "retrieval_gate") {
      const needRetrieval = meta.need_retrieval === false ? "bỏ qua truy xuất" : "truy xuất";
      const reason = typeof meta.reason === "string" ? meta.reason : "không có lý do";
      timeline.push({
        label: "Cổng truy xuất",
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
      const selected = typeof meta.selected_count === "number" ? meta.selected_count : 0;
      const generatedStatus = typeof meta.generated_status === "string" ? meta.generated_status : "answered";

      timeline.push({
        label: `Viết lại câu hỏi (Vòng ${loop})`,
        detail: loop === 1 ? `câu hỏi ban đầu=${query}` : `câu hỏi đã viết lại=${query}`,
        status: loop === 1 ? "info" : "success",
      });

      timeline.push({
        label: `Truy xuất (Vòng ${loop})`,
        detail: `số lượng=${retrieved}`,
        status: retrieved > 0 ? "success" : "warning",
      });

      timeline.push({
        label: `Xếp hạng lại (Vòng ${loop})`,
        detail: `số lượng=${reranked}`,
        status: reranked > 0 ? "success" : "warning",
      });

      timeline.push({
        label: `Tạo bản nháp (Vòng ${loop})`,
        detail: `trạng thái=${generatedStatus} | ngữ cảnh chọn=${selected}`,
        status: generatedStatus === "insufficient_evidence" || selected === 0 ? "warning" : "success",
      });

      const critique = isObject(meta.critique) ? meta.critique : undefined;
      const confidence = critique && typeof critique.confidence === "number" ? critique.confidence : null;
      const note = critique && typeof critique.note === "string" ? critique.note : "không có ghi chú";
      const retry = critique && critique.should_retry_retrieval === true;
      const refine = critique && critique.should_refine_answer === true;

      timeline.push({
        label: `Đánh giá (Vòng ${loop})`,
        detail: `${note}${confidence === null ? "" : ` | độ tin cậy=${confidence.toFixed(2)}`}`,
        status: retry || refine ? "warning" : "success",
      });

      let finalDecision = "hoàn tất câu trả lời";
      if (retry) finalDecision = "thử lại truy xuất";
      if (refine) finalDecision = "cải thiện câu trả lời";
      if (critique && critique.enough_evidence === false && !retry && !refine) finalDecision = "từ chối trả lời";

      timeline.push({
        label: `Quyết định cuối (Vòng ${loop})`,
        detail: finalDecision,
        status: retry || finalDecision === "từ chối trả lời" ? "warning" : "success",
      });
      continue;
    }

    if (meta.step === "grounding_check") {
      const groundedScore = typeof meta.grounded_score === "number" ? meta.grounded_score : 0;
      const citationCount = typeof meta.citation_count === "number" ? meta.citation_count : 0;
      const hallucination = meta.hallucination_detected === true;
      const llmFallbackUsed = meta.llm_fallback_used === true;
      const warning = hallucination || citationCount === 0 || llmFallbackUsed;

      timeline.push({
        label: "Grounding",
        detail: `grounded=${groundedScore.toFixed(3)} | citations=${citationCount}`,
        status: warning ? "warning" : "success",
      });
      continue;
    }

    if (meta.step === "hallucination_guard") {
      const triggered = meta.triggered === true;
      const refinedHallucination = meta.refined_hallucination_detected === true;
      timeline.push({
        label: "Grounding (Hallucination Guard)",
        detail: triggered
          ? refinedHallucination
            ? "đã refine nhưng vẫn không bám ngữ cảnh"
            : "đã refine lại theo ngữ cảnh"
          : "không kích hoạt",
        status: refinedHallucination ? "warning" : "success",
      });
      continue;
    }

    if (meta.step === "language_guard") {
      const mismatch = meta.language_mismatch === true;
      timeline.push({
        label: "Ngôn ngữ",
        detail: mismatch ? "phát hiện lệch ngôn ngữ trả lời" : "ngôn ngữ phù hợp",
        status: mismatch ? "warning" : "success",
      });
      continue;
    }

    timeline.push({ label: toTitleCase(entry.step), detail: entry.detail, status: entry.status });
  }

  if (timeline.length === 0) {
    return [{ label: "Luồng xử lý", detail: "Không thể phân tích dữ liệu luồng xử lý.", status: "warning" }];
  }
  return timeline;
}

function isCompare(result: QueryResult): result is CompareResult {
  return result.mode === "compare";
}

function TimelineList({ items }: { items: TimelineItem[] }) {
  const statusTranslation: Record<TraceStatus, string> = {
    success: "thành công",
    warning: "cảnh báo",
    error: "lỗi",
    info: "thông tin",
  };

  return (
    <ul className="space-y-2">
      {items.map((item, index) => (
        <li key={`${item.label}-${index}`} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium text-slate-700">{item.label}</p>
            <Badge variant="outline" className="capitalize">
              {statusTranslation[item.status]}
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
        <CardTitle className="text-base">{translations.trace.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {!result ? <p className="text-sm text-slate-500">{translations.trace.runQuery}</p> : null}

        {result && !isCompare(result) ? (
          <TimelineList items={result.mode === "advanced" ? timelineFromAdvanced(result.trace) : timelineFromStandard(result.trace)} />
        ) : null}

        {result && isCompare(result) ? (
          <div className="grid gap-3 xl:grid-cols-2">
            <ModeTrace label={translations.compare.standard} modeResult={result.standard} />
            <ModeTrace label={translations.compare.advanced} modeResult={result.advanced} />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
