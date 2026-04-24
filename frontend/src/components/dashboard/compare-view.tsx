import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CitationList } from "@/components/dashboard/citation-list";
import type { CompareResult, ModeResult } from "@/types/chat";
import { translations } from "@/lib/translations";

type CompareViewProps = {
  result: CompareResult;
};

function confidenceLabel(confidence: number | null): string {
  if (confidence === null) {
    return "n/a";
  }
  return `${Math.round(confidence * 100)}%`;
}

function reliabilitySummary(result: CompareResult): string {
  const standard = result.standard;
  const advanced = result.advanced;

  const statusScore = (status: string): number => {
    if (status === "answered") return 2;
    if (status === "partial") return 1;
    return 0;
  };

  const standardScore = statusScore(standard.status) + (standard.confidence ?? 0);
  const advancedScore = statusScore(advanced.status) + (advanced.confidence ?? 0);

  if (advancedScore > standardScore + 0.03) {
    return "Chế độ Nâng cao có vẻ đáng tin cậy hơn cho câu hỏi này (độ tin cậy/trạng thái cao hơn).";
  }
  if (standardScore > advancedScore + 0.03) {
    return "Chế độ Chuẩn có vẻ đáng tin cậy hơn cho câu hỏi này (độ tin cậy/trạng thái cao hơn).";
  }
  return "Cả hai chế độ đều có độ tin cậy tương đương cho câu hỏi này.";
}

function ModeColumn({ title, result }: { title: string; result: ModeResult }) {
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">trạng thái {result.status}</Badge>
          <Badge variant="muted">{translations.metrics.confidence} {confidenceLabel(result.confidence)}</Badge>
          <Badge variant="muted">{translations.metrics.latency} {result.latencyMs === null ? "n/a" : `${result.latencyMs}ms`}</Badge>
          <Badge variant="muted">{translations.citations.title} {result.citations.length}</Badge>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
          <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{result.answer}</p>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{translations.citations.title}</p>
          <CitationList citations={result.citations} compact />
        </div>
      </CardContent>
    </Card>
  );
}

export function CompareView({ result }: CompareViewProps) {
  const confidenceDelta =
    result.comparison.confidenceDelta === null ? "n/a" : result.comparison.confidenceDelta.toFixed(3);
  const latencyDelta = result.comparison.latencyDeltaMs === null ? "n/a" : `${result.comparison.latencyDeltaMs}ms`;
  const citationDelta = result.comparison.citationDelta === null ? "n/a" : `${result.comparison.citationDelta}`;

  return (
    <div className="space-y-4">
      <Card className="border-blue-200 bg-gradient-to-r from-blue-50 to-violet-50 shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Tóm tắt so sánh</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-sm text-slate-700">{reliabilitySummary(result)}</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">chênh lệch độ tin cậy {confidenceDelta}</Badge>
            <Badge variant="outline">chênh lệch thời gian {latencyDelta}</Badge>
            <Badge variant="outline">chênh lệch trích dẫn {citationDelta}</Badge>
          </div>
          {result.comparison.note ? <p className="text-xs text-slate-500">{result.comparison.note}</p> : null}
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <ModeColumn title="Kết quả Chuẩn" result={result.standard} />
        <ModeColumn title="Kết quả Nâng cao" result={result.advanced} />
      </div>
    </div>
  );
}
