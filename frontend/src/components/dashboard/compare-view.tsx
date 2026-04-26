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

function isFallbackUsed(result: ModeResult): boolean {
  const stopReason = result.stopReason?.toLowerCase() ?? "";
  return result.llmFallbackUsed || stopReason.includes("fallback");
}

function reliabilitySummary(result: CompareResult): string {
  const standard = result.standard;
  const responseLanguage = standard.responseLanguage;
  const explicitNote = result.comparison.note?.trim();
  if (explicitNote) {
    return explicitNote;
  }

  const winner = result.comparison.winner;
  if (responseLanguage === "vi") {
    if (winner === "standard") {
      return "Chuẩn đáng tin cậy hơn vì có trích dẫn và độ bám tài liệu cao hơn";
    }
    if (winner === "advanced") {
      return "Nâng cao đáng tin cậy hơn vì có trích dẫn và độ bám tài liệu cao hơn";
    }
    if (winner === "both_weak") {
      return "Cả hai cần kiểm tra lại vì thiếu bằng chứng đủ mạnh";
    }
    return "Hai chế độ có độ tin cậy tương đương, cần kiểm tra thêm theo ngữ cảnh";
  }

  if (winner === "standard") {
    return "Standard is more reliable due to stronger citations and groundedness.";
  }
  if (winner === "advanced") {
    return "Advanced is more reliable due to stronger citations and groundedness.";
  }
  if (winner === "both_weak") {
    return "Both branches need review due to weak evidence.";
  }
  return "Both branches are similarly reliable; review context to choose.";
}

function ModeColumn({ title, result }: { title: string; result: ModeResult }) {
  const fallbackUsed = isFallbackUsed(result);

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">trạng thái {result.status}</Badge>
          <Badge variant="muted">lang {result.responseLanguage}</Badge>
          <Badge variant="muted">{translations.metrics.confidence} {confidenceLabel(result.confidence)}</Badge>
          <Badge variant="muted">{translations.metrics.grounded} {(result.groundedScore * 100).toFixed(0)}%</Badge>
          <Badge variant="muted">{translations.metrics.latency} {result.latencyMs === null ? "n/a" : `${result.latencyMs}ms`}</Badge>
          <Badge variant="muted">{translations.citations.title} {result.citationCount}</Badge>
          {result.citationCount === 0 ? (
            <Badge variant="outline" className="border-amber-300 bg-amber-50 text-amber-700">
              {translations.answer.noCitationWarning}
            </Badge>
          ) : null}
          {result.hallucinationDetected ? (
            <Badge variant="outline" className="border-rose-300 bg-rose-50 text-rose-700">
              {translations.answer.hallucinationWarning}
            </Badge>
          ) : null}
          {result.languageMismatch ? (
            <Badge variant="outline" className="border-amber-300 bg-amber-50 text-amber-700">
              {translations.answer.languageMismatch}
            </Badge>
          ) : null}
          {fallbackUsed ? (
            <Badge variant="outline" className="border-orange-300 bg-orange-50 text-orange-700">
              {translations.answer.fallbackWarning}
            </Badge>
          ) : null}
          {result.llmFallbackUsed ? (
            <Badge variant="outline" className="border-orange-300 bg-orange-50 text-orange-700">
              {translations.answer.llmFallbackWarning}
            </Badge>
          ) : null}
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
  const winnerLabel =
    result.comparison.winner === null
      ? "n/a"
      : result.comparison.winner === "both_weak"
        ? "both weak"
        : result.comparison.winner;
  const standardScore =
    result.comparison.standardScore === null ? "n/a" : result.comparison.standardScore.toFixed(3);
  const advancedScore =
    result.comparison.advancedScore === null ? "n/a" : result.comparison.advancedScore.toFixed(3);
  const confidenceDelta =
    result.comparison.confidenceDelta === null ? "n/a" : result.comparison.confidenceDelta.toFixed(3);
  const latencyDelta = result.comparison.latencyDeltaMs === null ? "n/a" : `${result.comparison.latencyDeltaMs}ms`;
  const citationDelta = result.comparison.citationDelta === null ? "n/a" : `${result.comparison.citationDelta}`;
  const groundedDelta =
    result.comparison.groundedScoreDelta === null ? "n/a" : result.comparison.groundedScoreDelta.toFixed(3);

  return (
    <div className="space-y-4">
      <Card className="border-blue-200 bg-gradient-to-r from-blue-50 to-violet-50 shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Tóm tắt so sánh</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-sm text-slate-700">{reliabilitySummary(result)}</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">winner {winnerLabel}</Badge>
            <Badge variant="outline">điểm chuẩn {standardScore}</Badge>
            <Badge variant="outline">điểm nâng cao {advancedScore}</Badge>
            <Badge variant="outline">chênh lệch độ tin cậy {confidenceDelta}</Badge>
            <Badge variant="outline">chênh lệch độ bám tài liệu {groundedDelta}</Badge>
            <Badge variant="outline">chênh lệch thời gian {latencyDelta}</Badge>
            <Badge variant="outline">chênh lệch trích dẫn {citationDelta}</Badge>
          </div>
          {result.comparison.reasons.length > 0 ? (
            <div className="space-y-1">
              {result.comparison.reasons.map((reason, index) => (
                <p key={`${reason}-${index}`} className="text-xs text-slate-600">
                  {`- ${reason}`}
                </p>
              ))}
            </div>
          ) : null}
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
