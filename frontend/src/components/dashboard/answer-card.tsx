import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CitationList } from "@/components/dashboard/citation-list";
import type { ModeResult } from "@/types/chat";
import { translations } from "@/lib/translations";

type AnswerCardProps = {
  title?: string;
  result: ModeResult;
};

function confidenceLabel(confidence: number | null): string {
  if (confidence === null) {
    return "n/a";
  }
  return `${Math.round(confidence * 100)}%`;
}

export function AnswerCard({ title = "Câu trả lời", result }: AnswerCardProps) {
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline" className="capitalize">
            {translations.modes[result.mode]}
          </Badge>
          <Badge variant="outline">{result.status}</Badge>
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
          <CitationList citations={result.citations} />
        </div>
      </CardContent>
    </Card>
  );
}
