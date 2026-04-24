import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CitationList } from "@/components/dashboard/citation-list";
import type { ModeResult } from "@/types/chat";

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

export function AnswerCard({ title = "Assistant Answer", result }: AnswerCardProps) {
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline" className="capitalize">
            {result.mode}
          </Badge>
          <Badge variant="outline">{result.status}</Badge>
          <Badge variant="muted">confidence {confidenceLabel(result.confidence)}</Badge>
          <Badge variant="muted">latency {result.latencyMs === null ? "n/a" : `${result.latencyMs}ms`}</Badge>
          <Badge variant="muted">citations {result.citations.length}</Badge>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
          <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{result.answer}</p>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Citations</p>
          <CitationList citations={result.citations} />
        </div>
      </CardContent>
    </Card>
  );
}
