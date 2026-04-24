import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CitationList } from "@/components/dashboard/citation-list";
import type { CompareResult, ModeResult } from "@/types/chat";

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
    return "Advanced appears more reliable for this query (higher confidence/status mix).";
  }
  if (standardScore > advancedScore + 0.03) {
    return "Standard appears more reliable for this query (higher confidence/status mix).";
  }
  return "Both modes appear similarly reliable for this query.";
}

function ModeColumn({ title, result }: { title: string; result: ModeResult }) {
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">status {result.status}</Badge>
          <Badge variant="muted">confidence {confidenceLabel(result.confidence)}</Badge>
          <Badge variant="muted">latency {result.latencyMs === null ? "n/a" : `${result.latencyMs}ms`}</Badge>
          <Badge variant="muted">citations {result.citations.length}</Badge>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
          <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{result.answer}</p>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Citations</p>
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
          <CardTitle className="text-base">Comparison Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-sm text-slate-700">{reliabilitySummary(result)}</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">confidence delta {confidenceDelta}</Badge>
            <Badge variant="outline">latency delta {latencyDelta}</Badge>
            <Badge variant="outline">citation delta {citationDelta}</Badge>
          </div>
          {result.comparison.note ? <p className="text-xs text-slate-500">{result.comparison.note}</p> : null}
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <ModeColumn title="Standard Result" result={result.standard} />
        <ModeColumn title="Advanced Result" result={result.advanced} />
      </div>
    </div>
  );
}
