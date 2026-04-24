import type { CompareResult } from "@/types/chat";
import { AnswerPanel } from "@/components/chat/answer-panel";
import { CitationsPanel } from "@/components/chat/citations-panel";
import { SourcesPanel } from "@/components/chat/sources-panel";
import { WorkflowTracePanel } from "@/components/chat/workflow-trace-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

type CompareLayoutProps = {
  result: CompareResult;
};

export function CompareLayout({ result }: CompareLayoutProps) {
  const confidenceDelta =
    result.comparison.confidenceDelta === null ? "n/a" : result.comparison.confidenceDelta.toFixed(3);
  const latencyDelta = result.comparison.latencyDeltaMs === null ? "n/a" : `${result.comparison.latencyDeltaMs}ms`;
  const citationDelta = result.comparison.citationDelta === null ? "n/a" : `${result.comparison.citationDelta}`;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>Comparison Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          <p className="text-sm text-muted-foreground">{result.comparison.note || "No comparison note returned."}</p>
          <p className="text-xs text-muted-foreground">confidence_delta: {confidenceDelta}</p>
          <p className="text-xs text-muted-foreground">latency_delta_ms: {latencyDelta}</p>
          <p className="text-xs text-muted-foreground">citation_delta: {citationDelta}</p>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
          <AnswerPanel title="Standard Answer" result={result.standard} />
          <CitationsPanel title="Standard Citations" citations={result.standard.citations} />
          <SourcesPanel title="Standard Sources" sources={result.standard.sources} />
        </div>
        <div className="space-y-4">
          <AnswerPanel title="Advanced Answer" result={result.advanced} />
          <CitationsPanel title="Advanced Citations" citations={result.advanced.citations} />
          <SourcesPanel title="Advanced Sources" sources={result.advanced.sources} />
          <Separator />
          <WorkflowTracePanel title="Advanced Trace" trace={result.advanced.trace} />
        </div>
      </div>
    </div>
  );
}
