import type { ModeResult } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type AnswerPanelProps = {
  title?: string;
  result: ModeResult;
};

export function AnswerPanel({ title = "Answer Panel", result }: AnswerPanelProps) {
  const confidenceLabel = result.confidence === null ? "n/a" : `${Math.round(result.confidence * 100)}%`;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <Badge variant="muted">mode: {result.mode}</Badge>
          <Badge variant="outline">status: {result.status}</Badge>
          <Badge>confidence: {confidenceLabel}</Badge>
          {result.latencyMs !== null ? <Badge variant="muted">latency: {result.latencyMs}ms</Badge> : null}
          {result.loopCount !== null ? <Badge variant="outline">loops: {result.loopCount}</Badge> : null}
          {result.stopReason ? <Badge variant="outline">stop: {result.stopReason}</Badge> : null}
        </div>
        <p className="text-sm leading-6">{result.answer}</p>
      </CardContent>
    </Card>
  );
}
