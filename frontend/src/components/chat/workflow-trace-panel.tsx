import type { TraceEntry } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type WorkflowTracePanelProps = {
  trace: TraceEntry[];
  title?: string;
};

export function WorkflowTracePanel({ trace, title = "Workflow Trace Panel" }: WorkflowTracePanelProps) {
  if (trace.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No trace entries available for this response.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {trace.map((entry, index) => (
            <li key={`${entry.step}-${index}`} className="rounded-md border border-border bg-background px-3 py-2">
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <p className="text-sm font-medium">{entry.step}</p>
                <Badge variant={entry.status === "success" ? "default" : entry.status === "warning" ? "outline" : "muted"}>
                  {entry.status}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">{entry.detail}</p>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
