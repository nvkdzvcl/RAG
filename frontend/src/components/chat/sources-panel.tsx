import type { SourceReference } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type SourcesPanelProps = {
  sources: SourceReference[];
  title?: string;
};

export function SourcesPanel({ sources, title = "Sources Panel" }: SourcesPanelProps) {
  if (sources.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No source references were returned.</p>
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
          {sources.map((source) => (
            <li key={source.id} className="rounded-md border border-border bg-background px-3 py-2">
              <p className="text-sm font-medium">{source.title || source.source}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {source.source} • doc: {source.docId} • chunk: {source.chunkId}
                {source.section ? ` • section: ${source.section}` : ""}
                {source.page ? ` • page: ${source.page}` : ""}
              </p>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
