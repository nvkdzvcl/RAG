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
    <Card className="min-w-0 max-w-full">
      <CardHeader className="pb-3">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="min-w-0 max-w-full overflow-x-hidden">
        <ul className="space-y-2 min-w-0 max-w-full">
          {sources.map((source) => (
            <li
              key={source.id}
              className="min-w-0 max-w-full overflow-hidden rounded-md border border-border bg-background px-3 py-2"
            >
              <p className="truncate text-sm font-medium" title={source.title || source.source}>
                {source.title || source.source}
              </p>
              <p
                className="mt-1 text-xs text-muted-foreground"
                title={source.source}
                style={{ overflowWrap: "anywhere", wordBreak: "break-word" }}
              >
                {source.source} • doc: {source.docId} • chunk: {source.chunkId}
                {source.section ? ` • section: ${source.section}` : ""}
                {source.page ? ` • page: ${source.page}` : ""}
                {source.rerankScore !== null && source.rerankScore !== undefined
                  ? ` • rerank: ${source.rerankScore.toFixed(4)}`
                  : ""}
              </p>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
