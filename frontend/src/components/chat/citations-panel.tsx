import type { Citation } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type CitationsPanelProps = {
  citations: Citation[];
  title?: string;
};

export function CitationsPanel({ citations, title = "Citations Panel" }: CitationsPanelProps) {
  if (citations.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No citations were returned for this response.</p>
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
          {citations.map((item) => (
            <li key={item.id} className="rounded-md border border-border bg-background px-3 py-2 text-sm">
              <p className="font-medium">{item.title || item.source}</p>
              <p className="text-muted-foreground">
                {item.source} • chunk: {item.chunkId}
                {item.section ? ` • section: ${item.section}` : ""}
                {item.page ? ` • page: ${item.page}` : ""}
              </p>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
