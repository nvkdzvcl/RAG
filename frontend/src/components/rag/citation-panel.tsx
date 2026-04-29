import { useEffect, useMemo } from "react";
import { FileText } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { QueryResult } from "@/types/chat";
import {
  citationContext,
  citationDocumentName,
  citationScore,
  flattenCitationItems,
} from "@/components/rag/citation-utils";

type CitationPanelProps = {
  result: QueryResult | null;
  activeCitationId: string | null;
  query: string | null;
  onFocusCitation: (panelId: string) => void;
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTermsFromText(text: string | null): string[] {
  if (!text) {
    return [];
  }
  const stopwords = new Set([
    "about",
    "after",
    "also",
    "because",
    "before",
    "from",
    "that",
    "this",
    "with",
    "được",
    "trong",
    "những",
    "rằng",
    "theo",
    "với",
  ]);
  const matches = text.match(/[\p{L}\p{N}_-]{4,}/gu) ?? [];
  return Array.from(new Set(matches.map((item) => item.trim()).filter(Boolean)))
    .filter((item) => !stopwords.has(item.toLowerCase()))
    .sort((left, right) => right.length - left.length)
    .slice(0, 20);
}

function HighlightedText({ text, query }: { text: string; query: string | null }) {
  const terms = useMemo(() => highlightTermsFromText(query), [query]);
  if (terms.length === 0) {
    return <>{text}</>;
  }

  const pattern = terms.map(escapeRegExp).join("|");
  const splitRegex = new RegExp(`(${pattern})`, "giu");
  const matchRegex = new RegExp(`^(?:${pattern})$`, "iu");

  return (
    <>
      {text.split(splitRegex).map((part, index) =>
        matchRegex.test(part) ? (
          <mark key={`${part}-${index}`} className="rounded bg-success/15 px-0.5 text-success">
            {part}
          </mark>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        ),
      )}
    </>
  );
}

export function CitationPanel({
  result,
  activeCitationId,
  query,
  onFocusCitation,
}: CitationPanelProps) {
  const items = useMemo(() => flattenCitationItems(result), [result]);

  useEffect(() => {
    if (!activeCitationId) {
      return;
    }
    const element = document.getElementById(`citation-${activeCitationId}`);
    element?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeCitationId, items.length]);

  if (!result) {
    return (
      <div className="rounded-xl border border-dashed border-border px-4 py-5 text-sm leading-6 text-muted-foreground">
        Run a query to inspect cited chunks.
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="rounded-xl border border-warning/20 bg-warning/10 px-4 py-5 text-sm leading-6 text-warning">
        No citations were returned for the latest answer.
      </div>
    );
  }

  return (
    <div className="space-y-3 scroll-smooth">
      {items.map((item) => {
        const active = item.panelId === activeCitationId;
        const context = citationContext(item.citation);
        const score = citationScore(item.citation);
        return (
          <button
            key={item.panelId}
            id={`citation-${item.panelId}`}
            type="button"
            onClick={() => onFocusCitation(item.panelId)}
            className={cn(
              "w-full rounded-xl border p-3 text-left transition",
              active
                ? "animate-citation-focus scroll-mt-3 border-primary/60 bg-primary-light shadow-sm shadow-primary/10"
                : "scroll-mt-3 border-border bg-card/70 hover:-translate-y-px hover:border-primary/30 hover:bg-card active:translate-y-0 active:scale-[0.99]",
            )}
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent">
                <FileText className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
                    {citationDocumentName(item.citation)}
                  </p>
                  <Badge variant="outline" className="border-border bg-muted/50 text-[10px] uppercase text-muted-foreground">
                    {item.branchLabel}
                  </Badge>
                </div>
                <div className="mt-1 flex flex-wrap gap-2 font-mono text-[11px] text-muted-foreground">
                  <span>[{item.index + 1}]</span>
                  <span>chunk={item.citation.chunkId}</span>
                  {item.citation.page !== null && item.citation.page !== undefined ? (
                    <span>page={item.citation.page}</span>
                  ) : null}
                  {score !== null ? <span>score={score.toFixed(4)}</span> : null}
                </div>
                <p className="mt-3 line-clamp-6 whitespace-pre-wrap text-xs leading-6 text-muted-foreground">
                  {context.trim() ? (
                    <HighlightedText text={context.trim()} query={query} />
                  ) : (
                    "No chunk text available."
                  )}
                </p>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
