import { useEffect, useMemo } from "react";
import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { CitationPanelItem } from "@/components/rag/citation-utils";
import { citationContext, citationDocumentName } from "@/components/rag/citation-utils";

type CitationPreviewModalProps = {
  item: CitationPanelItem | null;
  query: string | null;
  onClose: () => void;
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
  return Array.from(new Set(matches.map((part) => part.trim()).filter(Boolean)))
    .filter((part) => !stopwords.has(part.toLowerCase()))
    .sort((left, right) => right.length - left.length)
    .slice(0, 24);
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
          <mark
            key={`${part}-${index}`}
            className="rounded bg-warning/30 px-0.5 text-foreground"
          >
            {part}
          </mark>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        ),
      )}
    </>
  );
}

export function CitationPreviewModal({
  item,
  query,
  onClose,
}: CitationPreviewModalProps) {
  useEffect(() => {
    if (!item) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [item, onClose]);

  if (!item) {
    return null;
  }

  const context = citationContext(item.citation).trim();
  const score = item.citation.score;
  const rerankScore = item.citation.rerankScore;
  const metadata = [
    item.citation.page !== null && item.citation.page !== undefined
      ? `Trang: ${item.citation.page}`
      : null,
    item.citation.blockType ? `block: ${item.citation.blockType}` : null,
    item.citation.ocr !== null && item.citation.ocr !== undefined
      ? `OCR: ${item.citation.ocr ? "có" : "không"}`
      : null,
    rerankScore !== null && rerankScore !== undefined
      ? `rerank: ${rerankScore.toFixed(4)}`
      : null,
    score !== null && score !== undefined
      ? `score: ${score.toFixed(4)}`
      : null,
  ].filter(Boolean);

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center bg-black/55 p-3 backdrop-blur-sm"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Ngữ cảnh trích dẫn"
        className="flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl border border-border bg-background shadow-soft"
      >
        <div className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div className="min-w-0">
            <p className="truncate text-xl font-semibold text-foreground">
              {citationDocumentName(item.citation)}
            </p>
            <p className="mt-1 text-sm text-muted-foreground" style={{ overflowWrap: "anywhere" }}>
              Đoạn: {item.citation.chunkId}
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={onClose}
            className="h-10 w-10 shrink-0 rounded-xl"
            title="Đóng"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-4 overflow-y-auto px-5 py-4">
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full border border-border bg-muted px-3 py-1 text-xs text-muted-foreground">
              [{item.index + 1}] • {item.branchLabel}
            </span>
            {metadata.map((meta) => (
              <span
                key={meta}
                className="rounded-full border border-border bg-muted px-3 py-1 text-xs text-muted-foreground"
              >
                {meta}
              </span>
            ))}
          </div>

          <div className="rounded-xl border border-border bg-muted/40 px-4 py-4">
            {context.length > 0 ? (
              <p className="whitespace-pre-wrap text-[15px] leading-9 text-foreground">
                <HighlightedText text={context} query={query} />
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                Không có ngữ cảnh gốc để hiển thị.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
