import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { Citation } from "@/types/chat";
import { translations } from "@/lib/translations";

type CitationListProps = {
  citations: Citation[];
  compact?: boolean;
  highlightText?: string;
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTermsFromText(text: string): string[] {
  const matches = text.match(/[\p{L}\p{N}_-]{4,}/gu) ?? [];
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
  return Array.from(new Set(matches.map((item) => item.trim()).filter((item) => item.length >= 4)))
    .filter((item) => !stopwords.has(item.toLowerCase()))
    .sort((left, right) => right.length - left.length)
    .slice(0, 24);
}

function HighlightedText({ text, highlightText }: { text: string; highlightText: string }) {
  const terms = useMemo(() => highlightTermsFromText(highlightText), [highlightText]);
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
          <mark key={`${part}-${index}`} className="rounded bg-amber-200/80 px-0.5 text-slate-900">
            {part}
          </mark>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        ),
      )}
    </>
  );
}

function citationContext(citation: Citation): string {
  return citation.content || citation.text || citation.snippet || "";
}

function CitationPreviewModal({
  citation,
  highlightText,
  onClose,
}: {
  citation: Citation;
  highlightText: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const context = citationContext(citation);
  const metadata = [
    citation.page !== null && citation.page !== undefined ? `${translations.citations.page}: ${citation.page}` : null,
    citation.blockType ? `block: ${citation.blockType}` : null,
    citation.ocr !== null && citation.ocr !== undefined ? `OCR: ${citation.ocr ? "có" : "không"}` : null,
    citation.rerankScore !== null && citation.rerankScore !== undefined
      ? `rerank: ${citation.rerankScore.toFixed(4)}`
      : null,
    citation.score !== null && citation.score !== undefined ? `score: ${citation.score.toFixed(4)}` : null,
  ].filter(Boolean);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-3">
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Ngữ cảnh trích dẫn"
        className="flex max-h-[86vh] w-full max-w-3xl flex-col rounded-xl border border-slate-200 bg-white shadow-xl"
      >
        <div className="flex items-start justify-between gap-3 border-b border-slate-200 px-4 py-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-800">
              {citation.fileName || citation.title || citation.source}
            </p>
            <p className="mt-1 text-xs text-slate-500" style={{ overflowWrap: "anywhere" }}>
              {translations.citations.chunk}: {citation.chunkId}
            </p>
          </div>
          <Button type="button" variant="outline" size="sm" onClick={onClose} className="h-8 w-8 shrink-0 p-0">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-3 overflow-y-auto px-4 py-3">
          {metadata.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {metadata.map((item) => (
                <span key={item} className="rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-600">
                  {item}
                </span>
              ))}
            </div>
          ) : null}

          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
            {context.trim().length > 0 ? (
              <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
                <HighlightedText text={context} highlightText={highlightText} />
              </p>
            ) : (
              <p className="text-sm text-slate-500">Không có ngữ cảnh gốc để hiển thị.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function CitationList({ citations, compact = false, highlightText = "" }: CitationListProps) {
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);

  if (citations.length === 0) {
    return <p className="text-sm text-slate-500">{translations.citations.noCitations}</p>;
  }

  return (
    <>
      <ul className="space-y-2">
        {citations.map((citation) => (
          <li key={citation.id}>
            <button
              type="button"
              onClick={() => setSelectedCitation(citation)}
              className={`w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-left transition hover:border-blue-300 hover:bg-blue-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-200 ${
                compact ? "text-xs" : "text-sm"
              }`}
            >
              <p className="font-medium text-slate-700">{citation.fileName || citation.title || citation.source}</p>
              <p className="mt-0.5 text-slate-500">
                {citation.fileName || citation.source} • {translations.citations.chunk}: {citation.chunkId}
                {citation.section ? ` • ${translations.citations.section}: ${citation.section}` : ""}
                {citation.page !== null && citation.page !== undefined
                  ? ` • ${translations.citations.page}: ${citation.page}`
                  : ""}
              </p>
              <p className="mt-1 text-xs font-medium text-blue-600">Nhấn để xem ngữ cảnh</p>
            </button>
          </li>
        ))}
      </ul>

      {selectedCitation ? (
        <CitationPreviewModal
          citation={selectedCitation}
          highlightText={highlightText}
          onClose={() => setSelectedCitation(null)}
        />
      ) : null}
    </>
  );
}
