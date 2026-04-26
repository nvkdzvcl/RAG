import type { Citation } from "@/types/chat";
import { translations } from "@/lib/translations";

type CitationListProps = {
  citations: Citation[];
  compact?: boolean;
};

export function CitationList({ citations, compact = false }: CitationListProps) {
  if (citations.length === 0) {
    return <p className="text-sm text-slate-500">{translations.citations.noCitations}</p>;
  }

  return (
    <ul className="space-y-2">
      {citations.map((citation) => (
        <li
          key={citation.id}
          className={`rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 ${compact ? "text-xs" : "text-sm"}`}
        >
          <p className="font-medium text-slate-700">{citation.fileName || citation.title || citation.source}</p>
          <p className="mt-0.5 text-slate-500">
            {citation.fileName || citation.source} • {translations.citations.chunk}: {citation.chunkId}
            {citation.section ? ` • ${translations.citations.section}: ${citation.section}` : ""}
            {citation.page ? ` • ${translations.citations.page}: ${citation.page}` : ""}
          </p>
        </li>
      ))}
    </ul>
  );
}
