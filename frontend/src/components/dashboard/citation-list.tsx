import type { Citation } from "@/types/chat";

type CitationListProps = {
  citations: Citation[];
  compact?: boolean;
};

export function CitationList({ citations, compact = false }: CitationListProps) {
  if (citations.length === 0) {
    return <p className="text-sm text-slate-500">No citations returned.</p>;
  }

  return (
    <ul className="space-y-2">
      {citations.map((citation) => (
        <li
          key={citation.id}
          className={`rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 ${compact ? "text-xs" : "text-sm"}`}
        >
          <p className="font-medium text-slate-700">{citation.title || citation.source}</p>
          <p className="mt-0.5 text-slate-500">
            {citation.source} • chunk: {citation.chunkId}
            {citation.section ? ` • section: ${citation.section}` : ""}
            {citation.page ? ` • page: ${citation.page}` : ""}
          </p>
        </li>
      ))}
    </ul>
  );
}
