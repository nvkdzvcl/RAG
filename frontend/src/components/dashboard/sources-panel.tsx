import type { CompareResult, ModeResult, QueryResult, SourceReference } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { translations } from "@/lib/translations";

type SourcesPanelProps = {
  result: QueryResult | null;
};

function SourceList({ sources }: { sources: SourceReference[] }) {
  if (sources.length === 0) {
    return <p className="text-sm text-slate-500">{translations.sources.noSources}</p>;
  }

  return (
    <ul className="space-y-2 min-w-0 max-w-full">
      {sources.map((source) => (
        <li
          key={source.id}
          className="min-w-0 max-w-full overflow-hidden rounded-lg border border-slate-200 bg-white px-3 py-2"
        >
          <p
            className="truncate text-sm font-medium text-slate-700"
            title={source.title || source.source}
          >
            {source.title || source.source}
          </p>
          <p
            className="mt-1 text-xs text-slate-500"
            title={source.source}
            style={{ overflowWrap: "anywhere", wordBreak: "break-word" }}
          >
            {source.source} • {translations.citations.chunk} {source.chunkId}
            {source.docId ? ` • doc ${source.docId}` : ""}
            {source.section ? ` • ${source.section}` : ""}
            {source.page ? ` • ${translations.citations.page} ${source.page}` : ""}
            {source.rerankScore !== null && source.rerankScore !== undefined
              ? ` • ${translations.sources.rerank} ${source.rerankScore.toFixed(4)}`
              : ""}
          </p>
        </li>
      ))}
    </ul>
  );
}

function ModeSources({ label, modeResult }: { label: string; modeResult: ModeResult }) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <SourceList sources={modeResult.sources} />
    </div>
  );
}

function isCompare(result: QueryResult): result is CompareResult {
  return result.mode === "compare";
}

export function SourcesPanel({ result }: SourcesPanelProps) {
  return (
    <Card className="min-w-0 max-w-full border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{translations.sources.title}</CardTitle>
      </CardHeader>
      <CardContent className="min-w-0 max-w-full space-y-3 overflow-x-hidden">
        {!result ? <p className="text-sm text-slate-500">{translations.sources.runQuery}</p> : null}

        {result && !isCompare(result) ? <SourceList sources={result.sources} /> : null}

        {result && isCompare(result) ? (
          <div className="grid gap-3 xl:grid-cols-2">
            <ModeSources label={translations.compare.standard} modeResult={result.standard} />
            <ModeSources label={translations.compare.advanced} modeResult={result.advanced} />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
