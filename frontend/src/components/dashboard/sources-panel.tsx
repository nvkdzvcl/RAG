import type { CompareResult, ModeResult, QueryResult, SourceReference } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { translations } from "@/lib/translations";

type SourcesPanelProps = {
  result: QueryResult | null;
};

function filenameFromSourcePath(path: string): string {
  const trimmed = path.trim();
  if (!trimmed) {
    return "unknown";
  }
  const normalized = trimmed.replace(/\\/g, "/").replace(/\/+$/, "");
  const candidate = normalized.split("/").filter(Boolean).pop();
  return candidate && candidate.length > 0 ? candidate : trimmed;
}

function sourcePrimaryTitle(source: SourceReference): string {
  if (source.fileName && source.fileName.trim().length > 0) {
    return source.fileName.trim();
  }
  if (source.source && source.source.trim().length > 0) {
    return filenameFromSourcePath(source.source);
  }
  if (source.title && source.title.trim().length > 0) {
    return filenameFromSourcePath(source.title);
  }
  return "unknown";
}

function sourceSecondaryMeta(source: SourceReference): string {
  const parts: string[] = [];
  if (source.page) {
    parts.push(`${translations.citations.page} ${source.page}`);
  }
  if (source.section) {
    parts.push(source.section);
  }
  if (source.rerankScore !== null && source.rerankScore !== undefined) {
    parts.push(`${translations.sources.rerank} ${source.rerankScore.toFixed(4)}`);
  }
  return parts.join(" • ");
}

function sourceDetailRows(source: SourceReference): string[] {
  const rows: string[] = [];
  if (source.fileName && source.fileName.trim().length > 0) {
    rows.push(`Tệp: ${source.fileName}`);
  }
  if (source.title && source.title.trim().length > 0) {
    rows.push(`Tiêu đề: ${source.title}`);
  }
  if (source.page) {
    rows.push(`${translations.citations.page}: ${source.page}`);
  }
  if (source.section) {
    rows.push(`${translations.citations.section}: ${source.section}`);
  }
  if (source.rerankScore !== null && source.rerankScore !== undefined) {
    rows.push(`${translations.sources.rerank}: ${source.rerankScore.toFixed(4)}`);
  }
  if (source.blockType) {
    rows.push(`Loại nội dung: ${source.blockType}`);
  }
  return rows;
}

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
          <p className="truncate text-sm font-medium text-slate-700">
            {sourcePrimaryTitle(source)}
          </p>
          {sourceSecondaryMeta(source) ? (
            <p className="mt-1 text-xs text-slate-500" style={{ overflowWrap: "anywhere", wordBreak: "break-word" }}>
              {sourceSecondaryMeta(source)}
            </p>
          ) : null}
          <details className="mt-1 rounded border border-slate-100 bg-slate-50 px-2 py-1">
            <summary className="cursor-pointer text-[11px] text-slate-500">Chi tiết nguồn</summary>
            <div className="mt-1 space-y-1 text-[11px] text-slate-500">
              {sourceDetailRows(source).map((row) => (
                <p key={row} style={{ overflowWrap: "anywhere", wordBreak: "break-word" }}>
                  {row}
                </p>
              ))}
            </div>
          </details>
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
