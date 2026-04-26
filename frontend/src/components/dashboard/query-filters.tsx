import { useMemo } from "react";

import type { DocumentRecord } from "@/types/document";

type QueryFiltersProps = {
  documents: DocumentRecord[];
  selectedDocIds: string[];
  onSelectedDocIdsChange: (next: string[]) => void;
  includeOcrOnly: boolean;
  onIncludeOcrOnlyChange: (next: boolean) => void;
  disabled?: boolean;
};

function sortedReadyDocuments(documents: DocumentRecord[]): DocumentRecord[] {
  return [...documents]
    .filter((item) => item.status === "ready")
    .sort((left, right) => left.filename.localeCompare(right.filename));
}

export function QueryFilters({
  documents,
  selectedDocIds,
  onSelectedDocIdsChange,
  includeOcrOnly,
  onIncludeOcrOnlyChange,
  disabled = false,
}: QueryFiltersProps) {
  const readyDocuments = useMemo(() => sortedReadyDocuments(documents), [documents]);
  const selectedSet = useMemo(() => new Set(selectedDocIds), [selectedDocIds]);
  const selectedDocuments = useMemo(
    () => readyDocuments.filter((item) => selectedSet.has(item.id)),
    [readyDocuments, selectedSet],
  );

  const summary =
    selectedDocuments.length > 0
      ? `Đang tìm trong: ${selectedDocuments.map((item) => item.filename).join(", ")}`
      : readyDocuments.length > 0
        ? "Đang tìm trong: tất cả tài liệu đã tải"
        : "Chưa có tài liệu sẵn sàng để lọc.";

  const toggleDocument = (documentId: string) => {
    if (selectedSet.has(documentId)) {
      onSelectedDocIdsChange(selectedDocIds.filter((item) => item !== documentId));
      return;
    }
    onSelectedDocIdsChange([...selectedDocIds, documentId]);
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white px-3 py-3 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Lọc tài liệu truy vấn</p>
        {selectedDocIds.length > 0 ? (
          <button
            type="button"
            onClick={() => onSelectedDocIdsChange([])}
            disabled={disabled}
            className="text-xs font-medium text-slate-600 underline decoration-slate-300 underline-offset-2 hover:text-slate-800 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            Bỏ chọn
          </button>
        ) : null}
      </div>

      <p className="mt-1 text-xs text-slate-600">{summary}</p>

      {readyDocuments.length > 0 ? (
        <details className="mt-2 rounded-lg border border-slate-200 bg-slate-50/70 px-2 py-2">
          <summary className="cursor-pointer text-xs font-medium text-slate-700">
            Chọn tài liệu ({selectedDocuments.length}/{readyDocuments.length})
          </summary>
          <div className="mt-2 max-h-28 space-y-1 overflow-y-auto pr-1">
            {readyDocuments.map((document) => {
              const checked = selectedSet.has(document.id);
              return (
                <label
                  key={document.id}
                  className="flex items-center gap-2 rounded-md px-2 py-1 text-xs text-slate-700 hover:bg-slate-100"
                >
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 accent-blue-600"
                    checked={checked}
                    disabled={disabled}
                    onChange={() => toggleDocument(document.id)}
                  />
                  <span className="truncate">{document.filename}</span>
                </label>
              );
            })}
          </div>
        </details>
      ) : null}

      <label className="mt-2 inline-flex items-center gap-2 text-xs text-slate-700">
        <input
          type="checkbox"
          className="h-3.5 w-3.5 accent-blue-600"
          checked={includeOcrOnly}
          disabled={disabled}
          onChange={(event) => onIncludeOcrOnlyChange(event.target.checked)}
        />
        Chỉ tìm trong nội dung OCR
      </label>
    </div>
  );
}
