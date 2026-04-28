import { useMemo } from "react";

import type { DocumentRecord } from "@/types/document";

export type QueryFileTypeFilter = "pdf" | "docx";
export type QueryOcrFilter = "all" | "only" | "exclude";

type QueryFiltersProps = {
  documents: DocumentRecord[];
  selectedDocIds: string[];
  onSelectedDocIdsChange: (next: string[]) => void;
  selectedFileTypes: QueryFileTypeFilter[];
  onSelectedFileTypesChange: (next: QueryFileTypeFilter[]) => void;
  uploadedAfter: string;
  onUploadedAfterChange: (next: string) => void;
  uploadedBefore: string;
  onUploadedBeforeChange: (next: string) => void;
  ocrFilter: QueryOcrFilter;
  onOcrFilterChange: (next: QueryOcrFilter) => void;
  disabled?: boolean;
};

const FILE_TYPE_OPTIONS: Array<{ value: QueryFileTypeFilter; label: string }> = [
  { value: "pdf", label: "PDF" },
  { value: "docx", label: "DOCX" },
];

function sortedReadyDocuments(documents: DocumentRecord[]): DocumentRecord[] {
  return [...documents]
    .filter((item) => item.status === "ready")
    .sort((left, right) => left.filename.localeCompare(right.filename));
}

function formatDateLabel(value: string): string {
  const [year, month, day] = value.split("-");
  if (!year || !month || !day) {
    return value;
  }
  return `${day}/${month}/${year}`;
}

export function QueryFilters({
  documents,
  selectedDocIds,
  onSelectedDocIdsChange,
  selectedFileTypes,
  onSelectedFileTypesChange,
  uploadedAfter,
  onUploadedAfterChange,
  uploadedBefore,
  onUploadedBeforeChange,
  ocrFilter,
  onOcrFilterChange,
  disabled = false,
}: QueryFiltersProps) {
  const readyDocuments = useMemo(() => sortedReadyDocuments(documents), [documents]);
  const selectedSet = useMemo(() => new Set(selectedDocIds), [selectedDocIds]);
  const selectedFileTypeSet = useMemo(() => new Set(selectedFileTypes), [selectedFileTypes]);
  const selectedDocuments = useMemo(
    () => readyDocuments.filter((item) => selectedSet.has(item.id)),
    [readyDocuments, selectedSet],
  );

  const hasActiveFilters =
    selectedDocIds.length > 0 ||
    selectedFileTypes.length > 0 ||
    uploadedAfter.length > 0 ||
    uploadedBefore.length > 0 ||
    ocrFilter !== "all";

  const summary = useMemo(() => {
    const documentLabel =
      selectedDocuments.length > 0
        ? selectedDocuments.map((item) => item.filename).join(", ")
        : readyDocuments.length > 0
          ? "tất cả tài liệu đã tải"
          : "chưa có tài liệu sẵn sàng";
    const activeParts: string[] = [];

    if (selectedFileTypes.length > 0) {
      activeParts.push(selectedFileTypes.map((item) => item.toUpperCase()).join(", "));
    }
    if (uploadedAfter && uploadedBefore) {
      activeParts.push(`Từ ${formatDateLabel(uploadedAfter)} đến ${formatDateLabel(uploadedBefore)}`);
    } else if (uploadedAfter) {
      activeParts.push(`Sau ${formatDateLabel(uploadedAfter)}`);
    } else if (uploadedBefore) {
      activeParts.push(`Trước ${formatDateLabel(uploadedBefore)}`);
    }
    if (ocrFilter === "only") {
      activeParts.push("Chỉ dùng OCR");
    } else if (ocrFilter === "exclude") {
      activeParts.push("Không dùng OCR");
    }

    return [`Đang tìm trong: ${documentLabel}`, ...activeParts].join(" • ");
  }, [ocrFilter, readyDocuments.length, selectedDocuments, selectedFileTypes, uploadedAfter, uploadedBefore]);

  const toggleDocument = (documentId: string) => {
    if (selectedSet.has(documentId)) {
      onSelectedDocIdsChange(selectedDocIds.filter((item) => item !== documentId));
      return;
    }
    onSelectedDocIdsChange([...selectedDocIds, documentId]);
  };

  const toggleFileType = (fileType: QueryFileTypeFilter) => {
    if (selectedFileTypeSet.has(fileType)) {
      onSelectedFileTypesChange(selectedFileTypes.filter((item) => item !== fileType));
      return;
    }
    onSelectedFileTypesChange([...selectedFileTypes, fileType]);
  };

  const resetFilters = () => {
    onSelectedDocIdsChange([]);
    onSelectedFileTypesChange([]);
    onUploadedAfterChange("");
    onUploadedBeforeChange("");
    onOcrFilterChange("all");
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white px-3 py-3 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Lọc tài liệu truy vấn</p>
        {hasActiveFilters ? (
          <button
            type="button"
            onClick={resetFilters}
            disabled={disabled}
            className="text-xs font-medium text-slate-600 underline decoration-slate-300 underline-offset-2 hover:text-slate-800 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            Đặt lại
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

      <details className="mt-2 rounded-lg border border-slate-200 bg-slate-50/70 px-2 py-2">
        <summary className="cursor-pointer text-xs font-medium text-slate-700">Bộ lọc nâng cao</summary>

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-500">Loại tệp</p>
            <div className="flex flex-wrap gap-2">
              {FILE_TYPE_OPTIONS.map((option) => (
                <label
                  key={option.value}
                  className="inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-700"
                >
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 accent-blue-600"
                    checked={selectedFileTypeSet.has(option.value)}
                    disabled={disabled}
                    onChange={() => toggleFileType(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-500">OCR</p>
            <select
              value={ocrFilter}
              disabled={disabled}
              onChange={(event) => onOcrFilterChange(event.target.value as QueryOcrFilter)}
              className="h-8 w-full rounded-md border border-slate-200 bg-white px-2 text-xs text-slate-700 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100"
            >
              <option value="all">Tất cả nội dung</option>
              <option value="only">Chỉ dùng OCR text</option>
              <option value="exclude">Bỏ qua OCR text</option>
            </select>
          </div>

          <label className="space-y-1 text-xs text-slate-600">
            <span className="font-semibold text-slate-500">Từ ngày</span>
            <input
              type="date"
              value={uploadedAfter}
              disabled={disabled}
              onChange={(event) => onUploadedAfterChange(event.target.value)}
              className="h-8 w-full rounded-md border border-slate-200 bg-white px-2 text-xs text-slate-700 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100"
            />
          </label>

          <label className="space-y-1 text-xs text-slate-600">
            <span className="font-semibold text-slate-500">Đến ngày</span>
            <input
              type="date"
              value={uploadedBefore}
              disabled={disabled}
              onChange={(event) => onUploadedBeforeChange(event.target.value)}
              className="h-8 w-full rounded-md border border-slate-200 bg-white px-2 text-xs text-slate-700 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100"
            />
          </label>
        </div>
      </details>
    </div>
  );
}
