import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import {
  AlertCircle,
  FileText,
  FileUp,
  Loader2,
  MoreHorizontal,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DocumentRecord } from "@/types/document";
import type { QueryFileTypeFilter, QueryOcrFilter } from "@/components/dashboard/query-filters";

type DocumentsManagementPanelProps = {
  documents: DocumentRecord[];
  isDocumentsLoading: boolean;
  documentsError: string | null;
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
  isUploading: boolean;
  isDeletingDocuments?: boolean;
  deletingDocumentId?: string | null;
  onRetryLoadDocuments?: () => void;
  onGoToUploadTab?: () => void;
  onRequestDeleteDocument?: (document: DocumentRecord) => void;
  onRequestDeleteSelected?: (documentIds: string[]) => void;
  onRequestViewDetails?: (document: DocumentRecord) => void;
  onRequestPreviewDocument?: (document: DocumentRecord) => void;
  onRequestViewChunks?: (document: DocumentRecord) => void;
};

const FILE_TYPE_OPTIONS: Array<{ value: QueryFileTypeFilter; label: string }> = [
  { value: "pdf", label: "PDF" },
  { value: "docx", label: "DOCX" },
];
const DOCUMENT_ROW_SKELETON_COUNT = 6;

function DocumentRowSkeleton({ rowIndex }: { rowIndex: number }) {
  return (
    <li className="border-t border-border/70 first:border-t-0">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <div className="h-4 w-4 animate-pulse rounded-sm border border-border/80 bg-muted/50" />
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <div className="h-4 w-4 animate-pulse rounded bg-muted/50" />
          <div className="min-w-0 flex-1 space-y-1.5">
            <div
              className="h-3.5 animate-pulse rounded bg-muted/55"
              style={{ width: rowIndex % 2 === 0 ? "58%" : "44%" }}
            />
            <div
              className="h-3 animate-pulse rounded bg-muted/45"
              style={{ width: rowIndex % 3 === 0 ? "36%" : "30%" }}
            />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <div className="h-5 w-12 animate-pulse rounded-full border border-border/80 bg-muted/45" />
          <div className="h-5 w-16 animate-pulse rounded-full border border-border/80 bg-muted/45" />
        </div>
      </div>
    </li>
  );
}

function formatCreatedTime(value: string | null): string {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear()
    && date.getMonth() === now.getMonth()
    && date.getDate() === now.getDate();
  const timePart = date.toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
  });
  if (sameDay) {
    return timePart;
  }
  const datePart = date.toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
  });
  return `${datePart} ${timePart}`;
}

function fileTypeFromFilename(filename: string): QueryFileTypeFilter | "other" {
  const normalized = filename.toLowerCase();
  if (normalized.endsWith(".pdf")) {
    return "pdf";
  }
  if (normalized.endsWith(".docx")) {
    return "docx";
  }
  return "other";
}

function statusLabel(status: DocumentRecord["status"]): string {
  if (status === "ready") {
    return "Sẵn sàng";
  }
  if (status === "error") {
    return "Lỗi";
  }
  if (status === "uploading") {
    return "Đang tải";
  }
  return "Đang xử lý";
}

function statusBadgeClass(status: DocumentRecord["status"]): string {
  if (status === "ready") {
    return "border-success/35 bg-success/10 text-success";
  }
  if (status === "error") {
    return "border-destructive/35 bg-destructive/10 text-destructive";
  }
  if (status === "uploading") {
    return "border-primary/35 bg-primary/10 text-primary";
  }
  return "border-warning/35 bg-warning/10 text-warning";
}

export function DocumentsManagementPanel({
  documents,
  isDocumentsLoading,
  documentsError,
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
  isUploading,
  isDeletingDocuments = false,
  deletingDocumentId = null,
  onRetryLoadDocuments,
  onGoToUploadTab,
  onRequestDeleteDocument,
  onRequestDeleteSelected,
  onRequestViewDetails,
  onRequestPreviewDocument,
  onRequestViewChunks,
}: DocumentsManagementPanelProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [openMenuDocId, setOpenMenuDocId] = useState<string | null>(null);
  const [lastSelectedReadyIndex, setLastSelectedReadyIndex] = useState<number | null>(
    null,
  );
  const menuContainerRef = useRef<HTMLDivElement | null>(null);
  const menuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const menuItemRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selectAllCheckboxRef = useRef<HTMLInputElement | null>(null);

  const selectedSet = useMemo(() => new Set(selectedDocIds), [selectedDocIds]);
  const readySet = useMemo(
    () => new Set(documents.filter((item) => item.status === "ready").map((item) => item.id)),
    [documents],
  );

  const filteredDocuments = useMemo(() => {
    const search = searchQuery.trim().toLowerCase();
    return [...documents]
      .filter((item) => {
        if (!search) {
          return true;
        }
        return item.filename.toLowerCase().includes(search);
      })
      .filter((item) => {
        if (selectedFileTypes.length === 0) {
          return true;
        }
        const type = fileTypeFromFilename(item.filename);
        return type !== "other" && selectedFileTypes.includes(type);
      })
      .filter((item) => {
        if (!uploadedAfter && !uploadedBefore) {
          return true;
        }
        if (!item.createdAt) {
          return false;
        }
        const created = new Date(item.createdAt);
        if (Number.isNaN(created.getTime())) {
          return false;
        }
        if (uploadedAfter) {
          const from = new Date(`${uploadedAfter}T00:00:00`);
          if (created < from) {
            return false;
          }
        }
        if (uploadedBefore) {
          const to = new Date(`${uploadedBefore}T23:59:59.999`);
          if (created > to) {
            return false;
          }
        }
        return true;
      })
      .sort((left, right) => {
        const leftTime = left.createdAt ? new Date(left.createdAt).getTime() : 0;
        const rightTime = right.createdAt ? new Date(right.createdAt).getTime() : 0;
        return rightTime - leftTime;
      });
  }, [documents, searchQuery, selectedFileTypes, uploadedAfter, uploadedBefore]);

  const visibleSelectableIds = useMemo(
    () => filteredDocuments.filter((item) => item.status === "ready").map((item) => item.id),
    [filteredDocuments],
  );
  const visibleSelectableDocuments = useMemo(
    () => filteredDocuments.filter((item) => item.status === "ready"),
    [filteredDocuments],
  );
  const visibleIndexByDocumentId = useMemo(() => {
    const mapping = new Map<string, number>();
    visibleSelectableDocuments.forEach((item, index) => {
      mapping.set(item.id, index);
    });
    return mapping;
  }, [visibleSelectableDocuments]);

  const selectedReadyIds = useMemo(
    () => selectedDocIds.filter((id) => readySet.has(id)),
    [selectedDocIds, readySet],
  );

  const selectedCount = selectedReadyIds.length;
  const selectedVisibleCount = visibleSelectableIds.filter((id) => selectedSet.has(id)).length;
  const allVisibleSelected =
    visibleSelectableIds.length > 0 && visibleSelectableIds.every((id) => selectedSet.has(id));
  const someVisibleSelected = selectedVisibleCount > 0 && !allVisibleSelected;

  const canDelete = !disabled && !isUploading && !isDeletingDocuments;
  const hasActiveFilters =
    selectedFileTypes.length > 0 || uploadedAfter.length > 0 || uploadedBefore.length > 0 || ocrFilter !== "all";
  const activeFilterCount =
    (selectedFileTypes.length > 0 ? 1 : 0)
    + (ocrFilter !== "all" ? 1 : 0)
    + (uploadedAfter.length > 0 ? 1 : 0)
    + (uploadedBefore.length > 0 ? 1 : 0);

  const toggleDocument = (
    document: DocumentRecord,
    options?: {
      shiftKey?: boolean;
    },
  ) => {
    if (document.status !== "ready" || disabled) {
      return;
    }

    const currentVisibleIndex = visibleIndexByDocumentId.get(document.id);
    const canRangeSelect =
      options?.shiftKey
      && typeof currentVisibleIndex === "number"
      && typeof lastSelectedReadyIndex === "number";

    if (canRangeSelect) {
      const rangeStart = Math.min(lastSelectedReadyIndex, currentVisibleIndex);
      const rangeEnd = Math.max(lastSelectedReadyIndex, currentVisibleIndex);
      const rangeDocumentIds = visibleSelectableDocuments
        .slice(rangeStart, rangeEnd + 1)
        .map((item) => item.id);
      const shouldSelectRange = !selectedSet.has(document.id);
      const nextSelected = new Set(selectedDocIds);
      rangeDocumentIds.forEach((docId) => {
        if (shouldSelectRange) {
          nextSelected.add(docId);
          return;
        }
        nextSelected.delete(docId);
      });
      onSelectedDocIdsChange(Array.from(nextSelected));
      setLastSelectedReadyIndex(currentVisibleIndex);
      return;
    }

    if (selectedSet.has(document.id)) {
      onSelectedDocIdsChange(selectedDocIds.filter((item) => item !== document.id));
      if (typeof currentVisibleIndex === "number") {
        setLastSelectedReadyIndex(currentVisibleIndex);
      }
      return;
    }
    onSelectedDocIdsChange([...selectedDocIds, document.id]);
    if (typeof currentVisibleIndex === "number") {
      setLastSelectedReadyIndex(currentVisibleIndex);
    }
  };

  const handleSelectAll = () => {
    if (disabled || visibleSelectableIds.length === 0) {
      return;
    }
    if (allVisibleSelected) {
      const hidden = selectedDocIds.filter((id) => !visibleSelectableIds.includes(id));
      onSelectedDocIdsChange(hidden);
      return;
    }
    const merged = [...selectedDocIds];
    for (const id of visibleSelectableIds) {
      if (!selectedSet.has(id)) {
        merged.push(id);
      }
    }
    onSelectedDocIdsChange(merged);
  };

  useEffect(() => {
    if (!selectAllCheckboxRef.current) {
      return;
    }
    selectAllCheckboxRef.current.indeterminate = someVisibleSelected;
  }, [someVisibleSelected]);

  const toggleFileType = (fileType: QueryFileTypeFilter) => {
    if (selectedFileTypes.includes(fileType)) {
      onSelectedFileTypesChange(selectedFileTypes.filter((item) => item !== fileType));
      return;
    }
    onSelectedFileTypesChange([...selectedFileTypes, fileType]);
  };

  const resetFilters = () => {
    onSelectedFileTypesChange([]);
    onUploadedAfterChange("");
    onUploadedBeforeChange("");
    onOcrFilterChange("all");
  };

  const focusMenuItem = (index: number) => {
    const totalItems = menuItemRefs.current.length;
    if (totalItems === 0) {
      return;
    }
    const nextIndex = (index + totalItems) % totalItems;
    menuItemRefs.current[nextIndex]?.focus();
  };

  const openContextMenu = (documentId: string) => {
    setOpenMenuDocId(documentId);
    window.requestAnimationFrame(() => {
      focusMenuItem(0);
    });
  };

  const handleMenuKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const activeElement = document.activeElement;
    const currentIndex = menuItemRefs.current.findIndex((item) => item === activeElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      focusMenuItem(currentIndex + 1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      focusMenuItem(currentIndex - 1);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusMenuItem(0);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      focusMenuItem(menuItemRefs.current.length - 1);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setOpenMenuDocId(null);
      menuTriggerRef.current?.focus();
    }
  };

  const handleCopyFilename = async (filename: string) => {
    try {
      await navigator.clipboard.writeText(filename);
    } catch {
      // Clipboard may be unavailable; keep silent fallback.
    } finally {
      setOpenMenuDocId(null);
    }
  };

  useEffect(() => {
    if (!openMenuDocId) {
      return;
    }

    const handlePointerDown = (event: MouseEvent | TouchEvent) => {
      const target = event.target as Node | null;
      if (!target) {
        return;
      }
      if (menuContainerRef.current?.contains(target) || menuTriggerRef.current?.contains(target)) {
        return;
      }
      setOpenMenuDocId(null);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      setOpenMenuDocId(null);
      menuTriggerRef.current?.focus();
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("touchstart", handlePointerDown);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("touchstart", handlePointerDown);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [openMenuDocId]);

  if (isDocumentsLoading && documents.length === 0) {
    return (
      <section className="rounded-xl border border-border/80 bg-card/90 p-5 shadow-sm">
        <div className="mb-3 flex items-center gap-2 text-foreground/80">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          <p className="text-sm font-semibold text-foreground">Đang tải danh sách tài liệu</p>
        </div>
        <ul
          aria-hidden="true"
          className="overflow-hidden rounded-xl border border-border/80 bg-background/70"
        >
          {Array.from({ length: DOCUMENT_ROW_SKELETON_COUNT }, (_, rowIndex) => (
            <DocumentRowSkeleton key={`document-skeleton-${rowIndex}`} rowIndex={rowIndex} />
          ))}
        </ul>
      </section>
    );
  }

  if (documentsError && documents.length === 0) {
    return (
      <section className="rounded-xl border border-border/80 bg-card/90 p-5 shadow-sm">
        <div className="space-y-3 rounded-lg bg-muted/35 px-4 py-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="mt-0.5 h-5 w-5 text-destructive" />
            <div>
              <p className="text-sm font-semibold text-foreground">Không thể tải danh sách tài liệu</p>
              <p className="text-xs text-foreground/70">{documentsError}</p>
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            className="h-9 gap-2 border-border/80 bg-background/80 text-foreground hover:bg-muted/40"
            onClick={onRetryLoadDocuments}
          >
            <RefreshCw className="h-4 w-4" />
            Thử lại
          </Button>
        </div>
      </section>
    );
  }

  if (documents.length === 0) {
    return (
      <section className="rounded-xl border border-border/80 bg-card/90 p-5 shadow-sm">
        <div className="space-y-3 rounded-lg bg-muted/35 px-4 py-5 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-border/80 bg-background/80">
            <FileUp className="h-6 w-6 text-primary" />
          </div>
          <div>
            <p className="text-sm font-semibold text-foreground">Chưa có tài liệu nào</p>
            <p className="text-xs text-foreground/70">Hãy tải lên PDF, DOCX, TXT hoặc MD để bắt đầu truy vấn.</p>
          </div>
          <div className="flex justify-center">
            <Button type="button" className="h-9" onClick={onGoToUploadTab}>
              Chuyển sang tab Upload
            </Button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="space-y-5 rounded-xl border border-border/80 bg-card/90 p-4 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <label className="flex h-9 flex-1 items-center gap-2 rounded-lg border border-border bg-background px-3">
          <Search className="h-4 w-4 text-muted-foreground" />
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Tìm tài liệu..."
            className="w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
          />
        </label>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            className="h-9 gap-2 border-border/80 bg-background/80 text-foreground hover:bg-muted/50"
            onClick={() => setShowFilters((previous) => !previous)}
          >
            <SlidersHorizontal className="h-4 w-4" />
            Lọc
            {activeFilterCount > 0 ? (
              <span className="inline-flex min-w-5 items-center justify-center rounded-full border border-primary/30 bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold leading-none text-primary">
                {activeFilterCount}
              </span>
            ) : null}
          </Button>
          {hasActiveFilters ? (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-9 px-2 text-xs text-foreground/80 hover:bg-muted/40 hover:text-foreground"
              onClick={resetFilters}
            >
              Xóa bộ lọc
            </Button>
          ) : null}
        </div>
      </div>

      {showFilters ? (
        <div className="space-y-3 rounded-lg bg-muted/35 px-3 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-foreground/80">Bộ lọc truy vấn</p>
            {hasActiveFilters ? (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="h-7 px-2 text-xs text-foreground/85 hover:bg-muted/40 hover:text-foreground"
                onClick={resetFilters}
              >
                Đặt lại
              </Button>
            ) : null}
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <p className="text-xs font-semibold text-foreground/85">Loại tệp</p>
              <div className="flex flex-wrap gap-2">
                {FILE_TYPE_OPTIONS.map((option) => (
                  <label
                    key={option.value}
                    className="inline-flex items-center gap-2 rounded-md border border-border/80 bg-background px-2 py-1 text-xs text-foreground"
                  >
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                      checked={selectedFileTypes.includes(option.value)}
                      disabled={disabled}
                      onChange={() => toggleFileType(option.value)}
                    />
                    {option.label}
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs font-semibold text-foreground/85">OCR</p>
              <select
                value={ocrFilter}
                disabled={disabled}
                onChange={(event) => onOcrFilterChange(event.target.value as QueryOcrFilter)}
                className="h-8 w-full rounded-md border border-border/80 bg-background px-2 text-xs text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 disabled:cursor-not-allowed disabled:bg-muted"
              >
                <option value="all">Tất cả nội dung</option>
                <option value="only">Chỉ dùng OCR text</option>
                <option value="exclude">Bỏ qua OCR text</option>
              </select>
            </div>

            <label className="space-y-1 text-xs text-foreground/85">
              <span className="font-semibold">Từ ngày</span>
              <input
                type="date"
                value={uploadedAfter}
                disabled={disabled}
                onChange={(event) => onUploadedAfterChange(event.target.value)}
                className="h-8 w-full rounded-md border border-border/80 bg-background px-2 text-xs text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 disabled:cursor-not-allowed disabled:bg-muted"
              />
            </label>

            <label className="space-y-1 text-xs text-foreground/85">
              <span className="font-semibold">Đến ngày</span>
              <input
                type="date"
                value={uploadedBefore}
                disabled={disabled}
                onChange={(event) => onUploadedBeforeChange(event.target.value)}
                className="h-8 w-full rounded-md border border-border/80 bg-background px-2 text-xs text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 disabled:cursor-not-allowed disabled:bg-muted"
              />
            </label>
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-2 border-y border-border/80 py-3">
        <div className="flex items-center gap-3">
          <label className="inline-flex items-center gap-2 text-sm text-foreground">
            <input
              ref={selectAllCheckboxRef}
              type="checkbox"
              checked={allVisibleSelected}
              disabled={disabled || visibleSelectableIds.length === 0}
              onChange={handleSelectAll}
              aria-label="Chọn tất cả tài liệu đang hiển thị"
              className="h-4 w-4 accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            />
            <span className="font-medium">Select all</span>
          </label>
          <p className="text-sm font-medium text-foreground">{selectedCount} selected</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-8 gap-1 border-border/80 bg-background/80 text-foreground/80 hover:bg-muted/45 hover:text-foreground"
            onClick={() => onRequestDeleteSelected?.(selectedReadyIds)}
            disabled={!canDelete || selectedReadyIds.length === 0}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Xóa đã chọn
          </Button>
        </div>
      </div>

      {isDocumentsLoading ? (
        <div className="flex items-center gap-2 rounded-md bg-muted/35 px-3 py-2 text-xs text-foreground/75">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
          Đang cập nhật danh sách tài liệu...
        </div>
      ) : null}

      {documentsError ? (
        <div className="flex items-start justify-between gap-3 rounded-md bg-destructive/10 px-3 py-2">
          <div className="flex min-w-0 items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
            <p className="text-xs text-foreground/80">{documentsError}</p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-7 shrink-0 px-2 text-xs text-foreground/80 hover:bg-muted/40 hover:text-foreground"
            onClick={onRetryLoadDocuments}
          >
            Thử lại
          </Button>
        </div>
      ) : null}

      {filteredDocuments.length === 0 ? (
        <div className="rounded-lg bg-muted/35 px-4 py-5 text-center">
          <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full border border-border/80 bg-background/80">
            <Search className="h-4 w-4 text-foreground/60" />
          </div>
          <p className="text-sm font-semibold text-foreground">Không tìm thấy tài liệu phù hợp</p>
          <p className="mt-1 text-xs text-foreground/70">Thử đổi từ khóa tìm kiếm hoặc nới bộ lọc hiện tại.</p>
          {(searchQuery.trim().length > 0 || hasActiveFilters) ? (
            <div className="mt-3 flex justify-center gap-2">
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-8 border-border/80 bg-background/80 text-foreground/85 hover:bg-muted/40 hover:text-foreground"
                onClick={() => setSearchQuery("")}
              >
                Xóa tìm kiếm
              </Button>
              {hasActiveFilters ? (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-8 text-foreground/85 hover:bg-muted/40 hover:text-foreground"
                  onClick={resetFilters}
                >
                  Xóa bộ lọc
                </Button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : (
        <ul className="overflow-hidden rounded-xl border border-border/80 bg-background/70">
          {filteredDocuments.map((document) => {
            const checked = selectedSet.has(document.id);
            const ready = document.status === "ready";
            const type = fileTypeFromFilename(document.filename);
            const isDeletingThisDocument = deletingDocumentId === document.id;
            const rowInteractive = ready && !disabled;
            const isMenuOpen = openMenuDocId === document.id;

            return (
              <li key={document.id} className="border-t border-border/70 first:border-t-0">
                <div
                  role="checkbox"
                  aria-checked={checked}
                  tabIndex={rowInteractive ? 0 : -1}
                  onClick={(event) => {
                    toggleDocument(document, { shiftKey: event.shiftKey });
                  }}
                  onKeyDown={(event) => {
                    if (!rowInteractive) {
                      return;
                    }
                    if (event.target !== event.currentTarget) {
                      return;
                    }
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      toggleDocument(document, { shiftKey: event.shiftKey });
                    }
                  }}
                  className={`flex items-center gap-3 px-4 py-2.5 transition-colors duration-150 ease-out ${
                    checked
                      ? "bg-primary/12 ring-1 ring-inset ring-primary/40 hover:bg-primary/15"
                      : "bg-transparent hover:bg-muted/45"
                  } ${rowInteractive ? "cursor-pointer" : "cursor-default"}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!rowInteractive}
                    readOnly
                    onClick={(event) => {
                      event.stopPropagation();
                      toggleDocument(document, { shiftKey: event.shiftKey });
                    }}
                    aria-label={`Chọn tài liệu ${document.filename}`}
                    className="h-4 w-4 accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  />

                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-foreground">{document.filename}</p>
                      <p className="text-xs text-foreground/75">
                        {document.chunkCount ?? "n/a"} chunks · {formatCreatedTime(document.createdAt)}
                      </p>
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-1.5">
                    <Badge variant="outline" className="border-border/80 bg-muted/55 text-foreground/75">
                      {type === "other" ? "FILE" : type.toUpperCase()}
                    </Badge>
                    <Badge variant="outline" className={statusBadgeClass(document.status)}>
                      {statusLabel(document.status)}
                    </Badge>
                    {onRequestDeleteDocument ? (
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 text-muted-foreground hover:bg-muted/45 hover:text-foreground"
                        title={isDeletingThisDocument ? "Đang xóa tài liệu..." : "Xóa tài liệu"}
                        aria-label={
                          isDeletingThisDocument
                            ? `Đang xóa tài liệu ${document.filename}`
                            : `Xóa tài liệu ${document.filename}`
                        }
                        disabled={!canDelete}
                        onClick={(event) => {
                          event.stopPropagation();
                          onRequestDeleteDocument(document);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    ) : null}
                    <div className="relative">
                      <Button
                        ref={(node) => {
                          if (isMenuOpen) {
                            menuTriggerRef.current = node;
                          }
                        }}
                        type="button"
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 text-muted-foreground hover:bg-muted/45 hover:text-foreground"
                        title="Tùy chọn khác"
                        aria-label={`Tùy chọn khác cho ${document.filename}`}
                        aria-haspopup="menu"
                        aria-expanded={isMenuOpen}
                        aria-controls={`document-menu-${document.id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          if (isMenuOpen) {
                            setOpenMenuDocId(null);
                            return;
                          }
                          openContextMenu(document.id);
                        }}
                        onKeyDown={(event) => {
                          event.stopPropagation();
                          if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            openContextMenu(document.id);
                          }
                        }}
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                      {isMenuOpen ? (
                        <div
                          ref={(node) => {
                            if (isMenuOpen) {
                              menuContainerRef.current = node;
                            }
                          }}
                          id={`document-menu-${document.id}`}
                          role="menu"
                          aria-label={`Menu cho ${document.filename}`}
                          className="absolute right-0 top-[calc(100%+6px)] z-30 w-44 rounded-lg border border-border bg-card py-1 shadow-subtle"
                          onClick={(event) => event.stopPropagation()}
                          onKeyDown={handleMenuKeyDown}
                        >
                          <button
                            ref={(node) => {
                              menuItemRefs.current[0] = node;
                            }}
                            type="button"
                            role="menuitem"
                            className="flex w-full items-center px-3 py-2 text-left text-sm text-foreground hover:bg-muted/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() => {
                              setOpenMenuDocId(null);
                              onRequestViewDetails?.(document);
                            }}
                          >
                            Xem chi tiết
                          </button>
                          <button
                            ref={(node) => {
                              menuItemRefs.current[1] = node;
                            }}
                            type="button"
                            role="menuitem"
                            className="flex w-full items-center px-3 py-2 text-left text-sm text-foreground hover:bg-muted/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() => {
                              setOpenMenuDocId(null);
                              onRequestPreviewDocument?.(document);
                            }}
                          >
                            Preview nội dung
                          </button>
                          <button
                            ref={(node) => {
                              menuItemRefs.current[2] = node;
                            }}
                            type="button"
                            role="menuitem"
                            className="flex w-full items-center px-3 py-2 text-left text-sm text-foreground hover:bg-muted/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() => {
                              setOpenMenuDocId(null);
                              onRequestViewChunks?.(document);
                            }}
                          >
                            Xem chunks
                          </button>
                          <button
                            ref={(node) => {
                              menuItemRefs.current[3] = node;
                            }}
                            type="button"
                            role="menuitem"
                            className="flex w-full items-center px-3 py-2 text-left text-sm text-foreground hover:bg-muted/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() => {
                              void handleCopyFilename(document.filename);
                            }}
                          >
                            Copy tên file
                          </button>
                          <div className="my-1 h-px bg-border" />
                          <button
                            ref={(node) => {
                              menuItemRefs.current[4] = node;
                            }}
                            type="button"
                            role="menuitem"
                            className="flex w-full items-center px-3 py-2 text-left text-sm text-destructive hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() => {
                              setOpenMenuDocId(null);
                              onRequestDeleteDocument?.(document);
                            }}
                          >
                            Xóa
                          </button>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
