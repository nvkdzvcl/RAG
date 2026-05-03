import { useEffect } from "react";
import { AlertCircle, FileText, Loader2, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export type DocumentPreviewChunk = {
  chunkId: string;
  text: string;
  branchLabel: string;
  blockType: string | null;
  page: number | null;
};

export type DocumentPreviewStats = {
  chunkCount: number | null;
  estimatedTokens: number | null;
  fileType: string | null;
  uploadedAt: string | null;
};

type DocumentPreviewPanelProps = {
  open: boolean;
  documentName: string;
  chunks: DocumentPreviewChunk[];
  stats: DocumentPreviewStats | null;
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
  onRetry?: () => void;
};

function formatUploadedTime(value: string | null): string {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

export function DocumentPreviewPanel({
  open,
  documentName,
  chunks,
  stats,
  isLoading,
  error,
  onClose,
  onRetry,
}: DocumentPreviewPanelProps) {
  useEffect(() => {
    if (!open) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-[70] bg-background/55 backdrop-blur-[1px]"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`Preview nội dung tài liệu ${documentName}`}
        className="absolute inset-y-0 right-0 flex w-full max-w-2xl flex-col border-l border-border bg-background shadow-soft"
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Preview nội dung</p>
            <p className="truncate text-xs text-muted-foreground">{documentName}</p>
          </div>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="h-8 w-8 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
            onClick={onClose}
            aria-label="Đóng preview tài liệu"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {stats ? (
            <div className="mb-3 grid gap-2 sm:grid-cols-2">
              <div className="rounded-lg border border-border/80 bg-card/75 px-3 py-2">
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Chunks</p>
                <p className="text-sm font-semibold text-foreground">{stats.chunkCount ?? "n/a"}</p>
              </div>
              <div className="rounded-lg border border-border/80 bg-card/75 px-3 py-2">
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Estimated tokens</p>
                <p className="text-sm font-semibold text-foreground">{stats.estimatedTokens ?? "n/a"}</p>
              </div>
              <div className="rounded-lg border border-border/80 bg-card/75 px-3 py-2">
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">File type</p>
                <div className="mt-0.5">
                  <Badge variant="outline" className="border-border/80 bg-muted/55 text-[11px] text-foreground/85">
                    {stats.fileType ?? "n/a"}
                  </Badge>
                </div>
              </div>
              <div className="rounded-lg border border-border/80 bg-card/75 px-3 py-2">
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Upload time</p>
                <p className="text-sm font-semibold text-foreground">{formatUploadedTime(stats.uploadedAt)}</p>
              </div>
            </div>
          ) : null}

          {isLoading ? (
            <div className="flex items-center gap-3 rounded-lg border border-border/80 bg-card/80 px-4 py-4">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <div>
                <p className="text-sm font-semibold text-foreground">Đang tải nội dung trích xuất</p>
                <p className="text-xs text-muted-foreground">Vui lòng chờ trong giây lát...</p>
              </div>
            </div>
          ) : null}

          {!isLoading && error ? (
            <div className="space-y-3 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="mt-0.5 h-4 w-4 text-destructive" />
                <div>
                  <p className="text-sm font-semibold text-foreground">Không thể tải preview tài liệu</p>
                  <p className="text-xs text-foreground/80">{error}</p>
                </div>
              </div>
              <div>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-8 border-border/80 bg-background/80 text-foreground hover:bg-muted/45"
                  onClick={onRetry}
                >
                  Thử lại
                </Button>
              </div>
            </div>
          ) : null}

          {!isLoading && !error && chunks.length === 0 ? (
            <div className="space-y-3 rounded-lg border border-border/80 bg-card/80 px-4 py-5 text-center">
              <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full border border-border/80 bg-muted/35">
                <FileText className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">Chưa có nội dung chunk để preview</p>
                <p className="text-xs text-foreground/80">
                  Hãy chạy truy vấn trước để hệ thống lấy chunk liên quan của tài liệu này.
                </p>
              </div>
            </div>
          ) : null}

          {!isLoading && !error && chunks.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs font-medium text-muted-foreground">
                Hiển thị {chunks.length} chunk nội dung đã trích xuất.
              </p>
              <ul className="space-y-2.5">
                {chunks.map((chunk, index) => (
                  <li
                    key={`${chunk.chunkId}-${chunk.branchLabel}-${index}`}
                    className="rounded-lg border border-border/80 bg-card/70 p-3"
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-1.5">
                      <Badge variant="outline" className="border-border/80 bg-muted/50 text-[11px] text-foreground/85">
                        Chunk {index + 1}
                      </Badge>
                      <Badge variant="outline" className="border-primary/30 bg-primary/10 text-[11px] text-primary">
                        {chunk.branchLabel}
                      </Badge>
                      <span className="truncate text-[11px] text-muted-foreground">{chunk.chunkId}</span>
                      {chunk.blockType ? (
                        <Badge variant="outline" className="border-border/80 bg-muted/50 text-[11px] text-foreground/80">
                          {chunk.blockType}
                        </Badge>
                      ) : null}
                      {typeof chunk.page === "number" ? (
                        <Badge variant="outline" className="border-border/80 bg-muted/50 text-[11px] text-foreground/80">
                          Trang {chunk.page}
                        </Badge>
                      ) : null}
                    </div>
                    <div className="rounded-md border-l-2 border-primary/35 bg-muted/30 px-3 py-2">
                      <p className="whitespace-pre-wrap text-sm leading-6 text-foreground">{chunk.text}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </aside>
    </div>
  );
}
