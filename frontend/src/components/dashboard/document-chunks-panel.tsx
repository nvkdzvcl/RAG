import { useEffect, useMemo, useState } from "react";
import { AlertCircle, FileText, Loader2, Search, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DocumentPreviewChunk } from "@/components/dashboard/document-preview-panel";

type DocumentChunksPanelProps = {
  open: boolean;
  documentName: string;
  chunks: DocumentPreviewChunk[];
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
  onRetry?: () => void;
};

function chunkCharLength(text: string): number {
  return text.trim().length;
}

export function DocumentChunksPanel({
  open,
  documentName,
  chunks,
  isLoading,
  error,
  onClose,
  onRetry,
}: DocumentChunksPanelProps) {
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    if (!open) {
      setSearchQuery("");
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

  const normalizedSearch = searchQuery.trim().toLowerCase();
  const filteredChunks = useMemo(() => {
    if (!normalizedSearch) {
      return chunks;
    }
    return chunks.filter((chunk) => {
      return (
        chunk.chunkId.toLowerCase().includes(normalizedSearch)
        || chunk.branchLabel.toLowerCase().includes(normalizedSearch)
        || (chunk.blockType ?? "").toLowerCase().includes(normalizedSearch)
        || chunk.text.toLowerCase().includes(normalizedSearch)
      );
    });
  }, [chunks, normalizedSearch]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-[72] bg-background/55 backdrop-blur-[1px]"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`Danh sách chunks tài liệu ${documentName}`}
        className="absolute inset-y-0 right-0 flex w-full max-w-3xl flex-col border-l border-border bg-background shadow-soft"
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Xem chunks</p>
            <p className="truncate text-xs text-muted-foreground">{documentName}</p>
          </div>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="h-8 w-8 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
            onClick={onClose}
            aria-label="Đóng danh sách chunks"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="border-b border-border px-4 py-3">
          <label className="flex h-9 items-center gap-2 rounded-lg border border-border bg-card/70 px-3">
            <Search className="h-4 w-4 text-muted-foreground" />
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Tìm trong chunks..."
              className="w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
            />
          </label>
          <p className="mt-2 text-xs text-muted-foreground">
            {normalizedSearch ? `Kết quả: ${filteredChunks.length}/${chunks.length} chunks` : `${chunks.length} chunks`}
          </p>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {isLoading ? (
            <div className="flex items-center gap-3 rounded-lg border border-border/80 bg-card/80 px-4 py-4">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <div>
                <p className="text-sm font-semibold text-foreground">Đang tải danh sách chunks</p>
                <p className="text-xs text-muted-foreground">Vui lòng chờ trong giây lát...</p>
              </div>
            </div>
          ) : null}

          {!isLoading && error ? (
            <div className="space-y-3 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="mt-0.5 h-4 w-4 text-destructive" />
                <div>
                  <p className="text-sm font-semibold text-foreground">Không thể tải chunks</p>
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
                <p className="text-sm font-semibold text-foreground">Chưa có chunks để hiển thị</p>
                <p className="text-xs text-foreground/80">
                  Hãy chạy truy vấn trước để lấy chunks liên quan của tài liệu này.
                </p>
              </div>
            </div>
          ) : null}

          {!isLoading && !error && chunks.length > 0 && filteredChunks.length === 0 ? (
            <div className="space-y-3 rounded-lg border border-border/80 bg-card/80 px-4 py-5 text-center">
              <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full border border-border/80 bg-muted/35">
                <Search className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">Không có chunk phù hợp</p>
                <p className="text-xs text-foreground/80">Thử từ khóa khác để lọc chunks.</p>
              </div>
            </div>
          ) : null}

          {!isLoading && !error && filteredChunks.length > 0 ? (
            <ol className="space-y-2.5">
              {filteredChunks.map((chunk, index) => (
                <li
                  key={`${chunk.chunkId}-${chunk.branchLabel}-${index}`}
                  className="rounded-lg border border-border/80 bg-card/70 p-3"
                >
                  <div className="mb-2 flex flex-wrap items-center gap-1.5">
                    <Badge variant="outline" className="border-border/80 bg-muted/50 text-[11px] text-foreground/85">
                      Chunk #{index + 1}
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
                    <Badge variant="outline" className="border-border/80 bg-muted/50 text-[11px] text-foreground/80">
                      {chunkCharLength(chunk.text)} ký tự
                    </Badge>
                  </div>
                  <p className="line-clamp-6 whitespace-pre-wrap text-sm leading-6 text-foreground">{chunk.text}</p>
                </li>
              ))}
            </ol>
          ) : null}
        </div>
      </aside>
    </div>
  );
}
