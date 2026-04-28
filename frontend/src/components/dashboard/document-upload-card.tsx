import { useEffect, useRef, useState } from "react";
import type { DragEvent as ReactDragEvent } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Circle,
  FileText,
  Loader2,
  Trash2,
  UploadCloud,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DocumentRecord, ProcessingStage, UploadBatchItem } from "@/types/document";
import { translations } from "@/lib/translations";
import {
  SUPPORTED_UPLOAD_ACCEPT,
  formatUploadValidationMessages,
  splitValidUploadFiles,
} from "@/lib/upload-files";

type DocumentUploadCardProps = {
  documents: DocumentRecord[];
  activeDocument: DocumentRecord | null;
  isUploading: boolean;
  isDeletingDocuments?: boolean;
  deletingDocumentId?: string | null;
  uploadMessage: string | null;
  uploadError: string | null;
  uploadBatchItems?: UploadBatchItem[];
  uploadBatchSummary?: string | null;
  onUploadFiles: (files: File[]) => Promise<void>;
  onRequestDeleteDocument?: (document: DocumentRecord) => void;
  onRequestDeleteAllDocuments?: () => void;
};

type StageState = "pending" | "in_progress" | "done";

const STAGES: Array<{ id: ProcessingStage; label: string }> = [
  { id: "splitting", label: "Chia nhỏ tài liệu" },
  { id: "embedding", label: "Tạo embeddings" },
  { id: "indexing", label: "Xây dựng chỉ mục" },
  { id: "ready", label: "Sẵn sàng" },
];

function formatCreatedTime(value: string | null): string {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function statusBadgeClass(status: DocumentRecord["status"]): string {
  if (status === "ready") {
    return "border-emerald-300 bg-emerald-50 text-emerald-700";
  }
  if (status === "error") {
    return "border-rose-300 bg-rose-50 text-rose-700";
  }
  if (status === "uploading") {
    return "border-blue-300 bg-blue-50 text-blue-700";
  }
  return "border-violet-300 bg-violet-50 text-violet-700";
}

function uploadBatchStatusLabel(status: UploadBatchItem["status"]): string {
  if (status === "pending") {
    return "pending";
  }
  if (status === "uploading") {
    return "uploading";
  }
  if (status === "success") {
    return "success";
  }
  return "error";
}

function uploadBatchStatusClass(status: UploadBatchItem["status"]): string {
  if (status === "success") {
    return "border-emerald-300 bg-emerald-50 text-emerald-700";
  }
  if (status === "error") {
    return "border-rose-300 bg-rose-50 text-rose-700";
  }
  if (status === "uploading") {
    return "border-blue-300 bg-blue-50 text-blue-700";
  }
  return "border-slate-300 bg-slate-50 text-slate-600";
}

function stageState(active: DocumentRecord | null, stage: ProcessingStage): StageState {
  if (!active) {
    return "pending";
  }

  if (active.status === "ready") {
    return "done";
  }
  if (!active.stage || active.stage === "error") {
    return stage === "splitting" ? "in_progress" : "pending";
  }

  const currentIndex = STAGES.findIndex((item) => item.id === active.stage);
  const index = STAGES.findIndex((item) => item.id === stage);
  if (currentIndex === -1 || index === -1) {
    return "pending";
  }
  if (index < currentIndex) {
    return "done";
  }
  if (index === currentIndex) {
    return "in_progress";
  }
  return "pending";
}

function StageIndicator({ state }: { state: StageState }) {
  if (state === "done") {
    return <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
  }
  if (state === "in_progress") {
    return <Loader2 className="h-4 w-4 animate-spin text-blue-600" />;
  }
  return <Circle className="h-4 w-4 text-slate-400" />;
}

function isFileDrag(event: DragEvent | ReactDragEvent<HTMLElement>): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

export function DocumentUploadCard({
  documents,
  activeDocument,
  isUploading,
  isDeletingDocuments = false,
  deletingDocumentId = null,
  uploadMessage,
  uploadError,
  uploadBatchItems = [],
  uploadBatchSummary = null,
  onUploadFiles,
  onRequestDeleteDocument,
  onRequestDeleteAllDocuments,
}: DocumentUploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileSelectionErrors, setFileSelectionErrors] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);

  const activeProgress = activeDocument?.uploadProgress;

  useEffect(() => {
    const preventPageFileDrop = (event: DragEvent) => {
      if (!isFileDrag(event)) {
        return;
      }
      event.preventDefault();
    };

    window.addEventListener("dragover", preventPageFileDrop);
    window.addEventListener("drop", preventPageFileDrop);
    return () => {
      window.removeEventListener("dragover", preventPageFileDrop);
      window.removeEventListener("drop", preventPageFileDrop);
    };
  }, []);

  const resetDragState = () => {
    dragDepthRef.current = 0;
    setIsDragging(false);
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) {
      return;
    }
    if (isUploading) {
      setFileSelectionErrors(["Đang tải tài liệu. Vui lòng đợi trước khi thêm tệp mới."]);
      return;
    }

    const selectedFiles = Array.from(files);
    const { rejected } = splitValidUploadFiles(selectedFiles);
    setFileSelectionErrors(formatUploadValidationMessages(rejected));

    await onUploadFiles(selectedFiles);
  };

  const handleDragEnter = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current += 1;
    setIsDragging(true);
  };

  const handleDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    setIsDragging(true);
  };

  const handleDragLeave = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    resetDragState();
    void handleFiles(event.dataTransfer.files);
  };

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{translations.upload.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <input
          ref={fileInputRef}
          type="file"
          accept={SUPPORTED_UPLOAD_ACCEPT}
          multiple
          className="hidden"
          onChange={(event) => {
            void handleFiles(event.target.files);
            event.currentTarget.value = "";
          }}
        />

        <div
          className={`rounded-xl border border-dashed px-4 py-5 transition ${
            isDragging ? "border-blue-500 bg-blue-50 ring-2 ring-blue-100" : "border-slate-300 bg-slate-50"
          }`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-2 text-center">
            <UploadCloud className="h-6 w-6 text-blue-600" />
            <p className="text-sm font-medium text-slate-700">Kéo thả tài liệu PDF hoặc DOCX vào đây</p>
            <p className="text-xs text-slate-500">hoặc nhấn để chọn tệp</p>
            <p className="text-xs font-medium text-slate-500">Hỗ trợ PDF, DOCX • Tối đa 50MB mỗi tệp</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-1 bg-accent hover:bg-accent/90"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              {translations.upload.button}
            </Button>
          </div>
        </div>

        {typeof activeProgress === "number" && (isUploading || activeDocument?.status !== "ready") ? (
          <div className="space-y-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>Tiến trình tải lên</span>
              <span>{activeProgress}%</span>
            </div>
            <div className="h-2 rounded-full bg-slate-200">
              <div
                className="h-2 rounded-full bg-primary transition-all"
                style={{ width: `${Math.max(0, Math.min(100, activeProgress))}%` }}
              />
            </div>
          </div>
        ) : null}

        {uploadBatchItems.length > 0 ? (
          <div className="space-y-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Danh sách tải lên</p>
              {uploadBatchSummary ? <p className="text-xs font-medium text-slate-600">{uploadBatchSummary}</p> : null}
            </div>
            <ul className="max-h-32 space-y-1 overflow-y-auto pr-1">
              {uploadBatchItems.map((item) => (
                <li key={item.id} className="flex items-center justify-between gap-2 rounded-md bg-slate-50 px-2 py-1">
                  <div className="min-w-0">
                    <p className="truncate text-xs font-medium text-slate-700">{item.filename}</p>
                    {item.message ? (
                      <p className="truncate text-[11px] text-slate-500">{item.message}</p>
                    ) : item.status === "uploading" && typeof item.progress === "number" ? (
                      <p className="text-[11px] text-slate-500">{item.progress}%</p>
                    ) : null}
                  </div>
                  <Badge variant="outline" className={`shrink-0 ${uploadBatchStatusClass(item.status)}`}>
                    {uploadBatchStatusLabel(item.status)}
                  </Badge>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {uploadMessage ? (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
            {uploadMessage}
          </div>
        ) : null}

        {uploadError ? (
          <div className="flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
            <AlertCircle className="h-4 w-4" />
            <span>{uploadError}</span>
          </div>
        ) : null}

        {fileSelectionErrors.length > 0 ? (
          <div className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1">
              {fileSelectionErrors.map((message) => (
                <p key={message}>{message}</p>
              ))}
            </div>
          </div>
        ) : null}

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Các giai đoạn xử lý</p>
          <ul className="space-y-1.5 rounded-lg border border-slate-200 bg-white px-3 py-3">
            {STAGES.map((stage) => {
              const state = stageState(activeDocument, stage.id);
              return (
                <li key={stage.id} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <StageIndicator state={state} />
                    <span className="text-sm text-slate-700">{stage.label}</span>
                  </div>
                  <span className="text-xs uppercase tracking-wide text-slate-400">
                    {state === "done" ? "hoàn tất" : state === "in_progress" ? "đang chạy" : "chờ"}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Tài liệu đã xử lý</p>
            {onRequestDeleteAllDocuments ? (
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={documents.length === 0 || isDeletingDocuments || isUploading}
                onClick={onRequestDeleteAllDocuments}
                className="h-8 gap-1 border-rose-300 bg-rose-50 px-2 text-rose-700 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span className="text-xs">Xóa tất cả</span>
              </Button>
            ) : null}
          </div>
          {documents.length === 0 ? (
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
              {translations.upload.noDocuments}. {translations.upload.uploadFirst}.
            </div>
          ) : (
            <ul className="max-h-44 space-y-2 overflow-y-auto pr-1">
              {documents.map((document) => {
                const isDeletingThisDocument = deletingDocumentId === document.id;
                const canDelete = !!onRequestDeleteDocument && !isUploading && !isDeletingDocuments;
                return (
                  <li key={document.id} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <FileText className="h-4 w-4 shrink-0 text-slate-500" />
                        <p className="line-clamp-1 text-sm font-medium text-slate-700">{document.filename}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={`capitalize ${statusBadgeClass(document.status)}`}>
                          {document.status === "ready" ? "sẵn sàng" : document.status === "error" ? "lỗi" : document.status === "uploading" ? "đang tải" : "đang xử lý"}
                        </Badge>
                        {onRequestDeleteDocument ? (
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={!canDelete}
                            onClick={() => onRequestDeleteDocument(document)}
                            className="h-8 gap-1 border-rose-300 bg-rose-50 px-2 text-rose-700 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                            title={isDeletingThisDocument ? "Đang xóa tài liệu..." : "Xóa tài liệu"}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            <span className="text-xs">{isDeletingThisDocument ? "Đang xóa" : "Xóa"}</span>
                          </Button>
                        ) : null}
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      số đoạn: {document.chunkCount ?? "n/a"} • tạo lúc: {formatCreatedTime(document.createdAt)}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
