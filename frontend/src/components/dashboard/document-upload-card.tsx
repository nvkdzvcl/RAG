import { useEffect, useRef, useState } from "react";
import type { DragEvent as ReactDragEvent } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Circle,
  Loader2,
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
  activeDocument: DocumentRecord | null;
  isUploading: boolean;
  uploadMessage: string | null;
  uploadError: string | null;
  uploadBatchItems?: UploadBatchItem[];
  uploadBatchSummary?: string | null;
  onUploadFiles: (files: File[]) => Promise<void>;
};

type StageState = "pending" | "in_progress" | "done";

const STAGES: Array<{ id: ProcessingStage; label: string }> = [
  { id: "splitting", label: "Chia nhỏ tài liệu" },
  { id: "embedding", label: "Tạo embeddings" },
  { id: "indexing", label: "Xây dựng chỉ mục" },
  { id: "ready", label: "Sẵn sàng" },
];
const COMPLETED_STEPS_SUMMARY = `Đã xử lý xong (${STAGES.length} bước)`;

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
    return "border-success/35 bg-success/10 text-success";
  }
  if (status === "error") {
    return "border-destructive/35 bg-destructive/10 text-destructive";
  }
  if (status === "uploading") {
    return "border-primary/35 bg-primary/10 text-primary";
  }
  return "border-border bg-muted text-muted-foreground";
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
    return <CheckCircle2 className="h-4 w-4 text-success" />;
  }
  if (state === "in_progress") {
    return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
  }
  return <Circle className="h-4 w-4 text-muted-foreground" />;
}

function isFileDrag(event: DragEvent | ReactDragEvent<HTMLElement>): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

export function DocumentUploadCard({
  activeDocument,
  isUploading,
  uploadMessage,
  uploadError,
  uploadBatchItems = [],
  uploadBatchSummary = null,
  onUploadFiles,
}: DocumentUploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileSelectionErrors, setFileSelectionErrors] = useState<string[]>([]);
  const [showCompletedSteps, setShowCompletedSteps] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);

  const activeProgress = activeDocument?.uploadProgress;
  const isActivelyProcessing =
    isUploading || activeDocument?.status === "uploading" || activeDocument?.status === "processing";
  const isProcessingComplete = !!activeDocument && activeDocument.status === "ready";
  const shouldRenderProcessingPanel = isActivelyProcessing || isProcessingComplete;
  const showProcessingSteps = isActivelyProcessing || showCompletedSteps;
  const normalizedUploadMessage = uploadMessage?.trim() ?? "";
  const normalizedBatchSummary = uploadBatchSummary?.trim() ?? "";
  const isRedundantUploadSuccessMessage =
    normalizedUploadMessage.length > 0
    && (
      normalizedUploadMessage === normalizedBatchSummary
      || /^Đã tải lên\s+\d+\/\d+\s+tệp$/i.test(normalizedUploadMessage)
    );

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

  useEffect(() => {
    if (isActivelyProcessing) {
      setShowCompletedSteps(true);
      return;
    }
    if (isProcessingComplete) {
      setShowCompletedSteps(false);
      return;
    }
    setShowCompletedSteps(false);
  }, [isActivelyProcessing, isProcessingComplete, activeDocument?.id]);

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
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{translations.upload.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
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
          className={`group flex min-h-[236px] flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-8 text-center transition ${
            isDragging
              ? "border-primary bg-primary/10 shadow-subtle"
              : "border-border bg-background hover:border-primary/40 hover:bg-muted/30"
          } ${isUploading ? "cursor-not-allowed opacity-80" : "cursor-pointer"}`}
          role="button"
          tabIndex={isUploading ? -1 : 0}
          aria-disabled={isUploading}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => {
            if (!isUploading) {
              fileInputRef.current?.click();
            }
          }}
          onKeyDown={(event) => {
            if (isUploading) {
              return;
            }
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              fileInputRef.current?.click();
            }
          }}
        >
          <div className="flex flex-col items-center gap-3 text-center">
            <div className="rounded-2xl border border-border bg-card p-4 shadow-sm transition group-hover:border-primary/35 group-hover:bg-primary/5">
              <UploadCloud className="h-8 w-8 text-primary" />
            </div>
            <p className="text-base font-semibold text-foreground">Kéo thả PDF, DOCX, TXT hoặc MD vào đây</p>
            <p className="text-sm text-muted-foreground">hoặc nhấn để chọn tệp</p>
            <p className="text-xs font-medium text-muted-foreground">Hỗ trợ PDF, DOCX, TXT, MD · Tối đa 50MB mỗi tệp</p>
            <Button
              type="button"
              size="default"
              className="mt-2"
              onClick={(event) => {
                event.stopPropagation();
                fileInputRef.current?.click();
              }}
              disabled={isUploading}
            >
              Chọn tài liệu
            </Button>
          </div>
        </div>

        {shouldRenderProcessingPanel ? (
          <section className="space-y-2 rounded-xl border border-border bg-card px-3 py-2.5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Tiến trình xử lý</p>
                <p className="text-sm text-foreground">
                  {isActivelyProcessing ? "Đang xử lý tài liệu" : COMPLETED_STEPS_SUMMARY}
                </p>
              </div>
              {!isActivelyProcessing ? (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs"
                  onClick={() => setShowCompletedSteps((previous) => !previous)}
                >
                  {showCompletedSteps ? "Thu gọn" : "Xem chi tiết"}
                </Button>
              ) : (
                <Badge variant="outline" className="border-primary/35 bg-primary/10 text-primary">
                  processing
                </Badge>
              )}
            </div>
            {showProcessingSteps ? (
              <div className="space-y-2">
                {typeof activeProgress === "number" && isActivelyProcessing ? (
                  <div className="space-y-2 rounded-lg border border-border bg-background px-3 py-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Tiến trình tải lên</span>
                      <span>{activeProgress}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-muted">
                      <div
                        className="h-2 rounded-full bg-primary transition-all"
                        style={{ width: `${Math.max(0, Math.min(100, activeProgress))}%` }}
                      />
                    </div>
                  </div>
                ) : null}

                <ol className="space-y-2">
                  {STAGES.map((stage) => {
                    const state = stageState(activeDocument, stage.id);
                    return (
                      <li key={stage.id} className="flex items-center justify-between rounded-lg border border-border bg-background px-3 py-1.5">
                        <div className="flex items-center gap-2">
                          <StageIndicator state={state} />
                          <span className="text-sm text-foreground">{stage.label}</span>
                        </div>
                        <span className="text-xs uppercase tracking-wide text-muted-foreground">
                          {state === "done" ? "complete" : state === "in_progress" ? "processing" : "pending"}
                        </span>
                      </li>
                    );
                  })}
                </ol>
              </div>
            ) : null}
          </section>
        ) : null}

        {uploadBatchItems.length > 0 ? (
          <div className="space-y-2 rounded-lg border border-border bg-background px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Danh sách tải lên</p>
              {uploadBatchSummary ? <p className="text-xs font-medium text-muted-foreground">{uploadBatchSummary}</p> : null}
            </div>
            <ul className="space-y-1">
              {uploadBatchItems.map((item) => (
                <li key={item.id} className="flex items-center justify-between gap-2 rounded-md bg-muted/40 px-2 py-1">
                  <div className="min-w-0">
                    <p className="truncate text-xs font-medium text-foreground">{item.filename}</p>
                    {item.message ? (
                      <p className="truncate text-[11px] text-muted-foreground">{item.message}</p>
                    ) : item.status === "uploading" && typeof item.progress === "number" ? (
                      <p className="text-[11px] text-muted-foreground">{item.progress}%</p>
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

        {uploadMessage && !isRedundantUploadSuccessMessage ? (
          <div className="rounded-lg border border-success/35 bg-success/10 px-3 py-2 text-sm text-success">
            {uploadMessage}
          </div>
        ) : null}

        {uploadError ? (
          <div className="flex items-center gap-2 rounded-lg border border-destructive/35 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            <span>{uploadError}</span>
          </div>
        ) : null}

        {fileSelectionErrors.length > 0 ? (
          <div className="flex gap-2 rounded-lg border border-warning/35 bg-warning/10 px-3 py-2 text-sm text-warning">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1">
              {fileSelectionErrors.map((message) => (
                <p key={message}>{message}</p>
              ))}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
