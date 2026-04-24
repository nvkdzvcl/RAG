import { useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Circle,
  FileText,
  Loader2,
  UploadCloud,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DocumentRecord, ProcessingStage } from "@/types/document";

type DocumentUploadCardProps = {
  documents: DocumentRecord[];
  activeDocument: DocumentRecord | null;
  isUploading: boolean;
  uploadMessage: string | null;
  uploadError: string | null;
  onUpload: (file: File) => Promise<void>;
};

type StageState = "pending" | "in_progress" | "done";

const STAGES: Array<{ id: ProcessingStage; label: string }> = [
  { id: "splitting", label: "Splitting document" },
  { id: "embedding", label: "Creating embeddings" },
  { id: "indexing", label: "Building index" },
  { id: "ready", label: "Ready" },
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

export function DocumentUploadCard({
  documents,
  activeDocument,
  isUploading,
  uploadMessage,
  uploadError,
  onUpload,
}: DocumentUploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeProgress = activeDocument?.uploadProgress;

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) {
      return;
    }
    await onUpload(files[0]);
  };

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Documents / Upload</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden"
          onChange={(event) => {
            void handleFiles(event.target.files);
            event.currentTarget.value = "";
          }}
        />

        <div
          className={`rounded-xl border border-dashed px-4 py-5 transition ${
            isDragging ? "border-blue-400 bg-blue-50" : "border-slate-300 bg-slate-50"
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            void handleFiles(event.dataTransfer.files);
          }}
        >
          <div className="flex flex-col items-center gap-2 text-center">
            <UploadCloud className="h-6 w-6 text-blue-600" />
            <p className="text-sm font-medium text-slate-700">Drag and drop a PDF document here</p>
            <p className="text-xs text-slate-500">or click browse files</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-1"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              Browse files
            </Button>
          </div>
        </div>

        {typeof activeProgress === "number" && (isUploading || activeDocument?.status !== "ready") ? (
          <div className="space-y-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>Upload progress</span>
              <span>{activeProgress}%</span>
            </div>
            <div className="h-2 rounded-full bg-slate-200">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-blue-600 to-violet-600 transition-all"
                style={{ width: `${Math.max(0, Math.min(100, activeProgress))}%` }}
              />
            </div>
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

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Processing Stages</p>
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
                    {state === "done" ? "done" : state === "in_progress" ? "running" : "pending"}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Processed Documents</p>
          {documents.length === 0 ? (
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
              No documents yet. Upload a PDF to start indexing.
            </div>
          ) : (
            <ul className="max-h-44 space-y-2 overflow-y-auto pr-1">
              {documents.map((document) => (
                <li key={document.id} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <FileText className="h-4 w-4 shrink-0 text-slate-500" />
                      <p className="line-clamp-1 text-sm font-medium text-slate-700">{document.filename}</p>
                    </div>
                    <Badge variant="outline" className={`capitalize ${statusBadgeClass(document.status)}`}>
                      {document.status}
                    </Badge>
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    chunk count: {document.chunkCount ?? "n/a"} • created: {formatCreatedTime(document.createdAt)}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
