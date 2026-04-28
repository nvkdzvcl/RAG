export type ProcessingStage = "splitting" | "embedding" | "indexing" | "ready" | "error";

export type DocumentStatus = "uploading" | "processing" | "ready" | "error";

export type DocumentRecord = {
  id: string;
  filename: string;
  status: DocumentStatus;
  stage: ProcessingStage | null;
  uploadProgress: number | null;
  chunkCount: number | null;
  createdAt: string | null;
  message: string | null;
  source: "backend" | "fallback";
};

export type UploadBatchStatus = "pending" | "uploading" | "success" | "error";

export type UploadBatchItem = {
  id: string;
  filename: string;
  status: UploadBatchStatus;
  progress: number | null;
  message: string | null;
};
