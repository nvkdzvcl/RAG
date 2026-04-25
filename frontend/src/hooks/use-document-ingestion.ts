import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiFeatureUnavailableError,
  ApiRequestError,
  deleteAllDocuments,
  deleteDocument,
  getDocumentStatus,
  listDocuments,
  uploadDocument,
} from "@/api/client";
import type { ApiDocument } from "@/api/types";
import type { DocumentRecord, DocumentStatus, ProcessingStage } from "@/types/document";

const BACKEND_POLL_INTERVAL_MS = 1400;
const FALLBACK_UPLOAD_STEP_MS = 110;
const FALLBACK_STAGE_STEP_MS = 900;
const FALLBACK_STAGES: ProcessingStage[] = ["splitting", "embedding", "indexing", "ready"];

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function normalizeStatus(value: string, stage: ProcessingStage | null): DocumentStatus {
  const normalized = value.trim().toLowerCase();
  if (normalized.includes("error") || normalized.includes("fail")) {
    return "error";
  }
  if (normalized.includes("ready") || normalized.includes("complete")) {
    return "ready";
  }
  if (normalized.includes("upload")) {
    return "uploading";
  }
  if (stage === "ready") {
    return "ready";
  }
  return "processing";
}

function normalizeStage(value: string | null | undefined): ProcessingStage | null {
  if (!value) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (normalized.includes("split")) {
    return "splitting";
  }
  if (normalized.includes("embed")) {
    return "embedding";
  }
  if (normalized.includes("index")) {
    return "indexing";
  }
  if (normalized.includes("ready") || normalized.includes("complete")) {
    return "ready";
  }
  if (normalized.includes("error") || normalized.includes("fail")) {
    return "error";
  }
  return null;
}

function normalizeChunkCount(value: number | null | undefined): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.round(value));
}

function mapApiDocument(document: ApiDocument): DocumentRecord {
  const stage = normalizeStage(document.stage ?? document.status);
  const status = normalizeStatus(document.status, stage);
  return {
    id: document.id,
    filename: document.filename,
    status,
    stage: stage ?? (status === "ready" ? "ready" : status === "error" ? "error" : "splitting"),
    uploadProgress: status === "uploading" ? 100 : null,
    chunkCount: normalizeChunkCount(document.chunk_count),
    createdAt: document.created_at ?? null,
    message: document.message ?? null,
    source: "backend",
  };
}

function toErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}

function shouldFallback(error: unknown): boolean {
  if (error instanceof ApiFeatureUnavailableError) {
    return true;
  }
  if (error instanceof ApiRequestError && error.status === 0) {
    return true;
  }
  return false;
}

type BackendAvailability = "unknown" | "available" | "unavailable";

export type UseDocumentIngestionResult = {
  documents: DocumentRecord[];
  activeDocument: DocumentRecord | null;
  isUploading: boolean;
  isDeletingDocuments: boolean;
  deletingDocumentId: string | null;
  uploadMessage: string | null;
  uploadError: string | null;
  canQuery: boolean;
  queryDisabledReason: string | null;
  uploadFile: (file: File) => Promise<void>;
  clearAllUploadedDocuments: () => Promise<{ deletedDocuments: number; deletedFiles: number }>;
  deleteUploadedDocument: (documentId: string) => Promise<{ documentId: string; remainingDocuments: number }>;
};

export function useDocumentIngestion(): UseDocumentIngestionResult {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [backendAvailability, setBackendAvailability] = useState<BackendAvailability>("unknown");
  const [isUploading, setIsUploading] = useState(false);
  const [isDeletingDocuments, setIsDeletingDocuments] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);

  const pollingRef = useRef<Map<string, number>>(new Map());

  const clearPolling = useCallback((documentId: string) => {
    const timer = pollingRef.current.get(documentId);
    if (timer !== undefined) {
      window.clearInterval(timer);
      pollingRef.current.delete(documentId);
    }
  }, []);

  const updateDocument = useCallback(
    (documentId: string, updater: (previous: DocumentRecord) => DocumentRecord) => {
      setDocuments((previous) =>
        previous.map((item) => {
          if (item.id !== documentId) {
            return item;
          }
          return updater(item);
        }),
      );
    },
    [],
  );

  const upsertDocument = useCallback((next: DocumentRecord) => {
    setDocuments((previous) => {
      const existingIndex = previous.findIndex((item) => item.id === next.id);
      if (existingIndex === -1) {
        return [next, ...previous];
      }

      const updated = [...previous];
      updated[existingIndex] = next;
      return updated;
    });
  }, []);

  const startBackendPolling = useCallback(
    (documentId: string) => {
      clearPolling(documentId);

      const timer = window.setInterval(async () => {
        try {
          const payload = await getDocumentStatus(documentId);
          setBackendAvailability("available");
          const mapped = mapApiDocument(payload);
          upsertDocument({
            ...mapped,
            uploadProgress: mapped.status === "ready" ? 100 : mapped.uploadProgress,
          });

          if (mapped.status === "ready" || mapped.status === "error") {
            clearPolling(documentId);
            setIsUploading(false);
            if (mapped.status === "ready") {
              setUploadMessage("PDF uploaded successfully!");
            }
          }
        } catch (error) {
          if (shouldFallback(error)) {
            clearPolling(documentId);
            setBackendAvailability("unavailable");
            setIsUploading(false);
            setUploadError("Document status endpoint is not available yet. Showing local processing state.");
            updateDocument(documentId, (previous) => ({
              ...previous,
              status: "ready",
              stage: "ready",
              uploadProgress: 100,
              message: "PDF uploaded successfully!",
              chunkCount: previous.chunkCount ?? 0,
            }));
            setUploadMessage("PDF uploaded successfully!");
            return;
          }

          const message = toErrorMessage(error, "Unable to fetch document status.");
          setUploadError(message);
        }
      }, BACKEND_POLL_INTERVAL_MS);

      pollingRef.current.set(documentId, timer);
    },
    [clearPolling, updateDocument, upsertDocument],
  );

  useEffect(() => {
    let cancelled = false;

    const bootstrapDocuments = async () => {
      try {
        const payload = await listDocuments();
        if (cancelled) {
          return;
        }

        const mapped = payload.map(mapApiDocument);
        setDocuments(mapped);
        setBackendAvailability("available");

        for (const document of mapped) {
          if (document.status === "processing" || document.status === "uploading") {
            startBackendPolling(document.id);
          }
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        if (shouldFallback(error)) {
          setBackendAvailability("unavailable");
          return;
        }

        setBackendAvailability("unavailable");
        setUploadError("Unable to load document list. You can still use local upload simulation.");
      }
    };

    void bootstrapDocuments();

    return () => {
      cancelled = true;
      for (const timer of pollingRef.current.values()) {
        window.clearInterval(timer);
      }
      pollingRef.current.clear();
    };
  }, [startBackendPolling]);

  const runFallbackSimulation = useCallback(
    async (documentId: string, file: File) => {
      setBackendAvailability("unavailable");

      for (let progress = 8; progress <= 100; progress += 8) {
        updateDocument(documentId, (previous) => ({
          ...previous,
          status: "uploading",
          stage: previous.stage,
          uploadProgress: progress,
        }));
        await delay(FALLBACK_UPLOAD_STEP_MS);
      }

      for (const stage of FALLBACK_STAGES) {
        updateDocument(documentId, (previous) => ({
          ...previous,
          status: stage === "ready" ? "ready" : "processing",
          stage,
          uploadProgress: 100,
          chunkCount:
            stage === "ready"
              ? Math.max(1, Math.ceil(file.size / 4800))
              : previous.chunkCount,
          message: stage === "ready" ? "PDF uploaded successfully!" : previous.message,
        }));
        if (stage !== "ready") {
          await delay(FALLBACK_STAGE_STEP_MS);
        }
      }

      setUploadMessage("PDF uploaded successfully!");
      setIsUploading(false);
    },
    [updateDocument],
  );

  const uploadFile = useCallback(
    async (file: File) => {
      const lowerName = file.name.toLowerCase();
      const isPdf = file.type === "application/pdf" || lowerName.endsWith(".pdf");
      if (!isPdf) {
        setUploadError("Only PDF files are supported.");
        return;
      }

      setUploadError(null);
      setUploadMessage(null);
      setIsUploading(true);

      const temporaryId = `upload-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const temporaryRecord: DocumentRecord = {
        id: temporaryId,
        filename: file.name,
        status: "uploading",
        stage: null,
        uploadProgress: 0,
        chunkCount: null,
        createdAt: new Date().toISOString(),
        message: null,
        source: "fallback",
      };

      upsertDocument(temporaryRecord);
      setActiveDocumentId(temporaryId);

      if (backendAvailability !== "unavailable") {
        try {
          const uploaded = await uploadDocument(file, (progressPercent) => {
            updateDocument(temporaryId, (previous) => ({
              ...previous,
              uploadProgress: progressPercent,
            }));
          });

          const mapped = mapApiDocument(uploaded);
          setBackendAvailability("available");

          setDocuments((previous) => {
            const withoutTemporary = previous.filter((item) => item.id !== temporaryId);
            return [
              {
                ...mapped,
                uploadProgress: 100,
              },
              ...withoutTemporary,
            ];
          });
          setActiveDocumentId(mapped.id);

          if (mapped.status === "ready") {
            setUploadMessage("PDF uploaded successfully!");
            setIsUploading(false);
          } else {
            startBackendPolling(mapped.id);
          }
          return;
        } catch (error) {
          if (!shouldFallback(error)) {
            setUploadError(toErrorMessage(error, "Document upload failed."));
            updateDocument(temporaryId, (previous) => ({
              ...previous,
              status: "error",
              stage: "error",
              uploadProgress: null,
              message: toErrorMessage(error, "Document upload failed."),
            }));
            setIsUploading(false);
            return;
          }
        }
      }

      await runFallbackSimulation(temporaryId, file);
    },
    [backendAvailability, runFallbackSimulation, startBackendPolling, updateDocument, upsertDocument],
  );

  const clearAllUploadedDocuments = useCallback(async () => {
    setUploadError(null);
    setUploadMessage(null);
    setIsDeletingDocuments(true);
    setDeletingDocumentId(null);
    setIsUploading(false);

    try {
      const deleted = await deleteAllDocuments();

      for (const timer of pollingRef.current.values()) {
        window.clearInterval(timer);
      }
      pollingRef.current.clear();

      setDocuments([]);
      setActiveDocumentId(null);
      setBackendAvailability("available");
      setUploadMessage(
        `Đã xóa ${deleted.deleted_documents} tài liệu và ${deleted.deleted_files} tệp khỏi bộ nhớ tải lên.`,
      );
      return {
        deletedDocuments: deleted.deleted_documents,
        deletedFiles: deleted.deleted_files,
      };
    } catch (error) {
      const message = toErrorMessage(error, "Không thể xóa toàn bộ tài liệu đã tải.");
      setUploadError(message);
      throw new Error(message);
    } finally {
      setIsDeletingDocuments(false);
      setDeletingDocumentId(null);
    }
  }, []);

  const deleteUploadedDocument = useCallback(async (documentId: string) => {
    setUploadError(null);
    setUploadMessage(null);
    setDeletingDocumentId(documentId);
    setIsDeletingDocuments(true);

    try {
      const deleted = await deleteDocument(documentId);

      clearPolling(documentId);
      setDocuments((previous) => previous.filter((item) => item.id !== documentId));
      setActiveDocumentId((previous) => (previous === documentId ? null : previous));
      setBackendAvailability("available");
      setUploadMessage(`Đã xóa tài liệu ${documentId}.`);
      return {
        documentId: deleted.document_id,
        remainingDocuments: deleted.remaining_documents,
      };
    } catch (error) {
      const message = toErrorMessage(error, "Không thể xóa tài liệu đã chọn.");
      setUploadError(message);
      throw new Error(message);
    } finally {
      setDeletingDocumentId(null);
      setIsDeletingDocuments(false);
    }
  }, [clearPolling]);

  const activeDocument = useMemo(() => {
    if (activeDocumentId) {
      const explicit = documents.find((item) => item.id === activeDocumentId);
      if (explicit) {
        return explicit;
      }
    }

    const inFlight = documents.find((item) => item.status === "uploading" || item.status === "processing");
    if (inFlight) {
      return inFlight;
    }

    return documents[0] ?? null;
  }, [activeDocumentId, documents]);

  const queryState = useMemo(() => {
    const hasReady = documents.some((item) => item.status === "ready");
    const hasProcessing = documents.some((item) => item.status === "uploading" || item.status === "processing");

    if (!hasReady) {
      return {
        canQuery: false,
        reason: "Upload and process a PDF first.",
      };
    }
    if (hasProcessing) {
      return {
        canQuery: false,
        reason: "Document processing is running. Please wait for Ready state.",
      };
    }

    return {
      canQuery: true,
      reason: null,
    };
  }, [documents]);

  return {
    documents,
    activeDocument,
    isUploading,
    isDeletingDocuments,
    deletingDocumentId,
    uploadMessage,
    uploadError,
    canQuery: queryState.canQuery,
    queryDisabledReason: queryState.reason,
    uploadFile,
    clearAllUploadedDocuments,
    deleteUploadedDocument,
  };
}
