import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiFeatureUnavailableError,
  ApiRequestError,
  deleteAllDocuments,
  deleteDocument,
  getDocumentStatus,
  listDocuments,
  reindexDocuments as requestReindexDocuments,
  updateChunkingSettings as requestUpdateChunkingSettings,
  updateRetrievalSettings as requestUpdateRetrievalSettings,
  uploadDocument,
} from "@/api/client";
import type {
  ApiChunkConfigMode,
  ApiChunkingMode,
  ApiDocument,
  ApiRetrievalConfigMode,
  ApiRetrievalMode,
} from "@/api/types";
import { getUploadFileValidationError } from "@/lib/upload-files";
import type {
  DocumentRecord,
  DocumentStatus,
  ProcessingStage,
  UploadBatchItem,
} from "@/types/document";

const BACKEND_POLL_INTERVAL_MS = 1400;
const FALLBACK_UPLOAD_STEP_MS = 110;
const FALLBACK_STAGE_STEP_MS = 900;
const FALLBACK_STAGES: ProcessingStage[] = ["splitting", "embedding", "indexing", "ready"];
const UPLOAD_SUCCESS_MESSAGE = "Document uploaded successfully!";

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
  uploadBatchItems: UploadBatchItem[];
  uploadBatchSummary: string | null;
  canQuery: boolean;
  queryDisabledReason: string | null;
  uploadFile: (file: File) => Promise<void>;
  uploadFiles: (files: File[]) => Promise<void>;
  clearAllUploadedDocuments: () => Promise<{ deletedDocuments: number; deletedFiles: number }>;
  deleteUploadedDocument: (documentId: string) => Promise<{ documentId: string; remainingDocuments: number }>;
  reindexDocuments: (payload: {
    mode: ApiChunkingMode;
    chunkSize: number;
    chunkOverlap: number;
  }) => Promise<{
    mode: ApiChunkingMode;
    chunkMode: ApiChunkConfigMode;
    chunkSize: number;
    chunkOverlap: number;
    reindexedDocuments: number;
    activeChunks: number;
  }>;
  updateRetrievalSettings: (payload: {
    mode: ApiRetrievalMode;
    topK: number;
  }) => Promise<{
    mode: ApiRetrievalMode;
    retrievalMode: ApiRetrievalConfigMode;
    topK: number;
    rerankTopN: number;
    contextTopK: number;
  }>;
};

export function useDocumentIngestion(): UseDocumentIngestionResult {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [backendAvailability, setBackendAvailability] = useState<BackendAvailability>("unknown");
  const [isUploading, setIsUploading] = useState(false);
  const [isDeletingDocuments, setIsDeletingDocuments] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadBatchItems, setUploadBatchItems] = useState<UploadBatchItem[]>([]);
  const [uploadBatchSummary, setUploadBatchSummary] = useState<string | null>(null);
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

  const updateUploadBatchItem = useCallback(
    (itemId: string, updater: (previous: UploadBatchItem) => UploadBatchItem) => {
      setUploadBatchItems((previous) =>
        previous.map((item) => (item.id === itemId ? updater(item) : item)),
      );
    },
    [],
  );

  const refreshDocuments = useCallback(async () => {
    try {
      const payload = await listDocuments();
      setDocuments(payload.map(mapApiDocument));
      setBackendAvailability("available");
    } catch (error) {
      if (shouldFallback(error)) {
        setBackendAvailability("unavailable");
      }
    }
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
              setUploadMessage(UPLOAD_SUCCESS_MESSAGE);
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
              message: UPLOAD_SUCCESS_MESSAGE,
              chunkCount: previous.chunkCount ?? 0,
            }));
            setUploadMessage(UPLOAD_SUCCESS_MESSAGE);
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
    async (documentId: string, file: File, options: { finalize?: boolean } = {}) => {
      const shouldFinalize = options.finalize ?? true;
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
          message: stage === "ready" ? UPLOAD_SUCCESS_MESSAGE : previous.message,
        }));
        if (stage !== "ready") {
          await delay(FALLBACK_STAGE_STEP_MS);
        }
      }

      if (shouldFinalize) {
        setUploadMessage(UPLOAD_SUCCESS_MESSAGE);
        setIsUploading(false);
      }
    },
    [updateDocument],
  );

  const uploadFiles = useCallback(
    async (files: File[]) => {
      const selectedFiles = files.filter(Boolean);
      if (selectedFiles.length === 0) {
        return;
      }

      const batchSeed = Date.now();
      let acceptedCount = 0;
      let successCount = 0;
      const initialItems: UploadBatchItem[] = selectedFiles.map((file, index) => {
        const validationError = getUploadFileValidationError(file);
        if (!validationError) {
          acceptedCount += 1;
        }
        return {
          id: `batch-${batchSeed}-${index}-${Math.random().toString(36).slice(2, 7)}`,
          filename: file.name,
          status: validationError ? "error" : "pending",
          progress: validationError ? null : 0,
          message: validationError,
        };
      });

      setUploadBatchItems(initialItems);
      setUploadBatchSummary(null);
      setUploadError(null);
      setUploadMessage(null);

      if (acceptedCount === 0) {
        setUploadError("Không có tệp hợp lệ để tải lên.");
        setUploadBatchSummary(`Đã tải lên 0/${selectedFiles.length} tệp`);
        return;
      }

      setIsUploading(true);
      let shouldUseBackend = backendAvailability !== "unavailable";

      for (const [index, file] of selectedFiles.entries()) {
        const batchItem = initialItems[index];
        if (!batchItem || batchItem.status === "error") {
          continue;
        }

        updateUploadBatchItem(batchItem.id, (previous) => ({
          ...previous,
          status: "uploading",
          progress: 0,
          message: null,
        }));

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

        let uploadedSuccessfully = false;

        if (shouldUseBackend) {
          try {
            const uploaded = await uploadDocument(file, (progressPercent) => {
              updateDocument(temporaryId, (previous) => ({
                ...previous,
                uploadProgress: progressPercent,
              }));
              updateUploadBatchItem(batchItem.id, (previous) => ({
                ...previous,
                progress: progressPercent,
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
            if (mapped.status !== "ready") {
              startBackendPolling(mapped.id);
            }
            uploadedSuccessfully = true;
          } catch (error) {
            if (!shouldFallback(error)) {
              const message = toErrorMessage(error, "Document upload failed.");
              updateDocument(temporaryId, (previous) => ({
                ...previous,
                status: "error",
                stage: "error",
                uploadProgress: null,
                message,
              }));
              updateUploadBatchItem(batchItem.id, (previous) => ({
                ...previous,
                status: "error",
                progress: null,
                message,
              }));
              continue;
            }
            shouldUseBackend = false;
          }
        }

        if (!uploadedSuccessfully) {
          await runFallbackSimulation(temporaryId, file, { finalize: false });
          uploadedSuccessfully = true;
        }

        if (uploadedSuccessfully) {
          successCount += 1;
          updateUploadBatchItem(batchItem.id, (previous) => ({
            ...previous,
            status: "success",
            progress: 100,
            message: "Hoàn tất",
          }));
        }
      }

      await refreshDocuments();
      setIsUploading(false);
      const summary = `Đã tải lên ${successCount}/${selectedFiles.length} tệp`;
      setUploadBatchSummary(summary);
      setUploadMessage(summary);
      if (successCount < acceptedCount) {
        setUploadError(`Có ${acceptedCount - successCount} tệp tải lên thất bại.`);
      }
    },
    [
      backendAvailability,
      refreshDocuments,
      runFallbackSimulation,
      startBackendPolling,
      updateDocument,
      updateUploadBatchItem,
      upsertDocument,
    ],
  );

  const uploadFile = useCallback(
    async (file: File) => {
      await uploadFiles([file]);
    },
    [uploadFiles],
  );

  const clearAllUploadedDocuments = useCallback(async () => {
    setUploadError(null);
    setUploadMessage(null);
    setUploadBatchItems([]);
    setUploadBatchSummary(null);
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

  const reindexDocuments = useCallback(async (payload: { mode: ApiChunkingMode; chunkSize: number; chunkOverlap: number }) => {
    setUploadError(null);
    setUploadMessage(null);
    let appliedMode: ApiChunkingMode = payload.mode;
    let appliedChunkMode: ApiChunkConfigMode = payload.mode === "custom" ? "custom" : "preset";
    let appliedChunkSize = payload.chunkSize;
    let appliedChunkOverlap = payload.chunkOverlap;
    let reindexedDocuments = 0;
    let activeChunks = 0;

    try {
      const updated = await requestUpdateChunkingSettings({
        mode: payload.mode,
        chunk_size: payload.mode === "custom" ? payload.chunkSize : undefined,
        chunk_overlap: payload.mode === "custom" ? payload.chunkOverlap : undefined,
      });
      appliedMode = updated.mode;
      appliedChunkMode = updated.chunk_mode;
      appliedChunkSize = updated.chunk_size;
      appliedChunkOverlap = updated.chunk_overlap;
      reindexedDocuments = updated.reindexed_documents;
      activeChunks = updated.active_chunks;
    } catch (error) {
      // Backward compatibility: fallback to legacy /documents/reindex endpoint.
      if (!(error instanceof ApiFeatureUnavailableError)) {
        throw error;
      }
      const legacy = await requestReindexDocuments({
        chunk_size: payload.chunkSize,
        chunk_overlap: payload.chunkOverlap,
      });
      reindexedDocuments = legacy.reindexed_documents;
      activeChunks = legacy.active_chunks;
      appliedChunkSize = legacy.chunk_size;
      appliedChunkOverlap = legacy.chunk_overlap;
      appliedMode = payload.mode;
      appliedChunkMode = payload.mode === "custom" ? "custom" : "preset";
    }

    try {
      const listed = await listDocuments();
      setDocuments(listed.map(mapApiDocument));
      setBackendAvailability("available");
    } catch (error) {
      if (shouldFallback(error)) {
        setBackendAvailability("unavailable");
      }
    }

    setUploadMessage(
      `Đã re-index với chunk=${appliedChunkSize}/${appliedChunkOverlap} trên ${reindexedDocuments} tài liệu.`,
    );
    return {
      mode: appliedMode,
      chunkMode: appliedChunkMode,
      chunkSize: appliedChunkSize,
      chunkOverlap: appliedChunkOverlap,
      reindexedDocuments,
      activeChunks,
    };
  }, []);

  const updateRetrievalSettings = useCallback(
    async (payload: { mode: ApiRetrievalMode; topK: number }) => {
      setUploadError(null);
      setUploadMessage(null);
      const updated = await requestUpdateRetrievalSettings({
        mode: payload.mode,
        top_k: payload.mode === "custom" ? payload.topK : undefined,
      });
      return {
        mode: updated.mode,
        retrievalMode: updated.retrieval_mode,
        topK: updated.top_k,
        rerankTopN: updated.rerank_top_n,
        contextTopK: updated.context_top_k,
      };
    },
    [],
  );

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
    uploadBatchItems,
    uploadBatchSummary,
    canQuery: queryState.canQuery,
    queryDisabledReason: queryState.reason,
    uploadFile,
    uploadFiles,
    clearAllUploadedDocuments,
    deleteUploadedDocument,
    reindexDocuments,
    updateRetrievalSettings,
  };
}
