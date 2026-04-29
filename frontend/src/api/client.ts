import type {
  ApiChunkingMode,
  ApiChunkConfigMode,
  ApiRetrievalMode,
  ApiRetrievalConfigMode,
  ApiDeleteAllDocumentsResponse,
  ApiDeleteDocumentResponse,
  ApiDocument,
  ApiDocumentStatusResponse,
  ApiHealthResponse,
  ApiUpdateChunkingSettingsRequest,
  ApiUpdateChunkingSettingsResponse,
  ApiUpdateRetrievalSettingsRequest,
  ApiUpdateRetrievalSettingsResponse,
  ApiReindexDocumentsRequest,
  ApiReindexDocumentsResponse,
  ApiQueryRequest,
  ApiQueryResponse,
  ApiUploadDocumentResponse,
} from "@/api/types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");
const QUERY_STREAM_ENABLED = String(import.meta.env.VITE_QUERY_STREAM_ENABLED ?? "true").toLowerCase() === "true";
const DOCUMENTS_BASE_URL = `${API_BASE_URL}/documents`;
const DOCUMENTS_UPLOAD_URL = `${DOCUMENTS_BASE_URL}/upload`;
const SETTINGS_BASE_URL = `${API_BASE_URL}/settings`;

export class ApiRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(`API error (${status}): ${message}`);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

export class ApiFeatureUnavailableError extends ApiRequestError {
  constructor(status: number, message: string) {
    super(status, message);
    this.name = "ApiFeatureUnavailableError";
  }
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      return payload.detail;
    }
  } catch {
    return response.statusText || "Request failed";
  }
  return response.statusText || "Request failed";
}

function isFeatureUnavailableStatus(status: number): boolean {
  return status === 404 || status === 405 || status === 501;
}

async function parseJsonResponse(response: Response): Promise<unknown> {
  if (response.status === 204) {
    return null;
  }
  return (await response.json()) as unknown;
}

async function handleApiFailure(response: Response): Promise<never> {
  const message = await parseError(response);
  if (isFeatureUnavailableStatus(response.status)) {
    throw new ApiFeatureUnavailableError(response.status, message);
  }
  throw new ApiRequestError(response.status, message);
}

function supportsStreamingResponse(): boolean {
  return typeof ReadableStream !== "undefined" && typeof TextDecoder !== "undefined";
}

function sanitizeStreamErrorMessage(raw: unknown): string {
  if (typeof raw !== "string") {
    return "Yêu cầu streaming thất bại.";
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return "Yêu cầu streaming thất bại.";
  }
  if (trimmed.includes("Traceback") || trimmed.includes("\n") || trimmed.length > 260) {
    return "Không thể hoàn tất yêu cầu streaming. Vui lòng thử lại.";
  }
  return trimmed;
}

function nowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalString(value: unknown): string | null | undefined {
  if (value === null) {
    return null;
  }
  return typeof value === "string" ? value : undefined;
}

function optionalNumber(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function normalizeDocument(payload: unknown): ApiDocument {
  if (!isObject(payload)) {
    throw new Error("Invalid document payload");
  }

  const idSource = payload.id ?? payload.document_id;
  const filenameSource = payload.filename ?? payload.name;

  return {
    id: typeof idSource === "string" && idSource.length > 0 ? idSource : `doc-${Date.now()}`,
    filename:
      typeof filenameSource === "string" && filenameSource.length > 0 ? filenameSource : "uploaded-document.pdf",
    status: typeof payload.status === "string" ? payload.status : "processing",
    stage: optionalString(payload.stage ?? payload.processing_stage),
    chunk_count: optionalNumber(payload.chunk_count ?? payload.chunks),
    created_at: optionalString(payload.created_at ?? payload.created_time),
    message: optionalString(payload.message),
  };
}

function normalizeDocumentList(payload: unknown): ApiDocument[] {
  if (payload === null) {
    return [];
  }

  if (Array.isArray(payload)) {
    return payload.map(normalizeDocument);
  }

  if (isObject(payload) && Array.isArray(payload.documents)) {
    return payload.documents.map(normalizeDocument);
  }

  throw new Error("Invalid documents list payload");
}

function normalizeDeleteAllDocumentsResponse(payload: unknown): ApiDeleteAllDocumentsResponse {
  if (!isObject(payload)) {
    throw new Error("Invalid delete-all documents payload");
  }

  const deletedDocuments = optionalNumber(payload.deleted_documents);
  const deletedFiles = optionalNumber(payload.deleted_files);

  return {
    status: payload.status === "deleted" ? "deleted" : "deleted",
    deleted_documents: typeof deletedDocuments === "number" ? deletedDocuments : 0,
    deleted_files: typeof deletedFiles === "number" ? deletedFiles : 0,
  };
}

function normalizeDeleteDocumentResponse(payload: unknown): ApiDeleteDocumentResponse {
  if (!isObject(payload)) {
    throw new Error("Invalid delete document payload");
  }

  const documentId = typeof payload.document_id === "string" ? payload.document_id : "";
  const remainingDocuments = optionalNumber(payload.remaining_documents);
  const deletedFiles = optionalNumber(payload.deleted_files);

  if (!documentId) {
    throw new Error("Invalid delete document payload: missing document_id");
  }

  return {
    status: payload.status === "deleted" ? "deleted" : "deleted",
    document_id: documentId,
    remaining_documents: typeof remainingDocuments === "number" ? remainingDocuments : 0,
    deleted_files: typeof deletedFiles === "number" ? deletedFiles : 0,
  };
}

function normalizeReindexDocumentsResponse(payload: unknown): ApiReindexDocumentsResponse {
  if (!isObject(payload)) {
    throw new Error("Invalid reindex payload");
  }

  const chunkSize = optionalNumber(payload.chunk_size);
  const chunkOverlap = optionalNumber(payload.chunk_overlap);
  const reindexedDocuments = optionalNumber(payload.reindexed_documents);
  const activeChunks = optionalNumber(payload.active_chunks);

  return {
    status: payload.status === "reindexed" ? "reindexed" : "reindexed",
    chunk_size: typeof chunkSize === "number" ? chunkSize : 0,
    chunk_overlap: typeof chunkOverlap === "number" ? chunkOverlap : 0,
    reindexed_documents: typeof reindexedDocuments === "number" ? reindexedDocuments : 0,
    active_chunks: typeof activeChunks === "number" ? activeChunks : 0,
  };
}

function normalizeChunkingMode(value: unknown): ApiChunkingMode {
  if (value === "small" || value === "medium" || value === "large" || value === "custom") {
    return value;
  }
  return "custom";
}

function normalizeChunkConfigMode(value: unknown): ApiChunkConfigMode {
  if (value === "preset" || value === "custom") {
    return value;
  }
  return "custom";
}

function normalizeRetrievalMode(value: unknown): ApiRetrievalMode {
  if (value === "low" || value === "balanced" || value === "high" || value === "custom") {
    return value;
  }
  return "custom";
}

function normalizeRetrievalConfigMode(value: unknown): ApiRetrievalConfigMode {
  if (value === "preset" || value === "custom") {
    return value;
  }
  return "custom";
}

function normalizeUpdateChunkingSettingsResponse(payload: unknown): ApiUpdateChunkingSettingsResponse {
  if (!isObject(payload)) {
    throw new Error("Invalid settings/chunking payload");
  }

  const chunkSize = optionalNumber(payload.chunk_size);
  const chunkOverlap = optionalNumber(payload.chunk_overlap);
  const reindexedDocuments = optionalNumber(payload.reindexed_documents);
  const activeChunks = optionalNumber(payload.active_chunks);

  return {
    status: payload.status === "reindexed" ? "reindexed" : "reindexed",
    mode: normalizeChunkingMode(payload.mode),
    chunk_mode: normalizeChunkConfigMode(payload.chunk_mode),
    chunk_size: typeof chunkSize === "number" ? chunkSize : 0,
    chunk_overlap: typeof chunkOverlap === "number" ? chunkOverlap : 0,
    reindexed_documents: typeof reindexedDocuments === "number" ? reindexedDocuments : 0,
    active_chunks: typeof activeChunks === "number" ? activeChunks : 0,
  };
}

function normalizeUpdateRetrievalSettingsResponse(payload: unknown): ApiUpdateRetrievalSettingsResponse {
  if (!isObject(payload)) {
    throw new Error("Invalid settings/retrieval payload");
  }

  const topK = optionalNumber(payload.top_k);
  const rerankTopN = optionalNumber(payload.rerank_top_n);
  const contextTopK = optionalNumber(payload.context_top_k);

  return {
    status: payload.status === "updated" ? "updated" : "updated",
    mode: normalizeRetrievalMode(payload.mode),
    retrieval_mode: normalizeRetrievalConfigMode(payload.retrieval_mode),
    top_k: typeof topK === "number" ? topK : 0,
    rerank_top_n: typeof rerankTopN === "number" ? rerankTopN : 0,
    context_top_k: typeof contextTopK === "number" ? contextTopK : 0,
  };
}

type RequestOptions = {
  signal?: AbortSignal;
};

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export async function postQuery(
  request: ApiQueryRequest,
  options: RequestOptions = {},
): Promise<ApiQueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal: options.signal,
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  return (await parseJsonResponse(response)) as ApiQueryResponse;
}

export type QueryStreamHandlers = {
  onStart?: (payload: Record<string, unknown>) => void;
  onRetrieval?: (payload: Record<string, unknown>) => void;
  onGenerationDelta?: (
    delta: string,
    meta: {
      event: Record<string, unknown>;
      time_to_first_token_ms: number | null;
    },
  ) => void;
  onTrace?: (payload: Record<string, unknown>, eventName: string) => void;
  onFinal?: (response: ApiQueryResponse, payload: Record<string, unknown>) => void;
  onError?: (message: string, payload: Record<string, unknown>) => void;
  onDone?: () => void;
  onDebugEvent?: (eventName: string, payload: Record<string, unknown>) => void;
};

function parseSseFrame(frame: string): { eventName: string; payload: Record<string, unknown> | null } {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const rawLine of frame.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return { eventName, payload: null };
  }

  try {
    const parsed = JSON.parse(dataLines.join("\n")) as unknown;
    if (isObject(parsed)) {
      return { eventName, payload: parsed };
    }
    return { eventName, payload: null };
  } catch {
    return { eventName, payload: null };
  }
}

async function fallbackToPostQuery(
  request: ApiQueryRequest,
  handlers: QueryStreamHandlers | undefined,
  options: RequestOptions = {},
): Promise<ApiQueryResponse> {
  const response = await postQuery(request, options);
  handlers?.onFinal?.(response, {
    type: "final",
    response,
    stream_fallback: true,
  });
  handlers?.onDone?.();
  return response;
}

export function isQueryStreamEnabled(): boolean {
  return QUERY_STREAM_ENABLED;
}

export async function postQueryStream(
  request: ApiQueryRequest,
  handlers?: QueryStreamHandlers,
  options: RequestOptions = {},
): Promise<ApiQueryResponse> {
  if (!supportsStreamingResponse()) {
    return fallbackToPostQuery(request, handlers, options);
  }

  const streamStartedAt = nowMs();
  let firstTokenMs: number | null = null;
  let sawAnyEvent = false;
  let doneNotified = false;

  try {
    const response = await fetch(`${API_BASE_URL}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      signal: options.signal,
    });

    if (!response.ok) {
      await handleApiFailure(response);
    }

    if (!response.body) {
      return fallbackToPostQuery(request, handlers, options);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResponse: ApiQueryResponse | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const frame = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        separatorIndex = buffer.indexOf("\n\n");

        const { eventName, payload } = parseSseFrame(frame);
        if (!payload) {
          continue;
        }
        sawAnyEvent = true;

        if (eventName === "start") {
          handlers?.onStart?.(payload);
          continue;
        }
        if (eventName === "retrieval") {
          handlers?.onRetrieval?.(payload);
          handlers?.onTrace?.(payload, eventName);
          continue;
        }
        if (eventName === "generation_delta") {
          const delta = typeof payload.delta === "string" ? payload.delta : "";
          if (delta && firstTokenMs === null) {
            firstTokenMs = Math.max(0, Math.round(nowMs() - streamStartedAt));
          }
          handlers?.onGenerationDelta?.(delta, {
            event: payload,
            time_to_first_token_ms: firstTokenMs,
          });
          handlers?.onTrace?.(payload, eventName);
          continue;
        }
        if (eventName === "generation") {
          if (typeof payload.delta === "string" && payload.delta.length > 0) {
            if (firstTokenMs === null) {
              firstTokenMs = Math.max(0, Math.round(nowMs() - streamStartedAt));
            }
            handlers?.onGenerationDelta?.(payload.delta, {
              event: payload,
              time_to_first_token_ms: firstTokenMs,
            });
          }
          handlers?.onTrace?.(payload, eventName);
          continue;
        }
        if (eventName === "final") {
          const responsePayload = payload.response;
          if (isObject(responsePayload)) {
            finalResponse = responsePayload as ApiQueryResponse;
            handlers?.onFinal?.(finalResponse, payload);
            handlers?.onTrace?.(payload, eventName);
          }
          continue;
        }
        if (eventName === "error") {
          const errorPayload = isObject(payload.error) ? payload.error : {};
          const rawMessage = typeof errorPayload.message === "string" ? errorPayload.message : "";
          const safeMessage = sanitizeStreamErrorMessage(rawMessage);
          handlers?.onError?.(safeMessage, payload);
          handlers?.onTrace?.(payload, eventName);
          throw new ApiRequestError(500, safeMessage);
        }
        if (eventName === "done") {
          handlers?.onDone?.();
          doneNotified = true;
          continue;
        }

        handlers?.onDebugEvent?.(eventName, payload);
        handlers?.onTrace?.(payload, eventName);
      }
    }

    buffer += decoder.decode();
    const trailing = buffer.trim();
    if (trailing) {
      const { eventName, payload } = parseSseFrame(trailing);
      if (payload) {
        sawAnyEvent = true;
        if (eventName === "final" && isObject(payload.response)) {
          finalResponse = payload.response as ApiQueryResponse;
          handlers?.onFinal?.(finalResponse, payload);
          handlers?.onTrace?.(payload, eventName);
        } else if (eventName === "done") {
          handlers?.onDone?.();
          doneNotified = true;
        } else {
          handlers?.onDebugEvent?.(eventName, payload);
          handlers?.onTrace?.(payload, eventName);
        }
      }
    }

    if (finalResponse) {
      if (!doneNotified) {
        handlers?.onDone?.();
        doneNotified = true;
      }
      return finalResponse;
    }

    if (!sawAnyEvent) {
      return fallbackToPostQuery(request, handlers, options);
    }
    throw new ApiRequestError(500, "Không nhận được phản hồi cuối từ stream.");
  } catch (error) {
    if (options.signal?.aborted || isAbortError(error)) {
      throw error;
    }
    if (!sawAnyEvent) {
      return fallbackToPostQuery(request, handlers, options);
    }
    throw error;
  } finally {
    if (sawAnyEvent && !doneNotified) {
      handlers?.onDone?.();
    }
  }
}

export async function getHealthStatus(): Promise<ApiHealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  return (await parseJsonResponse(response)) as ApiHealthResponse;
}

export async function uploadDocument(
  file: File,
  onProgress?: (progressPercent: number) => void,
): Promise<ApiUploadDocumentResponse> {
  return new Promise<ApiUploadDocumentResponse>((resolve, reject) => {
    const request = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    request.open("POST", DOCUMENTS_UPLOAD_URL, true);
    request.responseType = "json";

    request.upload.onprogress = (event) => {
      if (!event.lengthComputable || !onProgress) {
        return;
      }
      const percent = Math.max(0, Math.min(100, Math.round((event.loaded / event.total) * 100)));
      onProgress(percent);
    };

    request.onerror = () => {
      reject(new ApiRequestError(0, "Network error while uploading document"));
    };

    request.onload = () => {
      const { status } = request;
      if (status >= 200 && status < 300) {
        try {
          const payload = request.response as unknown;
          const normalized = normalizeDocument(payload);
          resolve(normalized);
        } catch {
          reject(new ApiRequestError(status, "Invalid upload response payload"));
        }
        return;
      }

      const payload = request.response;
      let message = request.statusText || "Request failed";
      if (isObject(payload) && typeof payload.detail === "string") {
        message = payload.detail;
      }

      if (isFeatureUnavailableStatus(status)) {
        reject(new ApiFeatureUnavailableError(status, message));
        return;
      }
      reject(new ApiRequestError(status, message));
    };

    request.send(formData);
  });
}

export async function getDocumentStatus(documentId: string): Promise<ApiDocumentStatusResponse> {
  const response = await fetch(`${DOCUMENTS_BASE_URL}/${encodeURIComponent(documentId)}/status`, {
    method: "GET",
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeDocument(payload);
}

export async function listDocuments(): Promise<ApiDocument[]> {
  const response = await fetch(DOCUMENTS_BASE_URL, {
    method: "GET",
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeDocumentList(payload);
}

export async function deleteAllDocuments(): Promise<ApiDeleteAllDocumentsResponse> {
  const response = await fetch(DOCUMENTS_BASE_URL, {
    method: "DELETE",
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeDeleteAllDocumentsResponse(payload);
}

export async function deleteDocument(documentId: string): Promise<ApiDeleteDocumentResponse> {
  const response = await fetch(`${DOCUMENTS_BASE_URL}/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeDeleteDocumentResponse(payload);
}

export async function reindexDocuments(
  request: ApiReindexDocumentsRequest,
): Promise<ApiReindexDocumentsResponse> {
  const response = await fetch(`${DOCUMENTS_BASE_URL}/reindex`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeReindexDocumentsResponse(payload);
}

export async function updateChunkingSettings(
  request: ApiUpdateChunkingSettingsRequest,
): Promise<ApiUpdateChunkingSettingsResponse> {
  const response = await fetch(`${SETTINGS_BASE_URL}/chunking`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeUpdateChunkingSettingsResponse(payload);
}

export async function updateRetrievalSettings(
  request: ApiUpdateRetrievalSettingsRequest,
): Promise<ApiUpdateRetrievalSettingsResponse> {
  const response = await fetch(`${SETTINGS_BASE_URL}/retrieval`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    await handleApiFailure(response);
  }

  const payload = await parseJsonResponse(response);
  return normalizeUpdateRetrievalSettingsResponse(payload);
}
