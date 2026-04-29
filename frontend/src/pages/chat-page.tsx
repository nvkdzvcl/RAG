import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Clock3, X } from "lucide-react";

import { ApiRequestError, getHealthStatus, isQueryStreamEnabled, postQuery, postQueryStream } from "@/api/client";
import type { ApiQueryRequest, ApiQueryResponse, ApiRetrievalMode } from "@/api/types";
import { apiToUi } from "@/api/transform";
import { AppShell } from "@/components/dashboard/app-shell";
import { DocumentUploadCard } from "@/components/dashboard/document-upload-card";
import {
  QueryFilters,
  type QueryFileTypeFilter,
  type QueryOcrFilter,
} from "@/components/dashboard/query-filters";
import { SettingsModal, type SettingsChangePayload } from "@/components/dashboard/settings-modal";
import { ChatContainer } from "@/components/rag/chat-container";
import { CitationPreviewModal } from "@/components/rag/citation-preview-modal";
import { Composer } from "@/components/rag/composer";
import { ModeSelector } from "@/components/rag/mode-selector";
import { Sidebar, type RecentChat } from "@/components/rag/sidebar";
import { SidePanel, type SidePanelTab } from "@/components/rag/side-panel";
import { ThemeToggle } from "@/components/rag/theme-toggle";
import { flattenCitationItems, type CitationPanelItem } from "@/components/rag/citation-utils";
import { AlertDialog } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { useDocumentIngestion } from "@/hooks/use-document-ingestion";
import {
  clearChatSessionsStorage,
  createChatSession,
  defaultSessionTitle,
  deriveSessionTitle,
  formatRelativeTime,
  loadChatSessions,
  persistChatSessions,
} from "@/lib/chat-sessions";
import {
  applyThemePreference,
  persistThemePreference,
  readThemePreference,
  type ThemeMode,
} from "@/lib/theme";
import type { Mode, QueryResult, SingleMode } from "@/types/chat";
import type { ChatSession, ChatSessionMessage } from "@/types/chat-session";
import type { DocumentRecord } from "@/types/document";

const DEFAULT_MODEL =
  (typeof import.meta.env.VITE_DEFAULT_LLM_MODEL === "string" &&
  import.meta.env.VITE_DEFAULT_LLM_MODEL.trim().length > 0
    ? import.meta.env.VITE_DEFAULT_LLM_MODEL.trim()
    : "qwen2.5:3b");

const MAX_MESSAGES_PER_SESSION = 24;
const MAX_STORED_SESSIONS = 80;
const STREAM_QUERY_ENABLED = isQueryStreamEnabled();

type StreamingStage = "starting" | "retrieving" | "reranking" | "generating" | "grounding";

type StreamProgress = {
  stage: StreamingStage;
  partialAnswer: string;
  stageReached: {
    retrieving: boolean;
    reranking: boolean;
    generating: boolean;
    grounding: boolean;
  };
  timeToFirstTokenMs: number | null;
  totalLatencyMs: number | null;
};

function initialStreamProgress(): StreamProgress {
  return {
    stage: "starting",
    partialAnswer: "",
    stageReached: {
      retrieving: false,
      reranking: false,
      generating: false,
      grounding: false,
    },
    timeToFirstTokenMs: null,
    totalLatencyMs: null,
  };
}

function fallbackErrorMessage(error: unknown): string {
  if (error instanceof ApiRequestError) {
    const cleaned = error.message.replace(/^API error \(\d+\):\s*/i, "").trim();
    return cleaned || "Không thể hoàn tất yêu cầu.";
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    const trimmed = error.message.trim();
    if (trimmed.includes("Traceback") || trimmed.includes("\n")) {
      return "Không thể hoàn tất yêu cầu. Vui lòng thử lại.";
    }
    return trimmed;
  }
  return "Lỗi yêu cầu không xác định";
}

function totalLatencyFromFinalResponse(payload: ApiQueryResponse): number | null {
  if (payload.mode === "compare") {
    const standardTrace = payload.standard.trace ?? [];
    const advancedTrace = payload.advanced.trace ?? [];
    const findCompareTotal = (trace: Array<Record<string, unknown>>): number | null => {
      for (const item of trace) {
        if (
          item.step === "compare_timing"
          && typeof item.compare_total_ms === "number"
          && Number.isFinite(item.compare_total_ms)
        ) {
          return item.compare_total_ms;
        }
      }
      return null;
    };
    const traced = findCompareTotal(standardTrace) ?? findCompareTotal(advancedTrace);
    if (traced !== null) {
      return traced;
    }
    if (
      typeof payload.standard.latency_ms === "number"
      && typeof payload.advanced.latency_ms === "number"
    ) {
      return payload.standard.latency_ms + payload.advanced.latency_ms;
    }
    return null;
  }

  if (typeof payload.latency_ms === "number" && Number.isFinite(payload.latency_ms)) {
    return payload.latency_ms;
  }
  return null;
}

function dateInputToIso(value: string, boundary: "start" | "end"): string | undefined {
  if (!value) {
    return undefined;
  }
  return boundary === "start" ? `${value}T00:00:00.000Z` : `${value}T23:59:59.999Z`;
}

function ocrFilterToPayload(value: QueryOcrFilter): boolean | undefined {
  if (value === "only") {
    return true;
  }
  if (value === "exclude") {
    return false;
  }
  return undefined;
}

function initialSessions(): ChatSession[] {
  const loaded = loadChatSessions();
  if (loaded.length > 0) {
    return loaded;
  }
  return [
    createChatSession({
      mode: "standard",
      model: DEFAULT_MODEL,
    }),
  ];
}

function normalizedModeForMessage(mode: Mode): SingleMode {
  if (mode === "advanced") {
    return "advanced";
  }
  return "standard";
}

function resultReferencesDeletedDocuments(result: QueryResult | null, deletedDocumentIds: Set<string>): boolean {
  if (!result || deletedDocumentIds.size === 0) {
    return false;
  }

  const hasDeleted = (docId: string): boolean => deletedDocumentIds.has(docId);
  if (result.mode === "compare") {
    const standardHit = result.standard.citations.some((item) => hasDeleted(item.docId));
    const advancedHit = result.advanced.citations.some((item) => hasDeleted(item.docId));
    return standardHit || advancedHit;
  }
  return result.citations.some((item) => hasDeleted(item.docId));
}

function firstUserMessageTitle(messages: ChatSessionMessage[]): string {
  const derived = deriveSessionTitle(messages);
  return derived || defaultSessionTitle();
}

export function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => initialSessions());
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [streamProgress, setStreamProgress] = useState<StreamProgress | null>(null);
  const [theme, setTheme] = useState<ThemeMode>(() => readThemePreference());
  const [sidePanelOpen, setSidePanelOpen] = useState(true);
  const [sidePanelTab, setSidePanelTab] = useState<SidePanelTab>("citations");
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);
  const [previewCitation, setPreviewCitation] = useState<CitationPanelItem | null>(null);
  const [showDocumentsModal, setShowDocumentsModal] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [retrievalMode, setRetrievalMode] = useState<ApiRetrievalMode>("high");
  const [topK, setTopK] = useState(8);
  const [rerankTopN, setRerankTopN] = useState(6);
  const [contextTopK, setContextTopK] = useState(4);
  const [showClearHistoryDialog, setShowClearHistoryDialog] = useState(false);
  const [showClearVectorDialog, setShowClearVectorDialog] = useState(false);
  const [selectedDocumentForDelete, setSelectedDocumentForDelete] = useState<DocumentRecord | null>(null);
  const [isDeletingAllDocuments, setIsDeletingAllDocuments] = useState(false);
  const [isDeletingSingleDocument, setIsDeletingSingleDocument] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [selectedFilterDocIds, setSelectedFilterDocIds] = useState<string[]>([]);
  const [selectedFileTypes, setSelectedFileTypes] = useState<QueryFileTypeFilter[]>([]);
  const [uploadedAfter, setUploadedAfter] = useState("");
  const [uploadedBefore, setUploadedBefore] = useState("");
  const [ocrFilter, setOcrFilter] = useState<QueryOcrFilter>("all");
  const abortControllerRef = useRef<AbortController | null>(null);

  const {
    documents,
    activeDocument,
    isUploading,
    isDeletingDocuments,
    deletingDocumentId,
    uploadMessage,
    uploadError,
    uploadBatchItems,
    uploadBatchSummary,
    canQuery,
    queryDisabledReason,
    uploadFiles,
    clearAllUploadedDocuments,
    deleteUploadedDocument,
    reindexDocuments,
    updateRetrievalSettings,
  } = useDocumentIngestion();

  useEffect(() => {
    if (activeSessionId) {
      return;
    }
    if (sessions.length > 0) {
      setActiveSessionId(sessions[0].id);
    }
  }, [activeSessionId, sessions]);

  useEffect(() => {
    const limited = sessions.slice(0, MAX_STORED_SESSIONS);
    persistChatSessions(limited);
  }, [sessions]);

  useEffect(() => {
    if (!toastMessage) {
      return;
    }
    const timeout = window.setTimeout(() => setToastMessage(null), 2200);
    return () => window.clearTimeout(timeout);
  }, [toastMessage]);

  useEffect(() => {
    applyThemePreference(theme);
    persistThemePreference(theme);
  }, [theme]);

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) ?? sessions[0] ?? null,
    [sessions, activeSessionId],
  );

  useEffect(() => {
    if (!activeSessionId && activeSession) {
      setActiveSessionId(activeSession.id);
    }
  }, [activeSession, activeSessionId]);

  useEffect(() => {
    let mounted = true;

    const loadBackendDefaultModel = async () => {
      try {
        const health = await getHealthStatus();
        if (!mounted || !activeSession) {
          return;
        }
        if (typeof health.llm_model !== "string" || health.llm_model.trim().length === 0) {
          return;
        }
        if (activeSession.messages.length > 0) {
          return;
        }

        const resolved = health.llm_model.trim();
        if (activeSession.model === resolved) {
          return;
        }
        setSessions((previous) =>
          previous.map((item) =>
            item.id === activeSession.id
              ? {
                  ...item,
                  model: resolved,
                }
              : item,
          ),
        );
      } catch {
        // Keep local default model when backend health/default model is unavailable.
      }
    };

    void loadBackendDefaultModel();
    return () => {
      mounted = false;
    };
  }, [activeSession]);

  const mode: Mode = activeSession?.mode ?? "standard";
  const selectedModel: string = activeSession?.model ?? DEFAULT_MODEL;
  const result: QueryResult | null = activeSession?.lastResult ?? null;
  const submittedQuery: string | null = activeSession?.lastSubmittedQuery ?? null;
  const messages: ChatSessionMessage[] = activeSession?.messages ?? [];

  const recentChats = useMemo<RecentChat[]>(() => {
    return sessions
      .filter((item) => item.messages.some((message) => message.role === "user"))
      .sort((left, right) => {
        if (left.pinned !== right.pinned) {
          return left.pinned ? -1 : 1;
        }
        return right.updatedAt.localeCompare(left.updatedAt);
      })
      .map((item) => ({
        id: item.id,
        title: item.title,
        pinned: item.pinned,
        timeLabel: formatRelativeTime(item.updatedAt),
      }));
  }, [sessions]);

  const canSubmit = query.trim().length > 0 && !isLoading && canQuery;
  const readyDocuments = useMemo(
    () => documents.filter((item) => item.status === "ready"),
    [documents],
  );

  useEffect(() => {
    if (selectedFilterDocIds.length === 0) {
      return;
    }
    const readyIds = new Set(readyDocuments.map((item) => item.id));
    const next = selectedFilterDocIds.filter((item) => readyIds.has(item));
    if (next.length !== selectedFilterDocIds.length) {
      setSelectedFilterDocIds(next);
    }
  }, [readyDocuments, selectedFilterDocIds]);

  const updateSessionById = (sessionId: string, updater: (previous: ChatSession) => ChatSession) => {
    setSessions((previous) =>
      previous.map((item) => (item.id === sessionId ? updater(item) : item)),
    );
  };

  const ensureActiveSession = (): ChatSession => {
    if (activeSession) {
      return activeSession;
    }
    const created = createChatSession({ mode: "standard", model: DEFAULT_MODEL });
    setSessions((previous) => [created, ...previous]);
    setActiveSessionId(created.id);
    return created;
  };

  const createFreshSession = (preserveMode: Mode, preserveModel: string): ChatSession => {
    const created = createChatSession({ mode: preserveMode, model: preserveModel });
    setSessions((previous) => [created, ...previous].slice(0, MAX_STORED_SESSIONS));
    setActiveSessionId(created.id);
    setQuery("");
    setError(null);
    setNotice(null);
    return created;
  };

  const clearCurrentSessionResult = (sessionId: string) => {
    const now = new Date().toISOString();
    updateSessionById(sessionId, (previous) => ({
      ...previous,
      updatedAt: now,
      lastResult: null,
      lastResultSummary: null,
      lastSubmittedQuery: null,
    }));
  };

  const runQuery = async (queryOverride?: string) => {
    const normalized = (queryOverride ?? query).trim();
    if (!normalized || isLoading) {
      return;
    }
    if (!canQuery) {
      setError(queryDisabledReason || "Vui lòng tải lên và xử lý tài liệu trước khi đặt câu hỏi.");
      return;
    }

    const currentSession = ensureActiveSession();
    const requestSessionId = currentSession.id;
    const requestMode = currentSession.mode;
    const requestModel = currentSession.model;
    const previousMessages = [...currentSession.messages];
    const requestChatHistoryPayload = previousMessages.map((item) => ({
      role: item.role,
      content: item.content,
    }));
    const userTimestamp = new Date().toISOString();
    const userMessage: ChatSessionMessage = {
      role: "user",
      content: normalized,
      mode: requestMode,
      model: requestModel,
      timestamp: userTimestamp,
      metadata: null,
    };

    updateSessionById(requestSessionId, (previous) => {
      const nextMessages = [...previous.messages, userMessage].slice(-MAX_MESSAGES_PER_SESSION);
      const nextTitle =
        previous.title === defaultSessionTitle() ? firstUserMessageTitle(nextMessages) : previous.title;
      return {
        ...previous,
        title: nextTitle,
        updatedAt: userTimestamp,
        mode: requestMode,
        model: requestModel,
        messages: nextMessages,
        lastSubmittedQuery: normalized,
      };
    });

    setIsLoading(true);
    setError(null);
    setNotice(null);
    setQuery("");
    setActiveCitationId(null);
    setPreviewCitation(null);
    setStreamProgress(initialStreamProgress());
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const selectedDocuments = selectedFilterDocIds.length > 0
        ? readyDocuments.filter((item) => selectedFilterDocIds.includes(item.id))
        : [];
      const requestPayload: ApiQueryRequest = {
        query: normalized,
        mode: requestMode,
        chat_history: requestChatHistoryPayload,
        model: requestModel,
        doc_ids: selectedFilterDocIds.length > 0 ? selectedFilterDocIds : undefined,
        filenames: selectedDocuments.length > 0 ? selectedDocuments.map((item) => item.filename) : undefined,
        file_types: selectedFileTypes.length > 0 ? selectedFileTypes : undefined,
        uploaded_after: dateInputToIso(uploadedAfter, "start"),
        uploaded_before: dateInputToIso(uploadedBefore, "end"),
        include_ocr: ocrFilterToPayload(ocrFilter),
      };

      const applyTraceStage = (payload: Record<string, unknown>, eventName: string) => {
        if (eventName === "retrieval") {
          setStreamProgress((previous) => {
            if (!previous) {
              return previous;
            }
            return {
              ...previous,
              stage: "reranking",
              stageReached: {
                ...previous.stageReached,
                retrieving: true,
                reranking: true,
              },
            };
          });
          return;
        }

        if (eventName === "generation") {
          const phase = typeof payload.phase === "string" ? payload.phase : "";
          if (phase === "started") {
            setStreamProgress((previous) => {
              if (!previous) {
                return previous;
              }
              return {
                ...previous,
                stage: "generating",
                stageReached: {
                  ...previous.stageReached,
                  generating: true,
                },
              };
            });
            return;
          }
          if (phase === "completed" || phase === "skipped") {
            setStreamProgress((previous) => {
              if (!previous) {
                return previous;
              }
              return {
                ...previous,
                stage: "grounding",
                stageReached: {
                  ...previous.stageReached,
                  generating: true,
                  grounding: true,
                },
              };
            });
            return;
          }
        }

        if (eventName === "advanced_stage") {
          const stage = typeof payload.stage === "string" ? payload.stage : "";
          if (stage === "retrieval_gate" || stage === "critique_loop") {
            setStreamProgress((previous) => {
              if (!previous) {
                return previous;
              }
              return {
                ...previous,
                stage: "reranking",
                stageReached: {
                  ...previous.stageReached,
                  retrieving: true,
                  reranking: true,
                },
              };
            });
            return;
          }
          if (
            stage === "refine"
            || stage === "language_guard"
            || stage === "hallucination_guard"
            || stage === "final_grounding"
          ) {
            setStreamProgress((previous) => {
              if (!previous) {
                return previous;
              }
              return {
                ...previous,
                stage: "grounding",
                stageReached: {
                  ...previous.stageReached,
                  grounding: true,
                },
              };
            });
          }
        }
      };

      const payload = STREAM_QUERY_ENABLED
        ? await postQueryStream(
            requestPayload,
            {
              onStart: () => {
                setStreamProgress((previous) => {
                  if (!previous) {
                    return initialStreamProgress();
                  }
                  return {
                    ...previous,
                    stage: "retrieving",
                  };
                });
              },
              onRetrieval: () => {
                setStreamProgress((previous) => {
                  if (!previous) {
                    return previous;
                  }
                  return {
                    ...previous,
                    stage: "reranking",
                    stageReached: {
                      ...previous.stageReached,
                      retrieving: true,
                      reranking: true,
                    },
                  };
                });
              },
              onGenerationDelta: (delta, meta) => {
                if (!delta) {
                  return;
                }
                setStreamProgress((previous) => {
                  const base = previous ?? initialStreamProgress();
                  return {
                    ...base,
                    stage: "generating",
                    partialAnswer: `${base.partialAnswer}${delta}`,
                    stageReached: {
                      ...base.stageReached,
                      generating: true,
                    },
                    timeToFirstTokenMs:
                      base.timeToFirstTokenMs ?? meta.time_to_first_token_ms ?? null,
                  };
                });
              },
              onTrace: (eventPayload, eventName) => {
                applyTraceStage(eventPayload, eventName);
                const compareTotalMs = eventPayload.compare_total_ms;
                if (
                  eventName === "compare_timing"
                  && typeof compareTotalMs === "number"
                  && Number.isFinite(compareTotalMs)
                ) {
                  setStreamProgress((previous) => {
                    if (!previous) {
                      return previous;
                    }
                    return {
                      ...previous,
                      totalLatencyMs: compareTotalMs,
                    };
                  });
                }
              },
              onFinal: (finalPayload) => {
                const totalLatency = totalLatencyFromFinalResponse(finalPayload);
                setStreamProgress((previous) => {
                  const base = previous ?? initialStreamProgress();
                  return {
                    ...base,
                    stage: "grounding",
                    stageReached: {
                      retrieving: true,
                      reranking: true,
                      generating: true,
                      grounding: true,
                    },
                    totalLatencyMs: totalLatency,
                  };
                });
              },
              onError: (streamErrorMessage) => {
                setError(streamErrorMessage);
              },
            },
            { signal: abortController.signal },
          )
        : await postQuery(requestPayload, { signal: abortController.signal });

      const mapped = apiToUi(payload);
      const now = new Date().toISOString();
      const assistantMessageContent =
        mapped.mode === "compare"
          ? mapped.comparison.preferredMode === "advanced"
            ? mapped.advanced.answer
            : mapped.comparison.preferredMode === "standard"
              ? mapped.standard.answer
              : mapped.comparison.note || mapped.advanced.answer || "Hoàn tất so sánh."
          : mapped.answer;

      const assistantMessage: ChatSessionMessage = {
        role: "assistant",
        content: assistantMessageContent,
        mode: mapped.mode === "compare" ? "advanced" : normalizedModeForMessage(mapped.mode),
        model: requestModel,
        timestamp: now,
        metadata: {
          result_mode: mapped.mode,
        },
      };

      updateSessionById(requestSessionId, (previous) => {
        const nextMessages = [...previous.messages, assistantMessage].slice(-MAX_MESSAGES_PER_SESSION);
        return {
          ...previous,
          updatedAt: now,
          mode: requestMode,
          model: requestModel,
          messages: nextMessages,
          lastResult: mapped,
          lastResultSummary: assistantMessageContent,
          lastSubmittedQuery: normalized,
        };
      });
    } catch (requestError) {
      const requestWasAborted =
        abortController.signal.aborted
        || (requestError instanceof DOMException && requestError.name === "AbortError");
      if (requestWasAborted) {
        setNotice("Generation stopped.");
        return;
      }
      setError(fallbackErrorMessage(requestError));
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
      setIsLoading(false);
      setStreamProgress(null);
    }
  };

  const handleNewChat = () => {
    createFreshSession(mode, selectedModel);
    setActiveCitationId(null);
    setPreviewCitation(null);
  };

  const handleStopGeneration = () => {
    abortControllerRef.current?.abort();
    setNotice("Generation stopped.");
    setIsLoading(false);
    setStreamProgress(null);
  };

  const handleClearHistory = () => {
    clearChatSessionsStorage();
    const fresh = createChatSession({ mode, model: selectedModel });
    setSessions([fresh]);
    setActiveSessionId(fresh.id);
    setQuery("");
    setError(null);
    setNotice(null);
    setPreviewCitation(null);
  };

  const handleClearVectorStore = async () => {
    setError(null);
    setNotice(null);
    setIsDeletingAllDocuments(true);
    const deletedDocIds = new Set(documents.map((item) => item.id));
    try {
      await clearAllUploadedDocuments();
      if (activeSession) {
        clearCurrentSessionResult(activeSession.id);
      }
      if (deletedDocIds.size > 0) {
        setNotice("Lưu ý: Một số câu trả lời cũ có thể tham chiếu tài liệu vừa bị xóa.");
      }
      return true;
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Không thể xóa tài liệu đã tải.";
      setError(message);
      return false;
    } finally {
      setIsDeletingAllDocuments(false);
    }
  };

  const handleDeleteSingleDocument = async () => {
    if (!selectedDocumentForDelete) {
      return false;
    }

    setError(null);
    setNotice(null);
    setIsDeletingSingleDocument(true);
    const deletingId = selectedDocumentForDelete.id;
    try {
      const deleted = await deleteUploadedDocument(deletingId);
      if (activeSession && resultReferencesDeletedDocuments(activeSession.lastResult ?? null, new Set([deleted.documentId]))) {
        clearCurrentSessionResult(activeSession.id);
      }
      setNotice("Lưu ý: Một số câu trả lời cũ có thể tham chiếu tài liệu vừa bị xóa.");
      setSelectedDocumentForDelete(null);
      return true;
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Không thể xóa tài liệu đã chọn.";
      setError(message);
      return false;
    } finally {
      setIsDeletingSingleDocument(false);
    }
  };

  const handleSettingsChange = async (payload: SettingsChangePayload) => {
    const notices: string[] = [];
    const shouldReindex =
      payload.chunking.chunkSize !== chunkSize || payload.chunking.chunkOverlap !== chunkOverlap;
    const shouldUpdateRetrieval =
      payload.retrieval.mode !== retrievalMode || payload.retrieval.topK !== topK;

    try {
      if (shouldReindex) {
        const reindexed = await reindexDocuments({
          mode: payload.chunking.mode,
          chunkSize: payload.chunking.chunkSize,
          chunkOverlap: payload.chunking.chunkOverlap,
        });
        setChunkSize(reindexed.chunkSize);
        setChunkOverlap(reindexed.chunkOverlap);
        notices.push(
          `Đã cập nhật ${reindexed.mode} (${reindexed.chunkSize}/${reindexed.chunkOverlap}). Re-index ${reindexed.reindexedDocuments} tài liệu, tổng chunk hoạt động: ${reindexed.activeChunks}.`,
        );
      }

      if (shouldUpdateRetrieval) {
        const updated = await updateRetrievalSettings({
          mode: payload.retrieval.mode,
          topK: payload.retrieval.topK,
        });
        setRetrievalMode(updated.mode);
        setTopK(updated.topK);
        setRerankTopN(updated.rerankTopN);
        setContextTopK(updated.contextTopK);
        notices.push(`Đã cập nhật truy xuất: top-k ${updated.topK}, rerank còn ${updated.rerankTopN}.`);
      }

      if (notices.length === 0) {
        setNotice("Không có thay đổi cấu hình.");
      } else {
        setNotice(notices.join(" "));
      }
      setError(null);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Không thể cập nhật cấu hình chunk/retrieval.";
      setError(message);
    }
  };

  const handleSelectRecent = (chat: RecentChat) => {
    setActiveSessionId(chat.id);
    setQuery("");
    setError(null);
    setNotice(null);
    setActiveCitationId(null);
    setPreviewCitation(null);
  };

  const handleTogglePinChat = (chat: RecentChat) => {
    setSessions((previous) =>
      previous.map((item) =>
        item.id === chat.id
          ? {
              ...item,
              pinned: !item.pinned,
            }
          : item,
      ),
    );
  };

  const handleDeleteChat = (chat: RecentChat) => {
    setSessions((previous) => {
      const remaining = previous.filter((item) => item.id !== chat.id);
      if (remaining.length === 0) {
        const created = createChatSession({ mode, model: selectedModel });
        setActiveSessionId(created.id);
        return [created];
      }
      if (activeSessionId === chat.id) {
        setActiveSessionId(remaining[0].id);
      }
      return remaining;
    });

    setError(null);
    setNotice(null);
    setActiveCitationId(null);
    setPreviewCitation(null);
  };

  const handleModeChange = (nextMode: Mode) => {
    if (!activeSession) {
      return;
    }
    updateSessionById(activeSession.id, (previous) => ({
      ...previous,
      mode: nextMode,
    }));
  };

  const handleModelChange = (nextModel: string) => {
    if (!activeSession) {
      return;
    }
    updateSessionById(activeSession.id, (previous) => ({
      ...previous,
      model: nextModel,
    }));
  };

  const handleCitationClick = (panelId: string) => {
    setActiveCitationId(panelId);
    setSidePanelTab("citations");
    const selected = flattenCitationItems(result).find((item) => item.panelId === panelId) ?? null;
    setPreviewCitation(selected);
  };

  const handleToggleDebug = () => {
    setSidePanelTab("debug");
    setSidePanelOpen(true);
  };

  const handleCopyNotice = () => {
    setToastMessage("Copied to clipboard");
  };

  const handleRegenerate = () => {
    if (!submittedQuery || isLoading) {
      return;
    }
    void runQuery(submittedQuery);
  };

  const resultLatencyMs =
    result?.mode === "compare"
      ? result.standard.latencyMs !== null && result.advanced.latencyMs !== null
        ? result.standard.latencyMs + result.advanced.latencyMs
        : null
      : result?.latencyMs ?? null;
  const latencyLabel =
    streamProgress?.totalLatencyMs !== null && streamProgress?.totalLatencyMs !== undefined
      ? `${streamProgress.totalLatencyMs}ms`
      : resultLatencyMs !== null
        ? `${resultLatencyMs}ms`
        : "n/a";

  const mainContent = (
    <div className="flex h-full min-h-0 flex-col">
      <header className="relative z-30 flex min-h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-background/75 px-4 backdrop-blur-xl md:px-6">
        <ModeSelector
          workflowMode={mode}
          onWorkflowModeChange={handleModeChange}
          selectedModel={selectedModel}
          onModelChange={handleModelChange}
          disabled={isLoading}
        />
        <div className="flex shrink-0 items-center gap-2">
          <div className="hidden items-center gap-2 rounded-xl border border-border bg-card/80 px-3 py-2 font-mono text-xs text-muted-foreground shadow-subtle md:flex">
            <Clock3 className="h-3.5 w-3.5 text-success" />
            <span>Latency:</span>
            <span className="font-semibold text-accent">{latencyLabel}</span>
          </div>
          <ThemeToggle theme={theme} onThemeChange={setTheme} />
        </div>
      </header>

      <ChatContainer
        messages={messages}
        result={result}
        isLoading={isLoading}
        error={error}
        notice={notice}
        streamProgress={streamProgress}
        canQuery={canQuery}
        onOpenDocuments={() => setShowDocumentsModal(true)}
        onCitationClick={handleCitationClick}
        onCopied={handleCopyNotice}
        onRegenerate={handleRegenerate}
        onToggleDebug={handleToggleDebug}
      />

      <Composer
        query={query}
        onQueryChange={setQuery}
        onSubmit={() => void runQuery()}
        onStop={handleStopGeneration}
        isLoading={isLoading}
        canSubmit={canSubmit}
        disabled={!canQuery}
        disabledReason={queryDisabledReason}
      />
    </div>
  );

  const inspectPanel = (
    <SidePanel
      open={sidePanelOpen}
      tab={sidePanelTab}
      onOpenChange={setSidePanelOpen}
      onTabChange={setSidePanelTab}
      result={result}
      activeCitationId={activeCitationId}
      query={submittedQuery}
      onFocusCitation={setActiveCitationId}
    />
  );

  return (
    <>
      <AppShell
        sidebar={
          <Sidebar
            activeSessionId={activeSessionId}
            onNewChat={handleNewChat}
            recentChats={recentChats}
            onSelectRecent={handleSelectRecent}
            onTogglePinChat={handleTogglePinChat}
            onDeleteChat={handleDeleteChat}
            documentsCount={documents.length}
            readyDocumentsCount={readyDocuments.length}
            onOpenDocuments={() => setShowDocumentsModal(true)}
            onClearHistory={() => setShowClearHistoryDialog(true)}
            onClearVectorStore={() => setShowClearVectorDialog(true)}
            onOpenSettings={() => setShowSettingsModal(true)}
          />
        }
        main={mainContent}
        inspect={inspectPanel}
        rightPanelOpen={sidePanelOpen}
      />

      {showDocumentsModal ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-3 backdrop-blur-sm dark:bg-black/70">
          <div className="flex max-h-[90vh] w-full max-w-4xl flex-col rounded-xl border border-border bg-background shadow-soft">
            <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
              <div>
                <p className="text-sm font-semibold text-foreground">Documents</p>
                <p className="text-xs text-muted-foreground">
                  {readyDocuments.length} ready of {documents.length} uploaded
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setShowDocumentsModal(false)}
                className="text-muted-foreground hover:text-foreground"
                title="Close documents"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 dark:[&_.bg-blue-50]:!bg-primary/10 dark:[&_.bg-emerald-50]:!bg-success/10 dark:[&_.bg-rose-50]:!bg-destructive/10 dark:[&_.bg-slate-100]:!bg-card dark:[&_.bg-slate-50]:!bg-card dark:[&_.bg-white]:!bg-background dark:[&_.border-blue-300]:!border-primary/30 dark:[&_.border-emerald-200]:!border-success/25 dark:[&_.border-rose-200]:!border-destructive/25 dark:[&_.border-slate-100]:!border-border dark:[&_.border-slate-200]:!border-border dark:[&_.border-slate-300]:!border-muted dark:[&_.text-blue-600]:!text-primary dark:[&_.text-emerald-700]:!text-success dark:[&_.text-rose-700]:!text-destructive dark:[&_.text-slate-400]:!text-muted-foreground dark:[&_.text-slate-500]:!text-muted-foreground dark:[&_.text-slate-600]:!text-muted-foreground dark:[&_.text-slate-700]:!text-foreground dark:[&_.text-slate-800]:!text-foreground dark:[&_.text-slate-900]:!text-foreground">
              <DocumentUploadCard
                documents={documents}
                activeDocument={activeDocument}
                isUploading={isUploading}
                isDeletingDocuments={isDeletingDocuments}
                deletingDocumentId={deletingDocumentId}
                uploadMessage={uploadMessage}
                uploadError={uploadError}
                uploadBatchItems={uploadBatchItems}
                uploadBatchSummary={uploadBatchSummary}
                onUploadFiles={uploadFiles}
                onRequestDeleteDocument={(document) => setSelectedDocumentForDelete(document)}
                onRequestDeleteAllDocuments={() => setShowClearVectorDialog(true)}
              />
              <QueryFilters
                documents={documents}
                selectedDocIds={selectedFilterDocIds}
                onSelectedDocIdsChange={setSelectedFilterDocIds}
                selectedFileTypes={selectedFileTypes}
                onSelectedFileTypesChange={setSelectedFileTypes}
                uploadedAfter={uploadedAfter}
                onUploadedAfterChange={setUploadedAfter}
                uploadedBefore={uploadedBefore}
                onUploadedBeforeChange={setUploadedBefore}
                ocrFilter={ocrFilter}
                onOcrFilterChange={setOcrFilter}
                disabled={isLoading}
              />
            </div>
          </div>
        </div>
      ) : null}

      <CitationPreviewModal
        item={previewCitation}
        query={submittedQuery}
        onClose={() => setPreviewCitation(null)}
      />

      {toastMessage ? (
        <div className="fixed bottom-5 right-5 z-[60] flex items-center gap-2 rounded-xl border border-success/20 bg-card px-4 py-3 text-sm text-success shadow-soft">
          <CheckCircle2 className="h-4 w-4 text-success" />
          {toastMessage}
        </div>
      ) : null}

      <SettingsModal
        open={showSettingsModal}
        onOpenChange={setShowSettingsModal}
        chunkSize={chunkSize}
        chunkOverlap={chunkOverlap}
        retrievalMode={retrievalMode}
        topK={topK}
        rerankTopN={rerankTopN}
        contextTopK={contextTopK}
        uploadedDocumentsCount={documents.length}
        onSettingsChange={handleSettingsChange}
      />

      <AlertDialog
        open={showClearHistoryDialog}
        onOpenChange={setShowClearHistoryDialog}
        title="Xóa lịch sử chat"
        description="Bạn có chắc chắn muốn xóa toàn bộ lịch sử chat? Hành động này không thể hoàn tác."
        onConfirm={handleClearHistory}
        confirmText="Xóa"
        cancelText="Hủy"
        variant="destructive"
      />

      <AlertDialog
        open={showClearVectorDialog}
        onOpenChange={setShowClearVectorDialog}
        title="Xóa tài liệu đã tải"
        description="Thao tác này sẽ xóa toàn bộ tài liệu đã upload và chỉ mục truy hồi."
        onConfirm={handleClearVectorStore}
        confirmText={isDeletingAllDocuments ? "Đang xóa..." : "Xóa"}
        cancelText="Hủy"
        variant="destructive"
        confirmDisabled={isDeletingAllDocuments}
        cancelDisabled={isDeletingAllDocuments}
      />

      <AlertDialog
        open={selectedDocumentForDelete !== null}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedDocumentForDelete(null);
          }
        }}
        title="Xóa tài liệu"
        description={
          selectedDocumentForDelete
            ? `Bạn có chắc chắn muốn xóa "${selectedDocumentForDelete.filename}"? Chỉ mục truy hồi sẽ được cập nhật lại.`
            : "Bạn có chắc chắn muốn xóa tài liệu này?"
        }
        onConfirm={handleDeleteSingleDocument}
        confirmText={isDeletingSingleDocument ? "Đang xóa..." : "Xóa"}
        cancelText="Hủy"
        variant="destructive"
        confirmDisabled={isDeletingSingleDocument}
        cancelDisabled={isDeletingSingleDocument}
      />
    </>
  );
}
