import { useEffect, useMemo, useRef, useState } from "react";

import { getHealthStatus, postQuery } from "@/api/client";
import type { ApiRetrievalMode } from "@/api/types";
import { apiToUi } from "@/api/transform";
import { AppShell } from "@/components/dashboard/app-shell";
import { ChatComposer } from "@/components/dashboard/chat-composer";
import { ChatPanel } from "@/components/dashboard/chat-panel";
import { DocumentUploadCard } from "@/components/dashboard/document-upload-card";
import { ModeSelector } from "@/components/dashboard/mode-selector";
import {
  QueryFilters,
  type QueryFileTypeFilter,
  type QueryOcrFilter,
} from "@/components/dashboard/query-filters";
import { Sidebar, type RecentChat } from "@/components/dashboard/sidebar";
import { SourcesPanel } from "@/components/dashboard/sources-panel";
import { WorkflowTrace } from "@/components/dashboard/workflow-trace";
import { SettingsModal, type SettingsChangePayload } from "@/components/dashboard/settings-modal";
import { AlertDialog } from "@/components/ui/alert-dialog";
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
  const documentsSectionRef = useRef<HTMLDivElement | null>(null);

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
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
      .map((item) => ({
        id: item.id,
        title: item.title,
        mode: item.mode,
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

  const runQuery = async () => {
    const normalized = query.trim();
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

    try {
      const selectedDocuments = selectedFilterDocIds.length > 0
        ? readyDocuments.filter((item) => selectedFilterDocIds.includes(item.id))
        : [];
      const payload = await postQuery({
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
      });

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
      const message = requestError instanceof Error ? requestError.message : "Lỗi yêu cầu không xác định";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    createFreshSession(mode, selectedModel);
  };

  const handleClearHistory = () => {
    clearChatSessionsStorage();
    const fresh = createChatSession({ mode, model: selectedModel });
    setSessions([fresh]);
    setActiveSessionId(fresh.id);
    setQuery("");
    setError(null);
    setNotice(null);
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

  const mainContent = (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-4">
      <ModeSelector
        mode={mode}
        onModeChange={handleModeChange}
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
        disabled={isLoading}
      />
      <div ref={documentsSectionRef}>
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
      </div>
      <ChatPanel messages={messages} result={result} isLoading={isLoading} error={error} notice={notice} />
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
      <ChatComposer
        query={query}
        onQueryChange={setQuery}
        onSubmit={runQuery}
        isLoading={isLoading}
        canSubmit={canSubmit}
        disabled={!canQuery}
        disabledReason={queryDisabledReason}
      />
    </div>
  );

  const inspectPanel = (
    <div className="space-y-4">
      <SourcesPanel result={result} />
      <WorkflowTrace result={result} />
    </div>
  );

  return (
    <>
      <AppShell
        sidebar={
          <Sidebar
            mode={mode}
            onNewChat={handleNewChat}
            recentChats={recentChats}
            onSelectRecent={handleSelectRecent}
            onOpenDocuments={() => {
              documentsSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
            onClearHistory={() => setShowClearHistoryDialog(true)}
            onClearVectorStore={() => setShowClearVectorDialog(true)}
            onOpenSettings={() => setShowSettingsModal(true)}
          />
        }
        main={mainContent}
        inspect={inspectPanel}
      />

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
