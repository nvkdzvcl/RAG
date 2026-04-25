import { useEffect, useState } from "react";

import { getHealthStatus, postQuery } from "@/api/client";
import { apiToUi } from "@/api/transform";
import { AppShell } from "@/components/dashboard/app-shell";
import { ChatComposer } from "@/components/dashboard/chat-composer";
import { ChatPanel } from "@/components/dashboard/chat-panel";
import { DocumentUploadCard } from "@/components/dashboard/document-upload-card";
import { ModeSelector } from "@/components/dashboard/mode-selector";
import { Sidebar, type RecentChat } from "@/components/dashboard/sidebar";
import { SourcesPanel } from "@/components/dashboard/sources-panel";
import { WorkflowTrace } from "@/components/dashboard/workflow-trace";
import { SettingsModal } from "@/components/dashboard/settings-modal";
import { AlertDialog } from "@/components/ui/alert-dialog";
import { useDocumentIngestion } from "@/hooks/use-document-ingestion";
import type { Mode, QueryResult } from "@/types/chat";

const RECENT_CHAT_SEED: RecentChat[] = [
  { id: "recent-1", title: "So sánh chất lượng grounding giữa các chế độ", mode: "compare", timeLabel: "12 phút trước" },
  { id: "recent-2", title: "Vòng lặp retry của chế độ nâng cao hoạt động như thế nào?", mode: "advanced", timeLabel: "38 phút trước" },
  { id: "recent-3", title: "RAG cơ bản là gì?", mode: "standard", timeLabel: "1 giờ trước" },
];

const DEFAULT_MODEL =
  (typeof import.meta.env.VITE_DEFAULT_LLM_MODEL === "string" &&
  import.meta.env.VITE_DEFAULT_LLM_MODEL.trim().length > 0
    ? import.meta.env.VITE_DEFAULT_LLM_MODEL.trim()
    : "qwen2.5:3b");

export function ChatPage() {
  const [mode, setMode] = useState<Mode>("standard");
  const [selectedModel, setSelectedModel] = useState<string>(DEFAULT_MODEL);
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<Array<Record<string, string>>>([]);
  const [recentChats, setRecentChats] = useState<RecentChat[]>(RECENT_CHAT_SEED);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [topK, setTopK] = useState(10);
  const [showClearHistoryDialog, setShowClearHistoryDialog] = useState(false);
  const [showClearVectorDialog, setShowClearVectorDialog] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  
  const {
    documents,
    activeDocument,
    isUploading,
    uploadMessage,
    uploadError,
    canQuery,
    queryDisabledReason,
    uploadFile,
  } = useDocumentIngestion();

  useEffect(() => {
    let mounted = true;

    const loadBackendDefaultModel = async () => {
      try {
        const health = await getHealthStatus();
        if (!mounted) {
          return;
        }
        if (typeof health.llm_model === "string" && health.llm_model.trim().length > 0) {
          setSelectedModel(health.llm_model.trim());
        }
      } catch {
        // Keep local default model when backend health/default model is unavailable.
      }
    };

    void loadBackendDefaultModel();
    return () => {
      mounted = false;
    };
  }, []);

  const canSubmit = query.trim().length > 0 && !isLoading && canQuery;

  const upsertRecentChat = (text: string, selectedMode: Mode) => {
    const timeLabel = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setRecentChats((previous) => {
      const nextEntry: RecentChat = {
        id: `${Date.now()}-${selectedMode}`,
        title: text,
        mode: selectedMode,
        timeLabel,
      };
      const deduped = previous.filter((item) => !(item.title === text && item.mode === selectedMode));
      return [nextEntry, ...deduped].slice(0, 6);
    });
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

    setIsLoading(true);
    setError(null);
    setSubmittedQuery(normalized);

    try {
      const payload = await postQuery({
        query: normalized,
        mode,
        chat_history: chatHistory,
        model: selectedModel,
      });
      const mapped = apiToUi(payload);
      setResult(mapped);
      setQuery("");
      upsertRecentChat(normalized, mode);

      const assistantMessage = mapped.mode === "compare" ? mapped.comparison.note || "Hoàn tất so sánh." : mapped.answer;
      setChatHistory((previous) =>
        [...previous, { role: "user", content: normalized }, { role: "assistant", content: assistantMessage }].slice(-12),
      );
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Lỗi yêu cầu không xác định";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const resetConversation = () => {
    setQuery("");
    setSubmittedQuery(null);
    setResult(null);
    setError(null);
    setChatHistory([]);
  };

  const handleClearHistory = () => {
    setRecentChats([]);
    resetConversation();
  };

  const handleClearVectorStore = async () => {
    // TODO: Call API to clear vector store
    console.log("Clear vector store - to be implemented");
    // For now, just show a message
    alert("Tính năng xóa vector store sẽ được triển khai trong phiên bản tiếp theo");
  };

  const handleChunkSettingsChange = (newChunkSize: number, newChunkOverlap: number, newTopK: number) => {
    setChunkSize(newChunkSize);
    setChunkOverlap(newChunkOverlap);
    setTopK(newTopK);
    console.log(`Settings updated: chunkSize=${newChunkSize}, chunkOverlap=${newChunkOverlap}, topK=${newTopK}`);
  };

  const handleSelectRecent = (chat: RecentChat) => {
    setMode(chat.mode);
    setQuery(chat.title);
  };

  const mainContent = (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-4">
      <ModeSelector
        mode={mode}
        onModeChange={setMode}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        disabled={isLoading}
      />
      <DocumentUploadCard
        documents={documents}
        activeDocument={activeDocument}
        isUploading={isUploading}
        uploadMessage={uploadMessage}
        uploadError={uploadError}
        onUpload={uploadFile}
      />
      <ChatPanel submittedQuery={submittedQuery} result={result} isLoading={isLoading} error={error} />
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
            onNewChat={resetConversation}
            recentChats={recentChats}
            onSelectRecent={handleSelectRecent}
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
        topK={topK}
        onSettingsChange={handleChunkSettingsChange}
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
        description="Bạn có chắc chắn muốn xóa toàn bộ tài liệu và vector store? Hành động này không thể hoàn tác."
        onConfirm={handleClearVectorStore}
        confirmText="Xóa"
        cancelText="Hủy"
        variant="destructive"
      />
    </>
  );
}
