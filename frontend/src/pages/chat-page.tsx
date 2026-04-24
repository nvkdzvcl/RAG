import { useState } from "react";

import { postQuery } from "@/api/client";
import { apiToUi } from "@/api/transform";
import { AppShell } from "@/components/dashboard/app-shell";
import { ChatComposer } from "@/components/dashboard/chat-composer";
import { ChatPanel } from "@/components/dashboard/chat-panel";
import { DocumentUploadCard } from "@/components/dashboard/document-upload-card";
import { MetricCards } from "@/components/dashboard/metric-cards";
import { ModeSelector } from "@/components/dashboard/mode-selector";
import { Sidebar, type RecentChat } from "@/components/dashboard/sidebar";
import { SourcesPanel } from "@/components/dashboard/sources-panel";
import { WorkflowTrace } from "@/components/dashboard/workflow-trace";
import { useDocumentIngestion } from "@/hooks/use-document-ingestion";
import type { Mode, QueryResult } from "@/types/chat";

const RECENT_CHAT_SEED: RecentChat[] = [
  { id: "recent-1", title: "Compare grounding quality across modes", mode: "compare", timeLabel: "12m ago" },
  { id: "recent-2", title: "How does advanced retry loop work?", mode: "advanced", timeLabel: "38m ago" },
  { id: "recent-3", title: "What is baseline RAG?", mode: "standard", timeLabel: "1h ago" },
];

export function ChatPage() {
  const [mode, setMode] = useState<Mode>("standard");
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<Array<Record<string, string>>>([]);
  const [recentChats, setRecentChats] = useState<RecentChat[]>(RECENT_CHAT_SEED);
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
      setError(queryDisabledReason || "Upload and process a document before querying.");
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
      });
      const mapped = apiToUi(payload);
      setResult(mapped);
      upsertRecentChat(normalized, mode);

      const assistantMessage = mapped.mode === "compare" ? mapped.comparison.note || "Compare run completed." : mapped.answer;
      setChatHistory((previous) =>
        [...previous, { role: "user", content: normalized }, { role: "assistant", content: assistantMessage }].slice(-12),
      );
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Unknown request error";
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

  const handleSelectRecent = (chat: RecentChat) => {
    setMode(chat.mode);
    setQuery(chat.title);
  };

  const mainContent = (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-4">
      <ModeSelector mode={mode} onModeChange={setMode} disabled={isLoading} />
      <MetricCards mode={mode} result={result} isLoading={isLoading} />
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
    <AppShell
      sidebar={
        <Sidebar
          mode={mode}
          onNewChat={resetConversation}
          recentChats={recentChats}
          onSelectRecent={handleSelectRecent}
        />
      }
      main={mainContent}
      inspect={inspectPanel}
    />
  );
}
