import { useState } from "react";

import { postQuery } from "@/api/client";
import { apiToUi } from "@/api/transform";
import { AnswerPanel } from "@/components/chat/answer-panel";
import { ChatInput } from "@/components/chat/chat-input";
import { CitationsPanel } from "@/components/chat/citations-panel";
import { CompareLayout } from "@/components/chat/compare-layout";
import { ModeSelector } from "@/components/chat/mode-selector";
import { SourcesPanel } from "@/components/chat/sources-panel";
import { WorkflowTracePanel } from "@/components/chat/workflow-trace-panel";
import type { Mode, QueryResult } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function isCompareResult(result: QueryResult): result is Extract<QueryResult, { mode: "compare" }> {
  return result.mode === "compare";
}

export function ChatPage() {
  const [mode, setMode] = useState<Mode>("standard");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<Array<Record<string, string>>>([]);

  const canSubmit = query.trim().length > 0 && !isLoading;

  const runQuery = async () => {
    const normalized = query.trim();
    if (!normalized || isLoading) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const payload = await postQuery({
        query: normalized,
        mode,
        chat_history: chatHistory,
      });
      const mapped = apiToUi(payload);
      setResult(mapped);

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

  return (
    <main className="mx-auto min-h-screen w-full max-w-7xl space-y-4 px-4 py-6 md:px-6 md:py-8">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">Self-RAG Query Console</h1>
        <p className="text-sm text-muted-foreground">Connected to backend query API for standard, advanced, and compare modes.</p>
      </header>

      <div className="grid gap-4 lg:grid-cols-[1.1fr_1fr]">
        <ModeSelector mode={mode} onModeChange={setMode} disabled={isLoading} />
        <ChatInput query={query} onQueryChange={setQuery} onSubmit={runQuery} isLoading={isLoading} canSubmit={canSubmit} />
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle>Usage</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Enter a question and click <span className="font-medium">Run Query</span> (or use Ctrl/Cmd+Enter).
          </p>
        </CardContent>
      </Card>

      {error ? (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Request Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      ) : null}

      {!canSubmit && !result && !error ? (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Ready</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Waiting for the first successful query response.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {result && isCompareResult(result) ? <CompareLayout result={result} /> : null}

      {result && !isCompareResult(result) ? (
        <div className="space-y-4">
          <AnswerPanel result={result} />
          <div className="grid gap-4 xl:grid-cols-2">
            <CitationsPanel citations={result.citations} />
            <SourcesPanel sources={result.sources} />
          </div>
          {result.mode === "advanced" ? <WorkflowTracePanel trace={result.trace} /> : null}
        </div>
      ) : null}
    </main>
  );
}
