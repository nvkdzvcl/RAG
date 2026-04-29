import { AlertTriangle, Database, FileSearch, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { CompareResult, ModeResult, QueryResult } from "@/types/chat";
import type { ChatSessionMessage } from "@/types/chat-session";
import { InlineCitationText, MessageBubble } from "@/components/rag/message-bubble";

type StreamProgress = {
  stage: "starting" | "retrieving" | "reranking" | "generating" | "grounding";
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

type ChatContainerProps = {
  messages: ChatSessionMessage[];
  result: QueryResult | null;
  isLoading: boolean;
  error: string | null;
  notice?: string | null;
  streamProgress?: StreamProgress | null;
  canQuery: boolean;
  onOpenDocuments: () => void;
  onCitationClick: (panelId: string) => void;
  onCopied: () => void;
  onRegenerate: () => void;
  onToggleDebug: () => void;
};

function isCompare(result: QueryResult): result is CompareResult {
  return result.mode === "compare";
}

function confidenceLabel(value: number | null): string {
  return value === null ? "n/a" : `${Math.round(value * 100)}%`;
}

function latencyLabel(value: number | null): string {
  return value === null ? "n/a" : `${value}ms`;
}

function modeBadges(result: ModeResult): string[] {
  const badges = [
    result.mode,
    result.status,
    `confidence ${confidenceLabel(result.confidence)}`,
    `grounded ${Math.round(result.groundedScore * 100)}%`,
    `latency ${latencyLabel(result.latencyMs)}`,
    `${result.citationCount} cites`,
  ];
  if (result.loopCount !== null) {
    badges.push(`${result.loopCount} loops`);
  }
  if (result.hallucinationDetected) {
    badges.push("review grounding");
  }
  return badges;
}

function stageLabel(stage: StreamProgress["stage"]): string {
  if (stage === "retrieving") return "Retrieving";
  if (stage === "reranking") return "Reranking";
  if (stage === "generating") return "Generating";
  if (stage === "grounding") return "Grounding";
  return "Starting";
}

function StageChip({ active, label }: { active: boolean; label: string }) {
  return (
    <span
      className={cn(
        "rounded-lg border px-2 py-1 text-[11px] font-semibold",
        active
          ? "border-success/30 bg-success/10 text-success"
          : "border-border bg-muted/40 text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}

function LoadingTrace({ streamProgress }: { streamProgress: StreamProgress | null | undefined }) {
  const stage = streamProgress?.stage ?? "starting";
  return (
    <div className="rounded-xl border border-border bg-primary-light/60 p-3">
      <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
        {stageLabel(stage)}
      </div>
      <div className="flex flex-wrap gap-2">
        <StageChip active={streamProgress?.stageReached.retrieving ?? false} label="Retrieval" />
        <StageChip active={streamProgress?.stageReached.reranking ?? false} label="Rerank" />
        <StageChip active={streamProgress?.stageReached.generating ?? false} label="LLM" />
        <StageChip active={streamProgress?.stageReached.grounding ?? false} label="Grounding" />
      </div>
      <div className="mt-3 flex flex-wrap gap-3 font-mono text-[11px] text-muted-foreground">
        <span>ttft={streamProgress?.timeToFirstTokenMs ?? "n/a"}ms</span>
        <span>total={streamProgress?.totalLatencyMs ?? "n/a"}ms</span>
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-2">
      <div className="h-3 w-11/12 animate-pulse rounded-full bg-muted" />
      <div className="h-3 w-4/5 animate-pulse rounded-full bg-muted" />
      <div className="h-3 w-2/3 animate-pulse rounded-full bg-muted" />
    </div>
  );
}

function BranchResult({
  title,
  result,
  onCitationClick,
}: {
  title: string;
  result: ModeResult;
  onCitationClick: (panelId: string) => void;
}) {
  return (
    <section className="min-w-0 rounded-xl border border-border bg-card/80 p-3">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <Badge variant="outline" className="border-border bg-muted/50 text-[11px] text-muted-foreground">
          {latencyLabel(result.latencyMs)}
        </Badge>
        <Badge variant="outline" className="border-border bg-muted/50 text-[11px] text-muted-foreground">
          {result.citationCount} cites
        </Badge>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-7 text-foreground">
        <InlineCitationText
          content={result.answer}
          citations={result.citations}
          branchKey={result.mode}
          onCitationClick={onCitationClick}
        />
      </p>
    </section>
  );
}

function comparisonSummary(result: CompareResult): string {
  if (result.comparison.note?.trim()) {
    return result.comparison.note.trim();
  }
  if (result.comparison.winner === "advanced") {
    return "Advanced produced the stronger grounded answer for this query.";
  }
  if (result.comparison.winner === "standard") {
    return "Standard produced the stronger grounded answer for this query.";
  }
  if (result.comparison.winner === "both_weak") {
    return "Both branches need review because the available evidence is weak.";
  }
  return "Both branches are available for side-by-side review.";
}

function EmptyState({ canQuery, onOpenDocuments }: { canQuery: boolean; onOpenDocuments: () => void }) {
  return (
    <div className="flex min-h-[420px] items-center justify-center px-4">
      <div className="w-full max-w-xl rounded-xl border border-border bg-card/85 p-6 text-center shadow-subtle">
        <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-xl border border-primary/20 bg-gradient-to-br from-primary-light to-accent/10">
          {canQuery ? <FileSearch className="h-7 w-7 text-primary" /> : <Database className="h-7 w-7 text-primary" />}
        </div>
        <p className="text-lg font-semibold text-foreground">{canQuery ? "Ready for grounded answers" : "No indexed documents"}</p>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          {canQuery
            ? "Ask a question to inspect answers, citations, and workflow telemetry in one place."
            : "Add documents to build the retrieval index before asking questions."}
        </p>
        {!canQuery ? (
          <Button type="button" onClick={onOpenDocuments} className="mt-5">
            Open documents
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export function ChatContainer({
  messages,
  result,
  isLoading,
  error,
  notice = null,
  streamProgress = null,
  canQuery,
  onOpenDocuments,
  onCitationClick,
  onCopied,
  onRegenerate,
  onToggleDebug,
}: ChatContainerProps) {
  const skipLastAssistant = result && !isLoading && messages[messages.length - 1]?.role === "assistant";
  const visibleMessages = skipLastAssistant ? messages.slice(0, -1) : messages;
  const hasConversation = visibleMessages.length > 0 || result || isLoading;
  const streamedContent = streamProgress?.partialAnswer.trim() || "";

  return (
    <div className="min-h-0 flex-1 scroll-smooth overflow-y-auto px-4 py-6 md:px-6">
      <div className="mx-auto flex max-w-4xl flex-col gap-5">
        {!hasConversation ? <EmptyState canQuery={canQuery} onOpenDocuments={onOpenDocuments} /> : null}

        {visibleMessages.map((message) => (
          <MessageBubble
            key={`${message.role}-${message.timestamp}-${message.content.slice(0, 20)}`}
            role={message.role}
            content={message.content}
            eyebrow={message.role === "user" ? "You" : "Assistant"}
            onCopied={onCopied}
          />
        ))}

        {isLoading ? (
          <MessageBubble
            role="assistant"
            content={streamedContent || "Working on the answer..."}
            onCopied={onCopied}
            isStreaming
          >
            {streamedContent ? null : <LoadingSkeleton />}
            <LoadingTrace streamProgress={streamProgress} />
          </MessageBubble>
        ) : null}

        {error ? (
          <div className="rounded-xl border border-destructive/25 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <div className="flex gap-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          </div>
        ) : null}

        {notice ? (
          <div className="rounded-xl border border-warning/25 bg-warning/10 px-4 py-3 text-sm text-warning">
            {notice}
          </div>
        ) : null}

        {result && !isLoading && !isCompare(result) ? (
          <MessageBubble
            role="assistant"
            content={result.answer}
            citations={result.citations}
            branchKey={result.mode}
            eyebrow="Assistant"
            badges={modeBadges(result)}
            onCitationClick={onCitationClick}
            onCopied={onCopied}
            onRegenerate={onRegenerate}
            onToggleDebug={onToggleDebug}
            canRegenerate
            showDebug
          />
        ) : null}

        {result && !isLoading && isCompare(result) ? (
          <MessageBubble
            role="assistant"
            content={comparisonSummary(result)}
            eyebrow="Comparison"
            badges={[
              `winner ${result.comparison.winner ?? "n/a"}`,
              `standard ${confidenceLabel(result.standard.confidence)}`,
              `advanced ${confidenceLabel(result.advanced.confidence)}`,
              `delta ${latencyLabel(result.comparison.latencyDeltaMs)}`,
            ]}
            onCopied={onCopied}
            onRegenerate={onRegenerate}
            onToggleDebug={onToggleDebug}
            canRegenerate
            showDebug
          >
            <div className="grid gap-3 xl:grid-cols-2">
              <BranchResult title="Standard" result={result.standard} onCitationClick={onCitationClick} />
              <BranchResult title="Advanced" result={result.advanced} onCitationClick={onCitationClick} />
            </div>
            {result.comparison.reasons.length > 0 ? (
              <div className="mt-3 rounded-xl border border-border bg-card/80 p-3">
                <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">Comparison notes</p>
                <ul className="space-y-1 text-sm leading-6 text-muted-foreground">
                  {result.comparison.reasons.map((reason, index) => (
                    <li key={`${reason}-${index}`}>{reason}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </MessageBubble>
        ) : null}
      </div>
    </div>
  );
}
