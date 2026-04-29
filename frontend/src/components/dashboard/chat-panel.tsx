import type { QueryResult } from "@/types/chat";
import { AnswerCard } from "@/components/dashboard/answer-card";
import { CompareView } from "@/components/dashboard/compare-view";
import { Card, CardContent } from "@/components/ui/card";
import { translations } from "@/lib/translations";
import { ChatMessage } from "@/components/dashboard/chat-message";
import type { ChatSessionMessage } from "@/types/chat-session";

type ChatPanelProps = {
  messages: ChatSessionMessage[];
  result: QueryResult | null;
  isLoading: boolean;
  error: string | null;
  notice?: string | null;
  streamProgress?: {
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
  } | null;
};

function isCompare(result: QueryResult): result is Extract<QueryResult, { mode: "compare" }> {
  return result.mode === "compare";
}

function stageLabel(stage: "starting" | "retrieving" | "reranking" | "generating" | "grounding"): string {
  if (stage === "retrieving") {
    return "Đang truy xuất tài liệu";
  }
  if (stage === "reranking") {
    return "Đang xếp hạng lại ngữ cảnh";
  }
  if (stage === "generating") {
    return "Đang sinh câu trả lời";
  }
  if (stage === "grounding") {
    return "Đang kiểm tra grounding";
  }
  return "Đang khởi tạo truy vấn";
}

function stageBadgeStyle(active: boolean): string {
  return active
    ? "border-emerald-300 bg-emerald-50 text-emerald-700"
    : "border-slate-200 bg-slate-100 text-slate-500";
}

export function ChatPanel({
  messages,
  result,
  isLoading,
  error,
  notice = null,
  streamProgress = null,
}: ChatPanelProps) {
  const userMessages = messages.filter((message) => message.role === "user");
  const showChatMessages = userMessages.length > 0 || result;
  const loadingContent = streamProgress?.partialAnswer.trim()
    ? streamProgress.partialAnswer
    : translations.answer.loading;
  const ttftValue = streamProgress && streamProgress.timeToFirstTokenMs !== null
    ? String(streamProgress.timeToFirstTokenMs)
    : "n/a";
  const totalLatencyValue = streamProgress && streamProgress.totalLatencyMs !== null
    ? String(streamProgress.totalLatencyMs)
    : "n/a";

  return (
    <div className="space-y-4">
      {!showChatMessages ? (
        <Card className="border-slate-200 shadow-sm">
          <CardContent className="py-12 text-center">
            <p className="text-sm text-slate-500">Gửi câu hỏi để xem kết quả từ các chế độ Self-RAG.</p>
          </CardContent>
        </Card>
      ) : null}

      {userMessages.map((message) => (
        <ChatMessage key={`${message.role}-${message.timestamp}-${message.content.slice(0, 12)}`} role={message.role} content={message.content} />
      ))}

      {isLoading ? (
        <div className="space-y-2">
          <ChatMessage role="assistant" content={loadingContent} />
          <Card className="border-slate-200 bg-slate-50 shadow-sm">
            <CardContent className="space-y-2 py-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {streamProgress ? stageLabel(streamProgress.stage) : "Đang xử lý"}
              </p>
              <div className="flex flex-wrap gap-2">
                <span className={`rounded-full border px-2 py-1 text-xs ${stageBadgeStyle(streamProgress?.stageReached.retrieving ?? false)}`}>
                  Retrieving
                </span>
                <span className={`rounded-full border px-2 py-1 text-xs ${stageBadgeStyle(streamProgress?.stageReached.reranking ?? false)}`}>
                  Reranking
                </span>
                <span className={`rounded-full border px-2 py-1 text-xs ${stageBadgeStyle(streamProgress?.stageReached.generating ?? false)}`}>
                  Generating
                </span>
                <span className={`rounded-full border px-2 py-1 text-xs ${stageBadgeStyle(streamProgress?.stageReached.grounding ?? false)}`}>
                  Grounding
                </span>
              </div>
              <div className="flex flex-wrap gap-3 text-xs text-slate-500">
                <span>
                  time_to_first_token_ms: {" "}
                  {ttftValue}
                </span>
                <span>
                  total_latency_ms: {" "}
                  {totalLatencyValue}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}

      {error ? (
        <Card className="border-rose-200 bg-rose-50 shadow-sm">
          <CardContent className="py-3">
            <p className="text-sm text-rose-700">❌ {error}</p>
          </CardContent>
        </Card>
      ) : null}

      {notice ? (
        <Card className="border-amber-200 bg-amber-50 shadow-sm">
          <CardContent className="py-3">
            <p className="text-sm text-amber-700">{notice}</p>
          </CardContent>
        </Card>
      ) : null}

      {result && !isLoading && !isCompare(result) ? (
        <div className="space-y-3">
          <AnswerCard result={result} />
        </div>
      ) : null}

      {result && !isLoading && isCompare(result) ? <CompareView result={result} /> : null}
    </div>
  );
}
