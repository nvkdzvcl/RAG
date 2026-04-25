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
};

function isCompare(result: QueryResult): result is Extract<QueryResult, { mode: "compare" }> {
  return result.mode === "compare";
}

export function ChatPanel({ messages, result, isLoading, error, notice = null }: ChatPanelProps) {
  const userMessages = messages.filter((message) => message.role === "user");
  const showChatMessages = userMessages.length > 0 || result;

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
        <ChatMessage role="assistant" content={translations.answer.loading} />
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
      
      {result && isCompare(result) ? <CompareView result={result} /> : null}
    </div>
  );
}
