import type { QueryResult } from "@/types/chat";
import { AnswerCard } from "@/components/dashboard/answer-card";
import { CompareView } from "@/components/dashboard/compare-view";
import { Card, CardContent } from "@/components/ui/card";
import { translations } from "@/lib/translations";
import { ChatMessage } from "@/components/dashboard/chat-message";

type ChatPanelProps = {
  submittedQuery: string | null;
  result: QueryResult | null;
  isLoading: boolean;
  error: string | null;
};

function isCompare(result: QueryResult): result is Extract<QueryResult, { mode: "compare" }> {
  return result.mode === "compare";
}

export function ChatPanel({ submittedQuery, result, isLoading, error }: ChatPanelProps) {
  // Show chat messages if we have a query
  const showChatMessages = submittedQuery || result;

  return (
    <div className="space-y-4">
      {!showChatMessages ? (
        <Card className="border-slate-200 shadow-sm">
          <CardContent className="py-12 text-center">
            <p className="text-sm text-slate-500">Gửi câu hỏi để xem kết quả từ các chế độ Self-RAG.</p>
          </CardContent>
        </Card>
      ) : null}

      {showChatMessages && submittedQuery ? (
        <ChatMessage role="user" content={submittedQuery} />
      ) : null}

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

      {result && !isLoading && !isCompare(result) ? (
        <div className="space-y-3">
          <ChatMessage role="assistant" content={result.answer} />
          <AnswerCard result={result} />
        </div>
      ) : null}
      
      {result && isCompare(result) ? <CompareView result={result} /> : null}
    </div>
  );
}
