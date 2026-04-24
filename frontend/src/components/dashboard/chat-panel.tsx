import type { QueryResult } from "@/types/chat";
import { AnswerCard } from "@/components/dashboard/answer-card";
import { CompareView } from "@/components/dashboard/compare-view";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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
  return (
    <div className="space-y-4">
      <Card className="border-slate-200 shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">User Query</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-7 text-slate-700">
            {submittedQuery || "Submit a query to see mode-specific Self-RAG outputs."}
          </p>
        </CardContent>
      </Card>

      {isLoading ? (
        <Card className="border-slate-200 shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Assistant Answer</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">Running retrieval, reranking, and generation...</p>
          </CardContent>
        </Card>
      ) : null}

      {error ? (
        <Card className="border-rose-200 bg-rose-50 shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-base text-rose-700">Request Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-rose-700">{error}</p>
          </CardContent>
        </Card>
      ) : null}

      {result && !isCompare(result) ? <AnswerCard result={result} /> : null}
      {result && isCompare(result) ? <CompareView result={result} /> : null}
    </div>
  );
}
