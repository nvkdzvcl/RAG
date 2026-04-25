import { Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

type ChatInputProps = {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  isLoading?: boolean;
  canSubmit?: boolean;
};

export function ChatInput({ query, onQueryChange, onSubmit, isLoading = false, canSubmit = true }: ChatInputProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>Chat Input</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Ask a question about your knowledge base..."
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
              event.preventDefault();
              onSubmit();
              return;
            }
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSubmit();
            }
          }}
          disabled={isLoading}
        />
        <div className="flex justify-end">
          <Button type="button" onClick={onSubmit} className="gap-2" disabled={!canSubmit || isLoading}>
            <Send className="h-4 w-4" />
            <span>{isLoading ? "Running..." : "Run Query"}</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
