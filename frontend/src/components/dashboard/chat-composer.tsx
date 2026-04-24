import { SendHorizonal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type ChatComposerProps = {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  canSubmit: boolean;
  disabled?: boolean;
  disabledReason?: string | null;
};

export function ChatComposer({
  query,
  onQueryChange,
  onSubmit,
  isLoading,
  canSubmit,
  disabled = false,
  disabledReason = null,
}: ChatComposerProps) {
  const composerDisabled = isLoading || disabled;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <Textarea
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder={composerDisabled && disabledReason ? disabledReason : "Ask a question about your documents..."}
        disabled={composerDisabled}
        className="min-h-[110px] resize-y border-slate-200 bg-white"
        onKeyDown={(event) => {
          if (composerDisabled) {
            return;
          }
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            onSubmit();
          }
        }}
      />
      <div className="mt-3 flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {disabled && disabledReason ? disabledReason : "Press Ctrl/Cmd + Enter to submit"}
        </p>
        <Button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit || composerDisabled}
          className="gap-2 bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:opacity-95"
        >
          <SendHorizonal className="h-4 w-4" />
          {isLoading ? "Running..." : "Send Query"}
        </Button>
      </div>
    </div>
  );
}
