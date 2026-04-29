import { useEffect, useRef } from "react";
import { SendHorizonal, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type ComposerProps = {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isLoading: boolean;
  canSubmit: boolean;
  disabled?: boolean;
  disabledReason?: string | null;
};

export function Composer({
  query,
  onQueryChange,
  onSubmit,
  onStop,
  isLoading,
  canSubmit,
  disabled = false,
  disabledReason = null,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const composerDisabled = disabled && !isLoading;

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [query]);

  return (
    <div className="border-t border-border bg-background/75 px-4 py-4 backdrop-blur-xl md:px-6">
      <div className="mx-auto max-w-4xl rounded-xl border border-primary/20 bg-card/95 p-2 shadow-soft">
        <Textarea
          ref={textareaRef}
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder={composerDisabled && disabledReason ? disabledReason : "Ask a grounded question about your documents..."}
          disabled={composerDisabled}
          className="max-h-[180px] min-h-[72px] resize-none border-0 bg-transparent px-3 py-3 text-[15px] leading-6 text-foreground shadow-none outline-none placeholder:text-muted-foreground focus-visible:ring-0"
          onKeyDown={(event) => {
            if (composerDisabled) {
              return;
            }
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (isLoading) {
                return;
              }
              onSubmit();
            }
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault();
              if (!isLoading) {
                onSubmit();
              }
            }
          }}
        />
        <div className="flex items-center justify-between gap-3 px-2 pb-1">
          <p className="text-xs text-muted-foreground">Enter to send, Shift+Enter for a new line</p>
          {isLoading ? (
            <Button
              type="button"
              variant="outline"
              onClick={onStop}
              className="h-9 gap-2 border-destructive/30 bg-destructive/10 text-destructive hover:bg-destructive/15"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              Stop
            </Button>
          ) : (
            <Button type="button" onClick={onSubmit} disabled={!canSubmit || composerDisabled} className="h-9 gap-2">
              <SendHorizonal className="h-4 w-4" />
              Send
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
