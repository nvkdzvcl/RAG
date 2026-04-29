import { Bug, Copy, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";

type MessageActionsProps = {
  content: string;
  onCopied?: () => void;
  onRegenerate?: () => void;
  onToggleDebug?: () => void;
  canRegenerate?: boolean;
  showDebug?: boolean;
};

export function MessageActions({
  content,
  onCopied,
  onRegenerate,
  onToggleDebug,
  canRegenerate = false,
  showDebug = false,
}: MessageActionsProps) {
  const handleCopy = async () => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      await navigator.clipboard.writeText(content);
      onCopied?.();
    }
  };

  return (
    <div className="flex items-center gap-1 opacity-0 transition-opacity duration-150 ease-out group-hover/message:opacity-100 group-focus-within/message:opacity-100">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={() => void handleCopy()}
        className="h-8 w-8 text-muted-foreground hover:text-foreground"
        title="Copy answer"
      >
        <Copy className="h-4 w-4" />
      </Button>
      {canRegenerate ? (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onRegenerate}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          title="Regenerate from the last question"
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
      ) : null}
      {showDebug ? (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onToggleDebug}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          title="Open debug panel"
        >
          <Bug className="h-4 w-4" />
        </Button>
      ) : null}
    </div>
  );
}
