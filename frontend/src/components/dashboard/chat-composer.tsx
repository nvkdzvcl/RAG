import { useEffect, useRef } from "react";
import { SendHorizonal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { translations } from "@/lib/translations";

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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 320)}px`;
  }, [query]);

  return (
    <div className="rounded-3xl border border-slate-200/90 bg-white/95 p-4 shadow-[0_16px_48px_rgba(15,23,42,0.12)]">
      <Textarea
        ref={textareaRef}
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder={composerDisabled && disabledReason ? disabledReason : translations.chat.placeholder}
        disabled={composerDisabled}
        className="min-h-[140px] max-h-[320px] resize-none rounded-2xl border-slate-200 bg-white px-4 py-3 text-[15px] leading-7 shadow-inner"
        onKeyDown={(event) => {
          if (composerDisabled) {
            return;
          }
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            onSubmit();
            return;
          }
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            onSubmit();
          }
        }}
      />
      <div className="mt-3 flex items-center justify-end">
        <Button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit || composerDisabled}
          className="h-11 rounded-full px-5 font-semibold gap-2 bg-primary text-white shadow-lg shadow-blue-200 hover:bg-primary/90"
        >
          <SendHorizonal className="h-4 w-4" />
          {isLoading ? translations.chat.sending : translations.chat.send}
        </Button>
      </div>
    </div>
  );
}
