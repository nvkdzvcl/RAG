import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Citation, SingleMode } from "@/types/chat";
import { citationPanelId } from "@/components/rag/citation-utils";
import { MessageActions } from "@/components/rag/message-actions";

type MessageBubbleProps = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  branchKey?: SingleMode;
  eyebrow?: string;
  badges?: string[];
  children?: ReactNode;
  onCitationClick?: (panelId: string) => void;
  onCopied?: () => void;
  onRegenerate?: () => void;
  onToggleDebug?: () => void;
  canRegenerate?: boolean;
  showDebug?: boolean;
  isStreaming?: boolean;
};

function CitationMarker({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-md border border-primary/35 bg-primary/15 px-1.5 py-0.5 font-mono text-[11px] font-semibold leading-none text-primary transition duration-150 ease-out hover:border-primary hover:bg-primary/25 hover:text-primary active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      title="Open citation"
    >
      {label}
    </button>
  );
}

export function InlineCitationText({
  content,
  citations,
  branchKey,
  onCitationClick,
}: {
  content: string;
  citations: Citation[];
  branchKey: SingleMode | undefined;
  onCitationClick?: (panelId: string) => void;
}) {
  if (citations.length === 0 || !branchKey) {
    return <>{content}</>;
  }

  const markerRegex = /\[(\d+)\]/g;
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let hasCitationMarker = false;
  let match: RegExpExecArray | null;

  while ((match = markerRegex.exec(content)) !== null) {
    const markerNumber = Number(match[1]);
    const citationIndex = markerNumber - 1;
    const citation = citations[citationIndex];
    if (!citation) {
      continue;
    }
    hasCitationMarker = true;
    if (match.index > lastIndex) {
      nodes.push(content.slice(lastIndex, match.index));
    }
    nodes.push(
      <CitationMarker
        key={`${match.index}-${markerNumber}`}
        label={`[${markerNumber}]`}
        onClick={() => onCitationClick?.(citationPanelId(branchKey, citation, citationIndex))}
      />,
    );
    lastIndex = markerRegex.lastIndex;
  }

  if (hasCitationMarker) {
    if (lastIndex < content.length) {
      nodes.push(content.slice(lastIndex));
    }
    return <>{nodes}</>;
  }

  return (
    <>
      {content}
      <span className="ml-1 inline-flex flex-wrap align-baseline">
        {citations.map((citation, index) => (
          <CitationMarker
            key={citationPanelId(branchKey, citation, index)}
            label={`[${index + 1}]`}
            onClick={() => onCitationClick?.(citationPanelId(branchKey, citation, index))}
          />
        ))}
      </span>
    </>
  );
}

export function MessageBubble({
  role,
  content,
  citations = [],
  branchKey,
  eyebrow,
  badges = [],
  children,
  onCitationClick,
  onCopied,
  onRegenerate,
  onToggleDebug,
  canRegenerate = false,
  showDebug = false,
  isStreaming = false,
}: MessageBubbleProps) {
  const isUser = role === "user";
  const avatar = isUser
    ? { src: "/avatars/user.jpg", alt: "User avatar" }
    : { src: "/avatars/ai.png", alt: "Assistant avatar" };
  const avatarElement = (
    <img
      src={avatar.src}
      alt={avatar.alt}
      className={cn(
        "mt-1 h-9 w-9 shrink-0 rounded-full object-cover shadow-sm",
        isUser ? "ring-2 ring-primary/30" : "ring-2 ring-accent/25",
      )}
    />
  );

  return (
    <article
      className={cn(
        "animate-message-in group/message flex w-full items-start gap-3",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      {!isUser ? avatarElement : null}
      <div
        className={cn(
          "max-w-[min(820px,92%)] rounded-xl border px-4 py-3 shadow-subtle transition-colors",
          isUser
            ? "border-primary/30 bg-primary-light text-secondary-foreground"
            : "border-border bg-card/95 text-card-foreground",
        )}
      >
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase text-muted-foreground">
              {eyebrow ?? (isUser ? "You" : "Assistant")}
            </span>
            {badges.map((badge) => (
              <Badge
                key={badge}
                variant="outline"
                className="border-border bg-muted/50 px-2 py-0 text-[10px] text-muted-foreground"
              >
                {badge}
              </Badge>
            ))}
          </div>
          {!isUser ? (
            <MessageActions
              content={content}
              onCopied={onCopied}
              onRegenerate={onRegenerate}
              onToggleDebug={onToggleDebug}
              canRegenerate={canRegenerate}
              showDebug={showDebug}
            />
          ) : null}
        </div>
        <div className="whitespace-pre-wrap text-[15px] leading-7 text-foreground">
          <InlineCitationText
            content={content}
            citations={citations}
            branchKey={branchKey}
            onCitationClick={onCitationClick}
          />
          {isStreaming ? (
            <span aria-hidden="true" className="stream-caret ml-1 inline-block h-4 w-[2px] rounded-full bg-primary" />
          ) : null}
        </div>
        {children ? <div className="mt-4">{children}</div> : null}
      </div>
      {isUser ? avatarElement : null}
    </article>
  );
}
