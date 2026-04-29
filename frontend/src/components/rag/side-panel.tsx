import { Bug, PanelRightClose, PanelRightOpen, Quote } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { QueryResult } from "@/types/chat";
import { CitationPanel } from "@/components/rag/citation-panel";
import { DebugPanel } from "@/components/rag/debug-panel";

export type SidePanelTab = "citations" | "debug";

type SidePanelProps = {
  open: boolean;
  tab: SidePanelTab;
  onOpenChange: (open: boolean) => void;
  onTabChange: (tab: SidePanelTab) => void;
  result: QueryResult | null;
  activeCitationId: string | null;
  query: string | null;
  onFocusCitation: (panelId: string) => void;
};

const TABS: Array<{ value: SidePanelTab; label: string; icon: LucideIcon }> = [
  { value: "citations", label: "Citations", icon: Quote },
  { value: "debug", label: "Debug", icon: Bug },
];

export function SidePanel({
  open,
  tab,
  onOpenChange,
  onTabChange,
  result,
  activeCitationId,
  query,
  onFocusCitation,
}: SidePanelProps) {
  if (!open) {
    return (
      <div className="animate-side-rail-in flex h-full flex-col items-center border-l border-border bg-background/70 py-4">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={() => onOpenChange(true)}
          className="text-muted-foreground hover:text-foreground"
          title="Open side panel"
        >
          <PanelRightOpen className="h-4 w-4" />
        </Button>
        <div className="mt-4 flex flex-col gap-2">
          {TABS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.value}
                type="button"
                onClick={() => {
                  onTabChange(item.value);
                  onOpenChange(true);
                }}
                className="rounded-xl p-2 text-muted-foreground transition duration-150 ease-out hover:-translate-y-px hover:bg-muted/60 hover:text-foreground active:translate-y-0 active:scale-95"
                title={item.label}
              >
                <Icon className="h-4 w-4" />
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-side-panel-in flex h-full min-h-0 flex-col border-l border-border bg-background/70">
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-background/60 px-3 backdrop-blur-xl">
        <div className="flex rounded-xl border border-border bg-card/80 p-1 shadow-subtle">
          {TABS.map((item) => {
            const active = item.value === tab;
            const Icon = item.icon;
            return (
              <button
                key={item.value}
                type="button"
                onClick={() => onTabChange(item.value)}
                className={cn(
                  "inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-xs font-semibold transition duration-150 ease-out hover:-translate-y-px active:translate-y-0 active:scale-[0.98]",
                  active ? "bg-primary text-primary-foreground shadow-sm shadow-primary/25" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {item.label}
              </button>
            );
          })}
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={() => onOpenChange(false)}
          className="text-muted-foreground hover:text-foreground"
          title="Collapse side panel"
        >
          <PanelRightClose className="h-4 w-4" />
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {tab === "citations" ? (
          <CitationPanel
            result={result}
            activeCitationId={activeCitationId}
            query={query}
            onFocusCitation={onFocusCitation}
          />
        ) : (
          <DebugPanel result={result} />
        )}
      </div>
    </div>
  );
}
