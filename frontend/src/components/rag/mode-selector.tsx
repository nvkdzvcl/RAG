import { Check, ChevronDown, GitCompareArrows, Gauge, ShieldCheck } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cn } from "@/lib/utils";
import type { Mode } from "@/types/chat";

type ModeSelectorProps = {
  workflowMode: Mode;
  onWorkflowModeChange: (mode: Mode) => void;
  selectedModel: string;
  onModelChange: (model: string) => void;
  disabled?: boolean;
};

type ModelOption = {
  value: "qwen2.5:3b" | "qwen2.5:7b" | "qwen3.5:4b" | "qwen3.5:9b";
  label: string;
  detail: string;
};

const WORKFLOW_OPTIONS: Array<{
  value: Mode;
  label: string;
  tooltip: string;
  icon: LucideIcon;
}> = [
  {
    value: "standard",
    label: "Standard",
    tooltip: "Baseline RAG: retrieve, rerank, select context, generate.",
    icon: Gauge,
  },
  {
    value: "advanced",
    label: "Advanced",
    tooltip: "Self-RAG workflow with retrieval gate, critique, retry/refine, and abstain path.",
    icon: ShieldCheck,
  },
  {
    value: "compare",
    label: "Compare",
    tooltip: "Runs Standard and Advanced for the same query and compares both outputs.",
    icon: GitCompareArrows,
  },
];

const MODEL_OPTIONS: ModelOption[] = [
  { value: "qwen2.5:3b", label: "Qwen 2.5 3B", detail: "Fast local baseline" },
  { value: "qwen2.5:7b", label: "Qwen 2.5 7B", detail: "Balanced quality" },
  { value: "qwen3.5:4b", label: "Qwen 3.5 4B", detail: "Light reasoning" },
  { value: "qwen3.5:9b", label: "Qwen 3.5 9B", detail: "Higher quality, slower" },
];

function Tooltip({ text }: { text: string }) {
  return (
    <span className="pointer-events-none absolute left-1/2 top-full z-40 mt-2 hidden w-56 -translate-x-1/2 rounded-xl border border-border bg-card px-3 py-2 text-left text-xs font-normal leading-5 text-muted-foreground shadow-soft group-hover:block">
      {text}
    </span>
  );
}

export function ModeSelector({
  workflowMode,
  onWorkflowModeChange,
  selectedModel,
  onModelChange,
  disabled = false,
}: ModeSelectorProps) {
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  const selectedModelOption = useMemo(
    () => MODEL_OPTIONS.find((option) => option.value === selectedModel),
    [selectedModel],
  );

  useEffect(() => {
    if (!isModelMenuOpen) {
      return;
    }

    const handleOutsideClick = (event: MouseEvent) => {
      if (!modelMenuRef.current?.contains(event.target as Node)) {
        setIsModelMenuOpen(false);
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsModelMenuOpen(false);
      }
    };

    window.addEventListener("mousedown", handleOutsideClick);
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("mousedown", handleOutsideClick);
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isModelMenuOpen]);

  return (
    <div className="relative z-40 flex min-w-0 flex-1 flex-wrap items-center gap-3">
      <div className="flex h-10 items-center rounded-xl border border-border bg-card/80 p-1 shadow-subtle">
        {WORKFLOW_OPTIONS.map((option) => {
          const active = workflowMode === option.value;
          const Icon = option.icon;
          return (
            <button
              key={option.value}
              type="button"
              disabled={disabled}
              onClick={() => onWorkflowModeChange(option.value)}
              title={option.tooltip}
              className={cn(
                "group relative inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-xs font-semibold transition duration-150 ease-out hover:-translate-y-px active:translate-y-0 active:scale-[0.98]",
                "disabled:cursor-not-allowed disabled:opacity-60",
                active
                  ? "bg-primary text-primary-foreground shadow-sm shadow-primary/25"
                  : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {option.label}
              <Tooltip text={option.tooltip} />
            </button>
          );
        })}
      </div>

      <div ref={modelMenuRef} className="relative z-50">
        <button
          type="button"
          disabled={disabled}
          onClick={() => setIsModelMenuOpen((current) => !current)}
          className={cn(
            "inline-flex h-10 items-center gap-2 rounded-xl border border-border bg-card/80 px-3 text-xs font-semibold text-foreground transition duration-150 ease-out hover:-translate-y-px hover:border-primary/40 hover:bg-surface-hover active:translate-y-0 active:scale-[0.98]",
            "disabled:cursor-not-allowed disabled:opacity-60",
          )}
          title="Model used for all workflow modes"
        >
          <span className="font-mono text-[11px] text-muted-foreground">model</span>
          <span>{selectedModelOption?.label ?? selectedModel}</span>
          <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition", isModelMenuOpen ? "rotate-180" : "")} />
        </button>

        {isModelMenuOpen ? (
          <div className="absolute right-0 z-[70] mt-2 w-[min(320px,calc(100vw-2rem))] rounded-xl border border-border bg-card p-2 shadow-soft">
            {MODEL_OPTIONS.map((option) => {
              const selected = option.value === selectedModel;
              return (
                <button
                  key={option.value}
                  type="button"
                  className={cn(
                    "flex w-full items-start gap-3 rounded-xl px-3 py-2 text-left transition duration-150 ease-out hover:-translate-y-px active:translate-y-0 active:scale-[0.99]",
                    selected ? "bg-primary/15 text-foreground" : "text-muted-foreground hover:bg-muted/60",
                  )}
                  onClick={() => {
                    onModelChange(option.value);
                    setIsModelMenuOpen(false);
                  }}
                >
                  <span
                    className={cn(
                      "mt-0.5 flex h-4 w-4 items-center justify-center rounded-full border",
                      selected ? "border-primary bg-primary text-primary-foreground" : "border-border text-transparent",
                    )}
                  >
                    <Check className="h-3 w-3" />
                  </span>
                  <span className="min-w-0">
                    <span className="block text-sm font-semibold">{option.label}</span>
                    <span className="block text-xs text-muted-foreground">{option.detail}</span>
                  </span>
                </button>
              );
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}
