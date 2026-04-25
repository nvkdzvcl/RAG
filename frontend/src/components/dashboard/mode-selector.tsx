import { Check, ChevronDown } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { Mode } from "@/types/chat";
import { cn } from "@/lib/utils";
import { translations } from "@/lib/translations";

type ModeSelectorProps = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  selectedModel: string;
  onModelChange: (model: string) => void;
  disabled?: boolean;
};

type ModelOption = {
  value: "qwen2.5:3b" | "qwen2.5:7b" | "qwen3.5:4b" | "qwen3.5:9b";
  label: string;
  speed: string;
  description: string;
};

const modeTabs: Array<{ value: Mode; label: string; subtitle: string }> = [
  { value: "standard", label: translations.modes.standard, subtitle: translations.modeDescriptions.standard },
  { value: "advanced", label: translations.modes.advanced, subtitle: translations.modeDescriptions.advanced },
  { value: "compare", label: translations.modes.compare, subtitle: translations.modeDescriptions.compare },
];

const MODEL_OPTIONS: ModelOption[] = [
  {
    value: "qwen2.5:3b",
    label: "Qwen 2.5 3B",
    speed: "Nhanh",
    description: "Phản hồi nhanh, phù hợp trao đổi thường ngày.",
  },
  {
    value: "qwen2.5:7b",
    label: "Qwen 2.5 7B",
    speed: "Cân bằng",
    description: "Cân bằng giữa tốc độ và chất lượng lập luận.",
  },
  {
    value: "qwen3.5:4b",
    label: "Qwen 3.5 4B",
    speed: "Tư duy nhẹ",
    description: "Tốt cho phân tích ngắn và tác vụ có cấu trúc.",
  },
  {
    value: "qwen3.5:9b",
    label: "Qwen 3.5 9B",
    speed: "Nặng / Chất lượng cao",
    description: "Ưu tiên chất lượng, có thể chậm hơn trên máy yếu.",
  },
];

export function ModeSelector({
  mode,
  onModeChange,
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
    <div className="rounded-2xl border border-slate-200 bg-white p-2 shadow-sm">
      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        {modeTabs.map((tab) => {
          const active = tab.value === mode;
          return (
            <button
              key={tab.value}
              type="button"
              disabled={disabled}
              onClick={() => onModeChange(tab.value)}
              className={cn(
                "rounded-xl border px-4 py-3 text-left transition",
                "disabled:cursor-not-allowed disabled:opacity-60",
                active
                  ? "border-primary/30 bg-blue-50 text-slate-900 shadow-sm"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50",
              )}
            >
              <p className="text-sm font-semibold">{tab.label}</p>
              <p className="text-xs text-slate-500">{tab.subtitle}</p>
            </button>
          );
        })}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-slate-600">{translations.model.sharedAcrossModes}</p>

        <div ref={modelMenuRef} className="relative">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setIsModelMenuOpen((current) => !current)}
            className={cn(
              "inline-flex h-9 items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 text-sm font-medium text-slate-800 transition hover:border-slate-300 hover:bg-slate-100",
              "disabled:cursor-not-allowed disabled:opacity-60",
            )}
          >
            <span>{selectedModelOption?.label ?? selectedModel}</span>
            <ChevronDown className={cn("h-4 w-4 transition", isModelMenuOpen ? "rotate-180" : "")} />
          </button>

          {isModelMenuOpen ? (
            <div className="absolute right-0 z-30 mt-2 w-[min(320px,calc(100vw-2rem))] rounded-2xl border border-slate-200 bg-white p-2 shadow-xl">
              <p className="px-2 pb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                {translations.model.title}
              </p>
              <div className="space-y-1">
                {MODEL_OPTIONS.map((option) => {
                  const selected = option.value === selectedModel;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      className={cn(
                        "flex w-full items-start gap-3 rounded-xl border px-3 py-2 text-left transition",
                        selected
                          ? "border-primary/30 bg-blue-50"
                          : "border-transparent hover:border-slate-200 hover:bg-slate-50",
                      )}
                      onClick={() => {
                        onModelChange(option.value);
                        setIsModelMenuOpen(false);
                      }}
                    >
                      <span
                        className={cn(
                          "mt-0.5 flex h-4 w-4 items-center justify-center rounded-full border",
                          selected ? "border-primary bg-primary text-white" : "border-slate-300 text-transparent",
                        )}
                      >
                        <Check className="h-3 w-3" />
                      </span>

                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-slate-800">{option.label}</span>
                          <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-700">
                            {option.speed}
                          </span>
                        </span>
                        <span className="mt-0.5 block text-xs text-slate-500">{option.description}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
