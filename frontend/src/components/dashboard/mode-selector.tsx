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

const modeTabs: Array<{ value: Mode; label: string; subtitle: string }> = [
  { value: "standard", label: translations.modes.standard, subtitle: translations.modeDescriptions.standard },
  { value: "advanced", label: translations.modes.advanced, subtitle: translations.modeDescriptions.advanced },
  { value: "compare", label: translations.modes.compare, subtitle: translations.modeDescriptions.compare },
];

const MODEL_OPTIONS = ["qwen2.5:3b", "qwen2.5:7b", "qwen3.5:4b", "qwen3.5:9b"] as const;

export function ModeSelector({
  mode,
  onModeChange,
  selectedModel,
  onModelChange,
  disabled = false,
}: ModeSelectorProps) {
  const modelOptions = MODEL_OPTIONS.includes(selectedModel as (typeof MODEL_OPTIONS)[number])
    ? MODEL_OPTIONS
    : [selectedModel, ...MODEL_OPTIONS];

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

      <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_220px] md:items-end">
        <p className="text-xs text-slate-600">
          {translations.model.active}: <span className="font-semibold text-slate-900">{selectedModel}</span>
        </p>

        <label className="min-w-0">
          <span className="mb-1 block text-xs font-medium text-slate-600">{translations.model.title}</span>
          <select
            value={selectedModel}
            onChange={(event) => onModelChange(event.target.value)}
            disabled={disabled}
            className="w-full rounded-lg border border-slate-200 bg-white px-2 py-2 text-sm text-slate-800 outline-none transition focus:border-primary/40 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {modelOptions.map((modelOption) => (
              <option key={modelOption} value={modelOption}>
                {modelOption}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="mt-2 text-xs text-slate-500">{translations.model.sharedAcrossModes}</p>
    </div>
  );
}
