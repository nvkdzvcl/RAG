import type { Mode } from "@/types/chat";
import { cn } from "@/lib/utils";

type ModeSelectorProps = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  disabled?: boolean;
};

const modeTabs: Array<{ value: Mode; label: string; subtitle: string }> = [
  { value: "standard", label: "Standard", subtitle: "Fast baseline" },
  { value: "advanced", label: "Advanced", subtitle: "Self-RAG loop" },
  { value: "compare", label: "Compare", subtitle: "Side-by-side" },
];

export function ModeSelector({ mode, onModeChange, disabled = false }: ModeSelectorProps) {
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
                  ? "border-blue-500/30 bg-gradient-to-r from-blue-50 to-violet-50 text-slate-900 shadow-sm"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50",
              )}
            >
              <p className="text-sm font-semibold">{tab.label}</p>
              <p className="text-xs text-slate-500">{tab.subtitle}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
