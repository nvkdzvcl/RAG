import type { Mode } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type ModeSelectorProps = {
  mode: Mode;
  onModeChange: (value: Mode) => void;
  disabled?: boolean;
};

const MODE_OPTIONS: Array<{ value: Mode; label: string; note: string }> = [
  { value: "standard", label: "Standard", note: "Baseline RAG" },
  { value: "advanced", label: "Advanced", note: "Self-RAG loop" },
  { value: "compare", label: "Compare", note: "Run both branches" },
];

export function ModeSelector({ mode, onModeChange, disabled = false }: ModeSelectorProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>Mode Selector</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-2">
        {MODE_OPTIONS.map((option) => {
          const isActive = option.value === mode;
          return (
            <Button
              key={option.value}
              type="button"
              variant={isActive ? "default" : "outline"}
              onClick={() => onModeChange(option.value)}
              className="gap-2"
              disabled={disabled}
            >
              <span>{option.label}</span>
              <Badge variant={isActive ? "outline" : "muted"}>{option.note}</Badge>
            </Button>
          );
        })}
      </CardContent>
    </Card>
  );
}
