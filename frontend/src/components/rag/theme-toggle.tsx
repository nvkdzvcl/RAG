import { Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { ThemeMode } from "@/lib/theme";

type ThemeToggleProps = {
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
};

export function ThemeToggle({ theme, onThemeChange }: ThemeToggleProps) {
  const isDark = theme === "dark";
  const nextTheme: ThemeMode = isDark ? "light" : "dark";

  return (
    <Button
      type="button"
      variant="outline"
      size="icon"
      onClick={() => onThemeChange(nextTheme)}
      className="h-10 w-10 bg-card/80"
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
    >
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}
