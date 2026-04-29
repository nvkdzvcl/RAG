export type ThemeMode = "dark" | "light";

export const THEME_STORAGE_KEY = "self-rag-theme";
export const DEFAULT_THEME: ThemeMode = "dark";

export function isThemeMode(value: unknown): value is ThemeMode {
  return value === "dark" || value === "light";
}

export function readThemePreference(): ThemeMode {
  if (typeof window === "undefined") {
    return DEFAULT_THEME;
  }

  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return isThemeMode(stored) ? stored : DEFAULT_THEME;
}

export function persistThemePreference(theme: ThemeMode): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(THEME_STORAGE_KEY, theme);
}

export function applyThemePreference(theme: ThemeMode): void {
  if (typeof document === "undefined") {
    return;
  }

  const root = document.documentElement;
  root.classList.remove(theme === "dark" ? "light" : "dark");
  root.classList.add(theme);
  root.style.colorScheme = theme;
}
