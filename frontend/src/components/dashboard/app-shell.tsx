import type { ReactNode } from "react";

type AppShellProps = {
  sidebar: ReactNode;
  main: ReactNode;
  inspect: ReactNode;
  rightPanelOpen?: boolean;
};

export function AppShell({ sidebar, main, inspect, rightPanelOpen = true }: AppShellProps) {
  return (
    <div className="rag-app-bg min-h-screen text-foreground lg:h-screen lg:overflow-hidden">
      <div
        className={
          rightPanelOpen
            ? "grid min-h-screen w-full grid-cols-1 transition-[grid-template-columns] duration-[180ms] ease-out lg:h-screen lg:grid-cols-[264px_minmax(0,1fr)_360px]"
            : "grid min-h-screen w-full grid-cols-1 transition-[grid-template-columns] duration-[180ms] ease-out lg:h-screen lg:grid-cols-[264px_minmax(0,1fr)_52px]"
        }
      >
        <aside className="min-w-0 border-b border-border bg-background/85 backdrop-blur-xl lg:h-screen lg:overflow-hidden lg:border-b-0">
          {sidebar}
        </aside>

        <main className="min-w-0 bg-background/75 backdrop-blur-xl lg:h-screen lg:overflow-hidden">
          {main}
        </main>

        <aside className="min-w-0 border-t border-border bg-background/85 backdrop-blur-xl lg:h-screen lg:overflow-hidden lg:border-t-0">
          {inspect}
        </aside>
      </div>
    </div>
  );
}
