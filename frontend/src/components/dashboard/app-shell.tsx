import type { ReactNode } from "react";

type AppShellProps = {
  sidebar: ReactNode;
  main: ReactNode;
  inspect: ReactNode;
};

export function AppShell({ sidebar, main, inspect }: AppShellProps) {
  return (
    <div className="min-h-screen bg-[#eef2ff] text-slate-900 lg:h-screen lg:overflow-hidden">
      <div className="grid min-h-screen w-full grid-cols-1 lg:h-screen lg:grid-cols-[280px_minmax(0,1fr)_360px]">
        <aside className="border-b border-slate-800 bg-slate-950 text-slate-100 lg:h-screen lg:overflow-hidden lg:border-b-0 lg:border-r lg:border-r-slate-800">
          {sidebar}
        </aside>

        <main className="min-w-0 bg-slate-50 px-4 py-4 md:px-6 md:py-6 lg:h-screen lg:overflow-y-auto">
          {main}
        </main>

        <aside className="min-w-0 border-t border-slate-200 bg-slate-100/70 px-4 py-4 md:px-5 md:py-6 lg:h-screen lg:overflow-y-auto lg:border-l lg:border-t-0 lg:border-slate-200">
          {inspect}
        </aside>
      </div>
    </div>
  );
}
