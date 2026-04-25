import {
  FileText,
  MessageSquarePlus,
  Settings,
  Trash2,
  Database,
} from "lucide-react";

import type { Mode } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { translations } from "@/lib/translations";

export type RecentChat = {
  id: string;
  title: string;
  mode: Mode;
  timeLabel: string;
};

type SidebarProps = {
  mode: Mode;
  onNewChat: () => void;
  recentChats: RecentChat[];
  onSelectRecent: (chat: RecentChat) => void;
  onClearHistory?: () => void;
  onClearVectorStore?: () => void;
  onOpenSettings?: () => void;
};

const navItems = [
  { key: "documents", label: "Tài liệu", icon: FileText },
  { key: "settings", label: "Cài đặt", icon: Settings },
];

export function Sidebar({ mode, onNewChat, recentChats, onSelectRecent, onClearHistory, onClearVectorStore, onOpenSettings }: SidebarProps) {
  const canClearHistory = recentChats.length > 0;

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 p-4 md:p-5">
      <div className="shrink-0 space-y-4">
        <div className="rounded-2xl bg-slate-900/80 p-4 ring-1 ring-slate-700">
          <div className="flex items-center gap-2">
            <div>
              <p className="text-lg font-semibold">{translations.appName}</p>
              <p className="text-xs text-slate-400">Hệ thống RAG ba chế độ</p>
            </div>
          </div>
        </div>

        <Button type="button" onClick={onNewChat} className="h-10 justify-start gap-2 bg-primary hover:bg-primary/90 text-white">
          <MessageSquarePlus className="h-4 w-4" />
          {translations.sidebar.newChat}
        </Button>

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Điều hướng</p>
          <div className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isSettings = item.key === "settings";
              return (
                <button
                  key={item.key}
                  type="button"
                  onClick={isSettings ? onOpenSettings : undefined}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-slate-300 transition hover:bg-slate-800 hover:text-white"
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-slate-800/90 bg-slate-900/45 p-3">
        <p className="shrink-0 text-xs font-semibold uppercase tracking-wide text-slate-400">
          {translations.sidebar.recentChats}
        </p>
        <div className="mt-2 min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
          {recentChats.length === 0 ? (
            <p className="px-2 py-2 text-xs text-slate-400">{translations.sidebar.noRecentChats}</p>
          ) : (
            recentChats.map((chat) => (
              <button
                key={chat.id}
                type="button"
                onClick={() => onSelectRecent(chat)}
                className="w-full rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-left transition hover:border-slate-700 hover:bg-slate-900"
              >
                <p className="line-clamp-1 text-sm text-slate-100">{chat.title}</p>
                <div className="mt-1 flex items-center justify-between text-[11px] text-slate-400">
                  <span>{chat.timeLabel}</span>
                  <span className="capitalize">{chat.mode}</span>
                </div>
              </button>
            ))
          )}
        </div>
        {onClearHistory && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onClearHistory}
            disabled={!canClearHistory}
            className="mt-3 w-full shrink-0 justify-start gap-2 border border-[rgba(148,163,184,0.25)] bg-[rgba(15,23,42,0.85)] text-[#cbd5e1] transition-colors hover:bg-[rgba(30,41,59,0.95)] hover:border-[rgba(148,163,184,0.38)] hover:text-[#e2e8f0] disabled:pointer-events-auto disabled:cursor-not-allowed disabled:border-[rgba(71,85,105,0.42)] disabled:bg-[rgba(15,23,42,0.55)] disabled:text-[rgba(148,163,184,0.72)] disabled:opacity-[0.45]"
          >
            <Trash2 className="h-3 w-3" />
            Xóa lịch sử
          </Button>
        )}
      </div>

      <div className="shrink-0 space-y-2">
        {onClearVectorStore && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onClearVectorStore}
            className="w-full justify-start gap-2 border border-[rgba(248,113,113,0.45)] bg-[rgba(127,29,29,0.18)] text-[#fecaca] transition-colors hover:bg-[rgba(127,29,29,0.32)] hover:border-[rgba(248,113,113,0.65)] hover:text-[#fee2e2] disabled:pointer-events-auto disabled:cursor-not-allowed disabled:border-[rgba(120,53,15,0.35)] disabled:bg-[rgba(127,29,29,0.12)] disabled:text-[rgba(252,165,165,0.68)] disabled:opacity-[0.45]"
          >
            <Database className="h-3 w-3" />
            Xóa tài liệu đã tải
          </Button>
        )}

        <div className="rounded-2xl border border-blue-500/30 bg-slate-900/70 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-400">Chế độ hiện tại</p>
          <div className="mt-2 flex items-center justify-between">
            <p className="text-sm font-semibold capitalize text-slate-100">{translations.modes[mode]}</p>
            <Badge variant="outline" className="border-blue-400/40 bg-blue-500/10 text-blue-200">
              Đang dùng
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}
