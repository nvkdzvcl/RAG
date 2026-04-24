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
  return (
    <div className="flex h-full flex-col gap-5 p-4 md:p-5">
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

      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">{translations.sidebar.recentChats}</p>
        <div className="space-y-1">
          {recentChats.length === 0 ? (
            <p className="px-3 py-2 text-xs text-slate-400">{translations.sidebar.noRecentChats}</p>
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
        {recentChats.length > 0 && onClearHistory && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onClearHistory}
            className="w-full justify-start gap-2 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white"
          >
            <Trash2 className="h-3 w-3" />
            Xóa lịch sử
          </Button>
        )}
      </div>

      {onClearVectorStore && (
        <div className="space-y-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onClearVectorStore}
            className="w-full justify-start gap-2 border-rose-700/50 text-rose-300 hover:bg-rose-900/20 hover:text-rose-200"
          >
            <Database className="h-3 w-3" />
            Xóa tài liệu đã tải
          </Button>
        </div>
      )}

      <div className="mt-auto rounded-2xl border border-blue-500/30 bg-slate-900/70 p-3">
        <p className="text-xs uppercase tracking-wide text-slate-400">Chế độ hiện tại</p>
        <div className="mt-2 flex items-center justify-between">
          <p className="text-sm font-semibold capitalize text-slate-100">{translations.modes[mode]}</p>
          <Badge variant="outline" className="border-blue-400/40 bg-blue-500/10 text-blue-200">
            Đang dùng
          </Badge>
        </div>
      </div>
    </div>
  );
}
