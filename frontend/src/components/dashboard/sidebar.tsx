import {
  BarChart3,
  FileSearch2,
  FileText,
  MessageSquarePlus,
  Settings,
  Sparkles,
} from "lucide-react";

import type { Mode } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

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
};

const navItems = [
  { key: "dashboard", label: "Dashboard", icon: BarChart3 },
  { key: "evaluation", label: "Evaluation", icon: FileSearch2 },
  { key: "documents", label: "Documents", icon: FileText },
  { key: "settings", label: "Settings", icon: Settings },
];

export function Sidebar({ mode, onNewChat, recentChats, onSelectRecent }: SidebarProps) {
  return (
    <div className="flex h-full flex-col gap-5 p-4 md:p-5">
      <div className="rounded-2xl bg-slate-900/80 p-4 ring-1 ring-slate-700">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 text-white shadow-md">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <p className="text-lg font-semibold">Self-RAG</p>
            <p className="text-xs text-slate-400">Three-Mode RAG System</p>
          </div>
        </div>
      </div>

      <Button type="button" onClick={onNewChat} className="h-10 justify-start gap-2 bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:opacity-95">
        <MessageSquarePlus className="h-4 w-4" />
        New Chat
      </Button>

      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Navigation</p>
        <div className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                type="button"
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
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Recent Chats</p>
        <div className="space-y-1">
          {recentChats.map((chat) => (
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
          ))}
        </div>
      </div>

      <div className="mt-auto rounded-2xl border border-blue-500/30 bg-slate-900/70 p-3">
        <p className="text-xs uppercase tracking-wide text-slate-400">Current Mode</p>
        <div className="mt-2 flex items-center justify-between">
          <p className="text-sm font-semibold capitalize text-slate-100">{mode}</p>
          <Badge variant="outline" className="border-blue-400/40 bg-blue-500/10 text-blue-200">
            Active
          </Badge>
        </div>
      </div>
    </div>
  );
}
