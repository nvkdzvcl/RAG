import {
  Database,
  Ellipsis,
  FileText,
  MessageSquare,
  MessageSquarePlus,
  Pin,
  Settings,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type RecentChat = {
  id: string;
  title: string;
  pinned: boolean;
  timeLabel: string;
};

type SidebarProps = {
  activeSessionId: string | null;
  recentChats: RecentChat[];
  documentsCount: number;
  readyDocumentsCount: number;
  onNewChat: () => void;
  onSelectRecent: (chat: RecentChat) => void;
  onTogglePinChat: (chat: RecentChat) => void;
  onDeleteChat: (chat: RecentChat) => void;
  onOpenDocuments: () => void;
  onOpenSettings: () => void;
  onClearHistory: () => void;
  onClearVectorStore: () => void;
};

export function Sidebar({
  activeSessionId,
  recentChats,
  documentsCount,
  readyDocumentsCount,
  onNewChat,
  onSelectRecent,
  onTogglePinChat,
  onDeleteChat,
  onOpenDocuments,
  onOpenSettings,
  onClearHistory,
  onClearVectorStore,
}: SidebarProps) {
  const [openMenuChatId, setOpenMenuChatId] = useState<string | null>(null);

  useEffect(() => {
    if (!openMenuChatId) {
      return;
    }

    const handleOutsideClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      if (target.closest("[data-session-menu]") || target.closest("[data-session-menu-trigger]")) {
        return;
      }
      setOpenMenuChatId(null);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpenMenuChatId(null);
      }
    };

    window.addEventListener("mousedown", handleOutsideClick);
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("mousedown", handleOutsideClick);
      window.removeEventListener("keydown", handleEscape);
    };
  }, [openMenuChatId]);

  return (
    <div className="flex h-full min-h-0 flex-col border-r border-border bg-background/70 px-3 py-4">
      <div className="mb-4 flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent text-primary-foreground shadow-sm shadow-primary/30">
          <Sparkles className="h-4 w-4" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">Self-RAG Console</p>
          <p className="text-xs text-muted-foreground">Grounded document chat</p>
        </div>
      </div>

      <Button type="button" onClick={onNewChat} className="mb-4 h-10 justify-start gap-2">
        <MessageSquarePlus className="h-4 w-4" />
        New Chat
      </Button>

      <nav className="space-y-1">
        <button
          type="button"
          onClick={onOpenDocuments}
          className="flex h-10 w-full items-center justify-between rounded-xl px-3 text-sm text-muted-foreground transition duration-150 ease-out hover:-translate-y-px hover:bg-muted/60 hover:text-foreground active:translate-y-0 active:scale-[0.98]"
        >
          <span className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Documents
          </span>
          <Badge variant="outline" className="border-border bg-muted/50 px-2 py-0 text-[11px] text-muted-foreground">
            {readyDocumentsCount}/{documentsCount}
          </Badge>
        </button>
        <button
          type="button"
          onClick={onOpenSettings}
          className="flex h-10 w-full items-center gap-2 rounded-xl px-3 text-sm text-muted-foreground transition duration-150 ease-out hover:-translate-y-px hover:bg-muted/60 hover:text-foreground active:translate-y-0 active:scale-[0.98]"
        >
          <Settings className="h-4 w-4" />
          Settings
        </button>
      </nav>

      <div className="mt-5 flex min-h-0 flex-1 flex-col">
        <div className="mb-2 flex items-center justify-between px-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Sessions</p>
          {recentChats.length > 0 ? (
            <button
              type="button"
              onClick={onClearHistory}
              className="rounded-md p-1 text-muted-foreground/70 transition duration-150 ease-out hover:-translate-y-px hover:bg-muted/60 hover:text-destructive active:translate-y-0 active:scale-95"
              title="Clear chat history"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          ) : null}
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
          {recentChats.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border px-3 py-4 text-xs leading-5 text-muted-foreground">
              Start a conversation to build a session history.
            </div>
          ) : (
            recentChats.map((chat) => {
              const active = chat.id === activeSessionId;
              const menuOpen = openMenuChatId === chat.id;
              return (
                <div key={chat.id} className="group/session relative">
                  <button
                    type="button"
                    onClick={() => {
                      setOpenMenuChatId(null);
                      onSelectRecent(chat);
                    }}
                    className={cn(
                      "w-full rounded-xl border px-3 py-2 pr-10 text-left transition duration-150 ease-out hover:-translate-y-px active:translate-y-0 active:scale-[0.99]",
                      active
                        ? "border-primary/50 bg-primary-light shadow-sm shadow-primary/10"
                        : "border-transparent hover:border-border hover:bg-muted/60",
                    )}
                  >
                    <div className="flex items-start gap-2">
                      <MessageSquare className={cn("mt-0.5 h-4 w-4 shrink-0", active ? "text-primary" : "text-muted-foreground/70")} />
                      <div className="min-w-0 flex-1">
                        <div className="flex min-w-0 items-center gap-1.5">
                          {chat.pinned ? <Pin className="h-3 w-3 shrink-0 text-primary" /> : null}
                          <p className={cn("line-clamp-1 text-sm", active ? "text-foreground" : "text-muted-foreground")}>
                            {chat.title}
                          </p>
                        </div>
                        <div className="mt-1 flex items-center justify-between gap-2">
                          <span className="text-[11px] text-muted-foreground">{chat.timeLabel}</span>
                        </div>
                      </div>
                    </div>
                  </button>

                  <button
                    type="button"
                    data-session-menu-trigger
                    onClick={(event) => {
                      event.stopPropagation();
                      setOpenMenuChatId((current) => (current === chat.id ? null : chat.id));
                    }}
                    className={cn(
                      "absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground transition duration-150 ease-out hover:bg-muted hover:text-foreground",
                      "opacity-100 md:opacity-0 md:group-hover/session:opacity-100 focus-visible:opacity-100",
                      menuOpen ? "opacity-100 bg-muted text-foreground" : "",
                    )}
                    title="Session actions"
                  >
                    <Ellipsis className="h-4 w-4" />
                  </button>

                  {menuOpen ? (
                    <div
                      data-session-menu
                      className="absolute right-2 top-10 z-20 w-36 rounded-xl border border-border bg-card p-1 shadow-soft"
                    >
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setOpenMenuChatId(null);
                          onTogglePinChat(chat);
                        }}
                        className="flex h-8 w-full items-center gap-2 rounded-lg px-2 text-xs text-foreground transition hover:bg-muted"
                      >
                        <Pin className="h-3.5 w-3.5" />
                        {chat.pinned ? "Bỏ ghim" : "Ghim"}
                      </button>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setOpenMenuChatId(null);
                          onDeleteChat(chat);
                        }}
                        className="mt-1 flex h-8 w-full items-center gap-2 rounded-lg px-2 text-xs text-destructive transition hover:bg-destructive/10"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        Xóa
                      </button>
                    </div>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-destructive/20 bg-destructive/10 p-3">
        <p className="text-xs font-semibold uppercase text-destructive">Vector store</p>
        <p className="mt-1 text-xs leading-5 text-destructive/80">
          Clears uploaded documents and local retrieval indexes.
        </p>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onClearVectorStore}
          className="mt-3 h-9 w-full justify-start gap-2 border-destructive/30 bg-destructive/10 text-destructive hover:bg-destructive/15"
        >
          <Database className="h-3.5 w-3.5" />
          Clear documents
        </Button>
      </div>
    </div>
  );
}
