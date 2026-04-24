import { Card } from "@/components/ui/card";

type ChatMessageProps = {
  role: "user" | "assistant";
  content: string;
};

export function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === "user";
  
  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="flex-shrink-0">
          <img
            src="/avatars/ai.png"
            alt="AI Avatar"
            className="h-10 w-10 rounded-full object-cover ring-2 ring-blue-500/20"
          />
        </div>
      )}
      
      <Card className={`max-w-[75%] border-slate-200 shadow-sm ${isUser ? "bg-blue-50" : "bg-white"}`}>
        <div className="px-4 py-3">
          <div className="mb-1 flex items-center gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              {isUser ? "Bạn" : "SmartDocAI"}
            </p>
          </div>
          <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{content}</p>
        </div>
      </Card>
      
      {isUser && (
        <div className="flex-shrink-0">
          <img
            src="/avatars/user.jpg"
            alt="User Avatar"
            className="h-10 w-10 rounded-full object-cover ring-2 ring-slate-200"
          />
        </div>
      )}
    </div>
  );
}
