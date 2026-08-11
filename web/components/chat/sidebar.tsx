"use client";

import { Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { ConversationSummary } from "@/lib/conversations";
import { cn } from "@/lib/utils";

type SidebarProps = {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
};

function ConversationItem({
  conversation,
  active,
  onSelect,
  onDelete,
}: {
  conversation: ConversationSummary;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const [confirming, setConfirming] = useState(false);

  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onSelect}
        aria-current={active ? "true" : undefined}
        className={cn(
          "w-full rounded-xl px-4 py-3 text-left text-sm transition-colors",
          active
            ? "bg-teal-bg font-medium text-teal-ink"
            : "text-body hover:bg-surface",
        )}
      >
        <span className="block truncate pr-6">{conversation.title}</span>
      </button>
      <button
        type="button"
        aria-label={
          confirming
            ? `Confirm delete ${conversation.title}`
            : `Delete ${conversation.title}`
        }
        onClick={() => {
          if (confirming) {
            onDelete();
            return;
          }
          setConfirming(true);
        }}
        onBlur={() => setConfirming(false)}
        className={cn(
          "absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1.5 transition-colors",
          confirming
            ? "bg-orange-bg text-orange-ink"
            : "text-faint opacity-0 hover:text-orange-ink focus-visible:opacity-100 group-hover:opacity-100",
        )}
      >
        <Trash2 className="size-4" />
      </button>
    </li>
  );
}

export function Sidebar({ conversations, activeId, onSelect, onNew, onDelete }: SidebarProps) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-4 p-4">
      <Button type="button" onClick={onNew} className="w-full">
        <Plus className="size-4" />
        New conversation
      </Button>

      <nav aria-label="Conversations" className="min-h-0 flex-1 overflow-y-auto">
        <ul className="flex flex-col gap-1">
          {conversations.map((conversation) => (
            <ConversationItem
              key={conversation.id}
              conversation={conversation}
              active={conversation.id === activeId}
              onSelect={() => onSelect(conversation.id)}
              onDelete={() => onDelete(conversation.id)}
            />
          ))}
        </ul>
      </nav>

      <footer className="px-2 pb-1 text-sm font-semibold tracking-tight text-ink">
        Live Long R&amp;D
        <span className="mt-0.5 block text-xs font-normal text-faint">
          Longevity research assistant
        </span>
      </footer>
    </div>
  );
}
