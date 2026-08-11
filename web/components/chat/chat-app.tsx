"use client";

import { PanelLeft, X } from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import {
  createConversationClient,
  type ConversationClient,
  type ConversationSummary,
} from "@/lib/conversations";
import { DraftStore } from "@/lib/drafts";
import type { Citation } from "@/lib/sse";

import { retryFailedTurn, withHistoryNotice } from "./chat-turns";
import { Composer } from "./composer";
import { EmptyState } from "./empty-state";
import { MessageList } from "./message-list";
import { Sidebar } from "./sidebar";
import type { PdfTarget, UiMessage } from "./types";

const PdfPanel = dynamic(() => import("@/components/pdf/pdf-panel"), { ssr: false });

const NEW_CONVERSATION_KEY = "__new__";
const DESKTOP_MEDIA_QUERY = "(min-width: 768px)";

export function ChatApp() {
  const clientRef = useRef<ConversationClient | null>(null);
  if (clientRef.current === null) {
    clientRef.current = createConversationClient();
  }
  const client = clientRef.current;

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [composer, setComposer] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pdf, setPdf] = useState<PdfTarget | null>(null);
  const [desktopPdfDocked, setDesktopPdfDocked] = useState(false);

  const draftsRef = useRef(new DraftStore());

  const refreshConversations = useCallback(async () => {
    setConversations(await client.listConversations());
  }, [client]);

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  useEffect(() => {
    const media = window.matchMedia(DESKTOP_MEDIA_QUERY);
    const update = () => setDesktopPdfDocked(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  const activeTitle = activeId
    ? (conversations.find((item) => item.id === activeId)?.title ?? "Live Long R&D")
    : "Live Long R&D";

  function draftKeyFor(id: string | null): string {
    return id ?? NEW_CONVERSATION_KEY;
  }

  function switchConversation(id: string | null) {
    if (streaming) {
      return;
    }
    draftsRef.current.set(draftKeyFor(activeId), composer);
    setActiveId(id);
    setComposer(draftsRef.current.get(draftKeyFor(id)));
    setPdf(null);
    setSidebarOpen(false);
    if (id === null) {
      setMessages([]);
      return;
    }
    void client.getConversation(id).then((conversation) => {
      setMessages(
        conversation.messages.map((message, index) => ({
          id: `${conversation.id}-${index}`,
          role: message.role,
          text: message.text,
          citations: message.citations,
        })),
      );
    });
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || streaming) {
      return;
    }

    const assistantId = crypto.randomUUID();
    const userId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      { id: userId, role: "user", text: message, citations: [] },
      {
        id: assistantId,
        role: "assistant",
        text: "",
        citations: [],
        replyTo: userId,
        retryMessage: message,
      },
    ]);
    setComposer("");
    draftsRef.current.set(draftKeyFor(activeId), "");
    setStreaming(true);

    let conversationId = activeId;
    await client.sendMessage({ conversationId, message }, (event) => {
      if (event.type === "conversation") {
        conversationId = event.id;
        setActiveId(event.id);
        setConversations((current) => {
          const rest = current.filter((item) => item.id !== event.id);
          return [
            { id: event.id, title: event.title, updated_at: new Date().toISOString() },
            ...rest,
          ];
        });
        return;
      }

      setMessages((current) =>
        current.map((item) => {
          if (item.id !== assistantId) {
            return item;
          }
          if (event.type === "token") {
            return { ...item, text: item.text + event.text };
          }
          if (event.type === "citations") {
            return { ...item, citations: [...item.citations, ...event.citations] };
          }
          if (event.type === "error") {
            return { ...item, error: event.message };
          }
          return item;
        }),
      );

      if (event.type === "history_notice") {
        setMessages((current) =>
          withHistoryNotice(current, event.text, crypto.randomUUID()),
        );
      }
    });

    setStreaming(false);
    void refreshConversations();
  }

  function retry(assistantId: string) {
    if (streaming) {
      return;
    }
    const retry = retryFailedTurn(messages, assistantId);
    if (!retry) {
      return;
    }
    setMessages(retry.messages);
    void send(retry.message);
  }

  function openCitation(messageId: string, citation: Citation, chipIndex: number) {
    setPdf({ documentId: citation.document_id, citation, chipIndex, messageId });
  }

  async function deleteConversation(id: string) {
    await client.deleteConversation(id);
    if (id === activeId) {
      setActiveId(null);
      setMessages([]);
      setComposer(draftsRef.current.get(NEW_CONVERSATION_KEY));
      setPdf(null);
    }
    await refreshConversations();
  }

  const sidebar = (
    <Sidebar
      conversations={conversations}
      activeId={activeId}
      onSelect={(id) => switchConversation(id)}
      onNew={() => switchConversation(null)}
      onDelete={(id) => void deleteConversation(id)}
    />
  );

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      <aside className="hidden w-[270px] shrink-0 bg-surface-2 md:block">{sidebar}</aside>

      <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <SheetContent side="left" className="bg-surface-2 p-0">
          {sidebar}
        </SheetContent>
      </Sheet>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 px-3 py-2 md:hidden">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Open conversations"
            onClick={() => setSidebarOpen(true)}
          >
            <PanelLeft className="size-5" />
          </Button>
          <h1 className="min-w-0 flex-1 truncate text-sm font-semibold text-ink">
            {activeTitle}
          </h1>
          {pdf ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Close document"
              onClick={() => setPdf(null)}
            >
              <X className="size-5" />
            </Button>
          ) : null}
        </header>

        {messages.length === 0 ? (
          <EmptyState onAsk={(question) => void send(question)} />
        ) : (
          <MessageList
            messages={messages}
            streaming={streaming}
            pdf={pdf}
            onOpenCitation={openCitation}
            onRetry={retry}
          />
        )}

        <Composer
          value={composer}
          disabled={streaming}
          onChange={setComposer}
          onSend={() => void send(composer)}
        />
      </main>

      {pdf && desktopPdfDocked ? (
        <section
          aria-label="Source document"
          className="hidden min-w-0 md:block md:w-[45%] md:shrink-0"
        >
          <PdfPanel
            key={`${pdf.documentId}-${pdf.chipIndex}-${pdf.messageId}`}
            documentId={pdf.documentId}
            citation={pdf.citation}
            onClose={() => setPdf(null)}
          />
        </section>
      ) : null}

      <Sheet
        open={pdf !== null && !desktopPdfDocked}
        onOpenChange={(open) => !open && setPdf(null)}
      >
        <SheetContent side="full" hideClose className="md:hidden">
          {pdf && !desktopPdfDocked ? (
            <PdfPanel
              key={`${pdf.documentId}-${pdf.chipIndex}-${pdf.messageId}-mobile`}
              documentId={pdf.documentId}
              citation={pdf.citation}
              onClose={() => setPdf(null)}
            />
          ) : null}
        </SheetContent>
      </Sheet>
    </div>
  );
}
