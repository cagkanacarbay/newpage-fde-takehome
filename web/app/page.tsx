"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { streamChat } from "@/lib/chat";
import type { Citation } from "@/lib/sse";

type Message = {
  id: string;
  role: "assistant" | "user";
  text: string;
  citations: Citation[];
  error?: string;
};

const initialMessage: Message = {
  id: "welcome",
  role: "assistant",
  text: "Ask a question about the longevity research corpus.",
  citations: [],
};

function citationLabel(citation: Citation) {
  return `${citation.document_id.split("-").slice(0, 3).join("-")} p.${citation.page}`;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([initialMessage]);
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const endOfMessages = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endOfMessages.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = question.trim();
    if (!message || isStreaming) {
      return;
    }

    const id = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      { id: `${id}-user`, role: "user", text: message, citations: [] },
      { id, role: "assistant", text: "", citations: [] },
    ]);
    setQuestion("");
    setIsStreaming(true);

    await streamChat(message, (chatEvent) => {
      if (chatEvent.type === "done") {
        setIsStreaming(false);
        return;
      }

      setMessages((current) =>
        current.map((item) => {
          if (item.id !== id) {
            return item;
          }
          if (chatEvent.type === "token") {
            return { ...item, text: item.text + chatEvent.text };
          }
          if (chatEvent.type === "citations") {
            return { ...item, citations: chatEvent.citations };
          }
          return { ...item, error: chatEvent.message };
        }),
      );
    });

    setIsStreaming(false);
  }

  return (
    <main className="mx-auto grid min-h-dvh max-w-3xl grid-rows-[auto_1fr_auto] px-4 sm:px-6">
      <header className="border-b border-zinc-200 py-5">
        <h1 className="text-lg font-semibold tracking-tight">Live Long R&amp;D</h1>
        <p className="mt-1 text-sm text-zinc-600">Research assistant</p>
      </header>

      <ScrollArea className="py-6">
        <section aria-label="Chat messages" className="space-y-5">
          {messages.map((message) => (
            <article
              key={message.id}
              className={
                message.role === "user"
                  ? "ml-auto max-w-[85%] rounded-2xl bg-zinc-950 px-4 py-3 text-white"
                  : "max-w-[90%] rounded-2xl border border-zinc-200 bg-white px-4 py-3 text-zinc-900"
              }
            >
              <p className="whitespace-pre-wrap text-sm leading-6">
                {message.text || (isStreaming ? "Thinking…" : "")}
              </p>
              {message.error ? (
                <p className="mt-3 text-sm text-red-700" role="alert">
                  {message.error}
                </p>
              ) : null}
              {message.citations.length > 0 ? (
                <div aria-label="Citations" className="mt-3 flex flex-wrap gap-2">
                  {message.citations.map((citation) => (
                    <span
                      key={`${citation.document_id}-${citation.page}-${citation.bbox.l}`}
                      className="rounded-full border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-xs text-zinc-700"
                      title={citation.snippet}
                    >
                      {citationLabel(citation)}
                    </span>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
          <div ref={endOfMessages} />
        </section>
      </ScrollArea>

      <form onSubmit={handleSubmit} className="border-t border-zinc-200 py-4">
        <label className="sr-only" htmlFor="question">
          Ask a research question
        </label>
        <div className="flex gap-2">
          <Input
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about longevity research"
            disabled={isStreaming}
          />
          <Button type="submit" disabled={isStreaming || !question.trim()}>
            Send
          </Button>
        </div>
      </form>
    </main>
  );
}
