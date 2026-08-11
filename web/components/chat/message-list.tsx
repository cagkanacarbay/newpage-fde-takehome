"use client";

import { RotateCcw } from "lucide-react";
import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { citationKey, numberCitations } from "@/lib/citations";
import type { Citation } from "@/lib/sse";

import { Markdown } from "./markdown";
import type { PdfTarget, UiMessage } from "./types";

const SCROLL_PIN_THRESHOLD_PX = 96;

function CitationChip({
  index,
  citation,
  active,
  onOpen,
}: {
  index: number;
  citation: Citation;
  active: boolean;
  onOpen: () => void;
}) {
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        onClick={onOpen}
        aria-pressed={active}
        aria-label={`Open citation ${index}: ${citation.snippet}`}
        className={
          active
            ? "rounded-full bg-teal px-2.5 py-1 text-xs font-medium text-on-accent transition-colors"
            : "rounded-full bg-surface px-2.5 py-1 text-xs font-medium text-teal-ink ring-1 ring-teal-mid transition-colors hover:bg-teal-bg"
        }
      >
        [{index}]
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 z-10 mb-2 hidden w-64 rounded-xl bg-ink px-3 py-2 text-xs leading-5 text-surface group-hover:block"
      >
        {citation.snippet}
      </span>
    </span>
  );
}

function SystemDivider({ text }: { text: string }) {
  return (
    <p role="note" className="mx-auto w-fit rounded-full bg-surface-2 px-4 py-1.5 text-center text-xs text-faint">
      {text}
    </p>
  );
}

type MessageListProps = {
  messages: UiMessage[];
  streaming: boolean;
  pdf: PdfTarget | null;
  onOpenCitation: (messageId: string, citation: Citation, chipIndex: number) => void;
  onRetry: (assistantId: string) => void;
};

export function MessageList({ messages, streaming, pdf, onOpenCitation, onRetry }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Pinned-to-bottom autoscroll: follow the stream only while the user is
  // already near the bottom, so reading up-thread is never yanked away.
  useEffect(() => {
    const container = containerRef.current;
    if (container && pinnedRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages, streaming]);

  return (
    <div
      ref={containerRef}
      onScroll={() => {
        const container = containerRef.current;
        if (!container) {
          return;
        }
        pinnedRef.current =
          container.scrollHeight - container.scrollTop - container.clientHeight <
          SCROLL_PIN_THRESHOLD_PX;
      }}
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
      aria-label="Chat messages"
    >
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6 sm:px-6">
        {messages.map((message) => {
          if (message.role === "system") {
            return <SystemDivider key={message.id} text={message.text} />;
          }

          if (message.role === "user") {
            return (
              <article
                key={message.id}
                className="ml-auto max-w-[85%] rounded-3xl bg-teal px-5 py-3 text-on-accent"
              >
                <p className="whitespace-pre-wrap text-sm leading-6">{message.text}</p>
              </article>
            );
          }

          const numbered = numberCitations(message.citations);
          return (
            <article key={message.id} className="max-w-full rounded-3xl bg-surface px-5 py-4">
              {message.text ? (
                <Markdown text={message.text} />
              ) : streaming && !message.error ? (
                <p className="flex items-center gap-1.5 py-1 text-sm text-faint" aria-live="polite">
                  <span className="size-1.5 animate-pulse rounded-full bg-teal" />
                  <span className="size-1.5 animate-pulse rounded-full bg-teal [animation-delay:150ms]" />
                  <span className="size-1.5 animate-pulse rounded-full bg-teal [animation-delay:300ms]" />
                  <span className="sr-only">Thinking</span>
                </p>
              ) : null}

              {message.error ? (
                <div
                  role="alert"
                  className="mt-2 flex items-center justify-between gap-3 rounded-2xl bg-orange-bg px-4 py-3"
                >
                  <p className="text-sm text-orange-ink">{message.error}</p>
                  <Button type="button" size="sm" variant="ghost" onClick={() => onRetry(message.id)}>
                    <RotateCcw className="size-3.5" />
                    Retry
                  </Button>
                </div>
              ) : null}

              {numbered.length > 0 ? (
                <div aria-label="Citations" className="mt-3 flex flex-wrap gap-2">
                  {numbered.map(({ citation, index }) => (
                    <CitationChip
                      key={citationKey(citation)}
                      index={index}
                      citation={citation}
                      active={
                        pdf !== null &&
                        pdf.messageId === message.id &&
                        pdf.chipIndex === index
                      }
                      onOpen={() => onOpenCitation(message.id, citation, index)}
                    />
                  ))}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </div>
  );
}
