"use client";

import { CircleCheck, LoaderCircle, RotateCcw, ShieldAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { fallbackCitations } from "@/lib/citations";
import type { Citation } from "@/lib/sse";

import { Markdown } from "./markdown";
import type { UiMessage } from "./types";

const SCROLL_PIN_THRESHOLD_PX = 96;

function FirstTokenTiming({ message }: { message: UiMessage }) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (message.responseStartedAtMs === undefined || message.firstTokenSeconds !== undefined) {
      return;
    }
    const update = () => {
      setElapsedSeconds(
        Math.max(0, (performance.now() - message.responseStartedAtMs!) / 1_000),
      );
    };
    update();
    const interval = window.setInterval(update, 100);
    return () => window.clearInterval(interval);
  }, [message.firstTokenSeconds, message.responseStartedAtMs]);

  if (message.responseStartedAtMs === undefined) {
    return null;
  }
  const text =
    message.firstTokenSeconds === undefined
      ? `Waiting for first token · ${elapsedSeconds.toFixed(1)} s`
      : `First token in ${message.firstTokenSeconds.toFixed(2)} s`;
  return (
    <p className="mt-3 text-xs font-medium tabular-nums text-faint" aria-live="polite">
      {text}
    </p>
  );
}

function VerificationStatus({ state }: { state: UiMessage["verification"] }) {
  if (state === undefined) {
    return null;
  }
  const content = {
    verifying: {
      icon: <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />,
      text: "Verifying claims",
      className: "bg-teal-bg text-teal-ink",
    },
    verified: {
      icon: <CircleCheck className="size-3.5" aria-hidden="true" />,
      text: "Claims verified",
      className: "bg-teal-bg text-teal-ink",
    },
    updated: {
      icon: <CircleCheck className="size-3.5" aria-hidden="true" />,
      text: "Claims verified · Answer updated",
      className: "bg-yellow-bg text-yellow-ink",
    },
    failed: {
      icon: <ShieldAlert className="size-3.5" aria-hidden="true" />,
      text: "Claim verification failed",
      className: "bg-orange-bg text-orange-ink",
    },
  }[state];
  return (
    <p
      role="status"
      className={`mt-3 flex w-fit items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${content.className}`}
    >
      {content.icon}
      {content.text}
    </p>
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
  onOpenCitation: (messageId: string, citation: Citation, chipIndex: number) => void;
  onRetry: (assistantId: string) => void;
};

export function MessageList({ messages, streaming, onOpenCitation, onRetry }: MessageListProps) {
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

          const legacyCitations = fallbackCitations(message.text, message.citations);
          return (
            <article
              key={message.id}
              className={`max-w-full rounded-3xl bg-surface px-5 py-4 ${
                message.verification === "updated" ? "verification-update" : ""
              }`}
            >
              {message.text ? (
                <Markdown
                  text={message.text}
                  citations={message.citations}
                  onOpenCitation={(citation, index) =>
                    onOpenCitation(message.id, citation, index)
                  }
                />
              ) : streaming && !message.error ? (
                <p className="flex items-center gap-1.5 py-1 text-sm text-faint" aria-live="polite">
                  <span className="size-1.5 animate-pulse rounded-full bg-teal" />
                  <span className="size-1.5 animate-pulse rounded-full bg-teal [animation-delay:150ms]" />
                  <span className="size-1.5 animate-pulse rounded-full bg-teal [animation-delay:300ms]" />
                  <span className="sr-only">Thinking</span>
                </p>
              ) : null}

              <FirstTokenTiming message={message} />
              <VerificationStatus state={message.verification} />

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

              {legacyCitations.length > 0 ? (
                <div className="mt-3 flex flex-wrap items-center gap-2" aria-label="Sources">
                  <span className="text-xs font-medium text-faint">Sources</span>
                  {legacyCitations.map(({ citation, index }) => (
                    <button
                      key={`${citation.document_id}-${citation.page}-${index}`}
                      type="button"
                      aria-label={`Open citation ${index}: ${citation.snippet}`}
                      title={citation.snippet}
                      onClick={() => onOpenCitation(message.id, citation, index)}
                      className="inline-flex rounded-full bg-teal-bg px-2 py-1 text-xs font-semibold text-teal-ink transition-colors hover:bg-teal hover:text-on-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal"
                    >
                      [{index}]
                    </button>
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
