import type { ChatEvent } from "@/lib/sse";

import type { UiMessage } from "./types";

export type Retry = {
  message: string;
  messages: UiMessage[];
};

export function applyAssistantEvent(
  message: UiMessage,
  event: ChatEvent,
  nowMs: number,
): UiMessage {
  if (message.role !== "assistant") {
    return message;
  }
  if (event.type === "token") {
    const firstTokenSeconds =
      message.firstTokenSeconds ??
      (message.responseStartedAtMs === undefined
        ? undefined
        : Math.round(((nowMs - message.responseStartedAtMs) / 1_000) * 100) / 100);
    return {
      ...message,
      text: message.text + event.text,
      firstTokenSeconds,
    };
  }
  if (event.type === "verification") {
    if (event.status === "started") {
      return { ...message, verification: "verifying" };
    }
    return {
      ...message,
      text: event.text,
      citations: event.citations,
      verification: event.changed ? "updated" : "verified",
    };
  }
  if (event.type === "citations") {
    return { ...message, citations: [...message.citations, ...event.citations] };
  }
  if (event.type === "error") {
    return {
      ...message,
      error: event.message,
      verification: message.verification === "verifying" ? "failed" : message.verification,
    };
  }
  return message;
}

export function retryFailedTurn(messages: UiMessage[], assistantId: string): Retry | null {
  const assistant = messages.find((message) => message.id === assistantId);
  if (
    assistant?.role !== "assistant" ||
    !assistant.error ||
    !assistant.replyTo ||
    !assistant.retryMessage
  ) {
    return null;
  }

  return {
    message: assistant.retryMessage,
    messages: messages.filter(
      (message) => message.id !== assistant.id && message.id !== assistant.replyTo,
    ),
  };
}

export function withHistoryNotice(
  messages: UiMessage[],
  text: string,
  id: string,
): UiMessage[] {
  const visibleMessages = messages.filter((message) => message.role !== "system");

  return [
    {
      id,
      role: "system",
      text,
      citations: [],
    },
    ...visibleMessages,
  ];
}
