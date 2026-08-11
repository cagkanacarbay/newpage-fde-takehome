import type { UiMessage } from "./types";

export type Retry = {
  message: string;
  messages: UiMessage[];
};

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
