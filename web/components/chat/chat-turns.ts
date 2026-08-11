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

export function withDroppedTurnsNotice(
  messages: UiMessage[],
  turns: number,
  id: string,
): UiMessage[] {
  if (turns < 1) {
    return messages;
  }

  let remainingMessages = turns * 2;
  const retained = messages.filter((message) => {
    if (message.role === "system" || remainingMessages === 0) {
      return true;
    }
    remainingMessages -= 1;
    return false;
  });

  return [
    {
      id,
      role: "system",
      text: `${turns} earlier ${turns === 1 ? "turn is" : "turns are"} not included in the assistant context.`,
      citations: [],
    },
    ...retained,
  ];
}
