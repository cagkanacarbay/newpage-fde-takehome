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

  const earlierNotice = messages.find((message) => message.role === "system");
  const earlierTurns = Number.parseInt(earlierNotice?.text ?? "", 10) || 0;
  const totalTurns = earlierTurns + turns;
  let remainingMessages = turns * 2;
  const retained = messages.filter((message) => {
    if (message.role === "system") {
      return false;
    }
    if (remainingMessages === 0) {
      return true;
    }
    remainingMessages -= 1;
    return false;
  });

  return [
    {
      id,
      role: "system",
      text: `${totalTurns} earlier ${totalTurns === 1 ? "turn is" : "turns are"} not included in the assistant context.`,
      citations: [],
    },
    ...retained,
  ];
}
