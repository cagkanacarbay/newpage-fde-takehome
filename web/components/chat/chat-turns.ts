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
  const visibleMessages = messages.filter((message) => message.role !== "system");
  const droppedMessageIds = new Set<string>();
  let droppedTurns = 0;

  for (let index = 0; index < visibleMessages.length - 1 && droppedTurns < turns; index += 1) {
    const user = visibleMessages[index];
    const assistant = visibleMessages[index + 1];
    if (user.role !== "user" || assistant.role !== "assistant" || assistant.error) {
      continue;
    }
    droppedMessageIds.add(user.id);
    droppedMessageIds.add(assistant.id);
    droppedTurns += 1;
    index += 1;
  }

  const retained = visibleMessages.filter((message) => !droppedMessageIds.has(message.id));

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
