import { readSseStream, type ChatEvent } from "./sse";

export type StreamChatOptions = {
  conversationId?: string;
  baseUrl?: string;
};

export async function streamChat(
  message: string,
  onEvent: (event: ChatEvent) => void,
  options: StreamChatOptions = {},
): Promise<void> {
  const baseUrl =
    options.baseUrl ?? process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";

  try {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, conversation_id: options.conversationId }),
    });

    if (!response.ok) {
      throw new Error(`Chat request failed with status ${response.status}.`);
    }
    if (!response.body) {
      throw new Error("Chat response did not include a stream.");
    }

    await readSseStream(response.body, onEvent);
  } catch (error) {
    onEvent({
      type: "error",
      message: error instanceof Error ? error.message : "The chat request failed.",
    });
  }
}
