import { streamChat } from "./chat";
import type { ChatEvent, Citation } from "./sse";

export type ConversationSummary = {
  id: string;
  title: string;
  updated_at: string;
};

export type StoredMessage = {
  role: "user" | "assistant" | "system";
  text: string;
  citations: Citation[];
  created_at: string;
};

export type Conversation = {
  id: string;
  title: string;
  messages: StoredMessage[];
};

export type SendMessageInput = {
  conversationId: string | null;
  message: string;
};

/** The conversation-persistence interface used by the chat UI. */
export interface ConversationClient {
  listConversations(): Promise<ConversationSummary[]>;
  createConversation(): Promise<Pick<ConversationSummary, "id" | "title">>;
  getConversation(id: string): Promise<Conversation>;
  deleteConversation(id: string): Promise<void>;
  sendMessage(input: SendMessageInput, onEvent: (event: ChatEvent) => void): Promise<void>;
}

export function createConversationClient(): ConversationClient {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";
  return new ApiConversationClient(baseUrl);
}

class ApiConversationClient implements ConversationClient {
  constructor(private readonly baseUrl: string) {}

  async listConversations(): Promise<ConversationSummary[]> {
    const response = await fetch(`${this.baseUrl}/api/conversations`);
    if (!response.ok) {
      throw new Error(`Conversation list failed with status ${response.status}.`);
    }
    return (await response.json()) as ConversationSummary[];
  }

  async createConversation(): Promise<Pick<ConversationSummary, "id" | "title">> {
    const response = await fetch(`${this.baseUrl}/api/conversations`, { method: "POST" });
    if (!response.ok) {
      throw new Error(`Conversation creation failed with status ${response.status}.`);
    }
    return (await response.json()) as Pick<ConversationSummary, "id" | "title">;
  }

  async getConversation(id: string): Promise<Conversation> {
    const response = await fetch(
      `${this.baseUrl}/api/conversations/${encodeURIComponent(id)}`,
    );
    if (!response.ok) {
      throw new Error(`Conversation loading failed with status ${response.status}.`);
    }
    return (await response.json()) as Conversation;
  }

  async deleteConversation(id: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/conversations/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    );
    if (!response.ok) {
      throw new Error(`Conversation deletion failed with status ${response.status}.`);
    }
  }

  sendMessage(
    input: SendMessageInput,
    onEvent: (event: ChatEvent) => void,
  ): Promise<void> {
    return streamChat(input.message, onEvent, {
      baseUrl: this.baseUrl,
      conversationId: input.conversationId ?? undefined,
    });
  }
}
