import { streamChat, streamMockReply } from "./chat";
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

/**
 * The conversation-persistence contract the UI talks to. The real backend
 * (ticket #32) and the in-memory mock both implement this, so swapping them
 * is a no-op for the UI.
 */
export interface ConversationClient {
  listConversations(): Promise<ConversationSummary[]>;
  createConversation(): Promise<ConversationSummary>;
  getConversation(id: string): Promise<Conversation>;
  deleteConversation(id: string): Promise<void>;
  sendMessage(input: SendMessageInput, onEvent: (event: ChatEvent) => void): Promise<void>;
}

const TITLE_MAX_LENGTH = 60;

/** Conversation title from the first user message, capped at ~60 chars. */
export function titleFromMessage(message: string): string {
  const collapsed = message.replace(/\s+/g, " ").trim();
  if (collapsed.length <= TITLE_MAX_LENGTH) {
    return collapsed;
  }
  const cut = collapsed.lastIndexOf(" ", TITLE_MAX_LENGTH);
  return `${collapsed.slice(0, cut > 0 ? cut : TITLE_MAX_LENGTH)}…`;
}

export function createConversationClient(
  baseUrl: string | undefined = process.env.NEXT_PUBLIC_API_BASE_URL,
): ConversationClient {
  const normalized = baseUrl?.replace(/\/$/, "");
  return normalized ? new HttpConversationClient(normalized) : new MockConversationClient();
}

class HttpConversationClient implements ConversationClient {
  constructor(private readonly baseUrl: string) {}

  async listConversations(): Promise<ConversationSummary[]> {
    return this.request("/api/conversations");
  }

  async createConversation(): Promise<ConversationSummary> {
    return this.request("/api/conversations", { method: "POST" });
  }

  async getConversation(id: string): Promise<Conversation> {
    return this.request(`/api/conversations/${encodeURIComponent(id)}`);
  }

  async deleteConversation(id: string): Promise<void> {
    await this.request(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  }

  async sendMessage(
    input: SendMessageInput,
    onEvent: (event: ChatEvent) => void,
  ): Promise<void> {
    await streamChat(
      input.message,
      onEvent,
      input.conversationId ? { conversationId: input.conversationId } : {},
    );
  }

  private async request(path: string, init?: RequestInit) {
    const response = await fetch(`${this.baseUrl}${path}`, init);
    if (!response.ok) {
      throw new Error(`Request to ${path} failed with status ${response.status}.`);
    }
    if (response.status === 204) {
      return undefined;
    }
    return response.json();
  }
}

type MockRecord = Conversation & { updated_at: string };

function seedConversations(): MockRecord[] {
  const now = Date.now();
  const seed = (
    offsetMinutes: number,
    title: string,
    exchange: [string, string],
  ): MockRecord => {
    const stamp = new Date(now - offsetMinutes * 60_000).toISOString();
    return {
      id: crypto.randomUUID(),
      title,
      updated_at: stamp,
      messages: [
        { role: "user", text: exchange[0], citations: [], created_at: stamp },
        { role: "assistant", text: exchange[1], citations: [], created_at: stamp },
      ],
    };
  };

  return [
    seed(
      25,
      "Senolytics in idiopathic pulmonary fibrosis",
      [
        "What did the first-in-human senolytics trials report?",
        "The D+Q pilot studies reported improved physical-function measures in IPF, with the caveat that larger controlled trials are needed.",
      ],
    ),
    seed(
      60 * 5,
      "Rapamycin vs metformin as geroprotectors",
      [
        "Does metformin extend lifespan like rapamycin does?",
        "Across the dietary-restriction comparison, rapamycin mirrored the lifespan effects of dietary restriction while metformin did not.",
      ],
    ),
    seed(
      60 * 26,
      "Hallmarks of aging as therapeutic targets",
      [
        "Which hallmarks of aging are druggable today?",
        "The hallmarks framework groups targets such as cellular senescence, mitochondrial dysfunction, and epigenetic alteration; several already have candidate interventions.",
      ],
    ),
    seed(
      60 * 50,
      "Epigenetic clocks across species",
      [
        "Do epigenetic clocks work in insects?",
        "A pan-mammalian age predictor exists, but there is still no evidence for an environmentally responsive epigenetic clock in insects.",
      ],
    ),
  ];
}

class MockConversationClient implements ConversationClient {
  private readonly records = new Map<string, MockRecord>();

  constructor() {
    for (const record of seedConversations()) {
      this.records.set(record.id, record);
    }
  }

  listConversations(): Promise<ConversationSummary[]> {
    const summaries = [...this.records.values()].map(
      ({ id, title, updated_at }) => ({ id, title, updated_at }),
    );
    summaries.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    return Promise.resolve(summaries);
  }

  createConversation(): Promise<ConversationSummary> {
    const record = this.createRecord("New conversation");
    return Promise.resolve(this.summaryOf(record));
  }

  getConversation(id: string): Promise<Conversation> {
    const record = this.records.get(id);
    if (!record) {
      return Promise.reject(new Error(`Unknown conversation: ${id}`));
    }
    return Promise.resolve({
      id: record.id,
      title: record.title,
      messages: record.messages.map((message) => ({ ...message })),
    });
  }

  deleteConversation(id: string): Promise<void> {
    this.records.delete(id);
    return Promise.resolve();
  }

  async sendMessage(
    input: SendMessageInput,
    onEvent: (event: ChatEvent) => void,
  ): Promise<void> {
    const record = input.conversationId
      ? this.records.get(input.conversationId)
      : undefined;
    const target = record ?? this.createRecord(titleFromMessage(input.message));

    if (!record && input.conversationId) {
      throw new Error(`Unknown conversation: ${input.conversationId}`);
    }

    onEvent({ type: "conversation", id: target.id, title: target.title });

    const stamp = new Date().toISOString();
    target.messages.push({
      role: "user",
      text: input.message,
      citations: [],
      created_at: stamp,
    });

    let text = "";
    let citations: Citation[] = [];
    await streamMockReply((event) => {
      if (event.type === "token") {
        text += event.text;
      }
      if (event.type === "citations") {
        citations = event.citations;
      }
      onEvent(event);
    });

    target.messages.push({
      role: "assistant",
      text,
      citations,
      created_at: new Date().toISOString(),
    });
    target.updated_at = new Date().toISOString();
  }

  private createRecord(title: string): MockRecord {
    const record: MockRecord = {
      id: crypto.randomUUID(),
      title,
      updated_at: new Date().toISOString(),
      messages: [],
    };
    this.records.set(record.id, record);
    return record;
  }

  private summaryOf(record: MockRecord): ConversationSummary {
    return { id: record.id, title: record.title, updated_at: record.updated_at };
  }
}
