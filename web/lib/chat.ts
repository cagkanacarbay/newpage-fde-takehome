import { readSseStream, type ChatEvent, type Citation } from "./sse";

const mockCitation: Citation = {
  document_id: "015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human",
  page: 2,
  heading_path: ["Research in context", "1. Introduction"],
  bbox: { l: 310.5, t: 332.6, r: 561.6, b: 300.3 },
  snippet: "Dasatinib plus quercetin was evaluated in a first-in-human pilot study.",
};

const mockCitationSecond: Citation = {
  document_id: "013-ivimey-cook-2025-rapamycin-not-metformin-dietary-restriction",
  page: 3,
  heading_path: ["2. Results"],
  bbox: { l: 56.7, t: 640.1, r: 300.2, b: 590.4 },
  snippet: "Rapamycin, but not metformin, mirrored the lifespan extension of dietary restriction.",
};

const mockEvents: ChatEvent[] = [
  {
    type: "token",
    text: "The first-in-human pilot study evaluated **dasatinib plus quercetin** (D+Q) in people with idiopathic pulmonary fibrosis. ",
  },
  {
    type: "token",
    text: "It reported improved physical-function measures, while its small size means the result needs larger controlled studies.\n\n",
  },
  {
    type: "token",
    text: "Related work compares geroprotectors directly: rapamycin, but not metformin, mirrors the lifespan effects of dietary restriction.",
  },
  { type: "citations", citations: [mockCitation, mockCitationSecond, mockCitation] },
  { type: "done" },
];

const pause = () => new Promise<void>((resolve) => setTimeout(resolve, 80));

/** Mock assistant reply: tokens, citations, done. No conversation event. */
export async function streamMockReply(onEvent: (event: ChatEvent) => void): Promise<void> {
  for (const event of mockEvents) {
    await pause();
    onEvent(event);
  }
}

export type StreamChatOptions = {
  conversationId?: string;
};

export async function streamChat(
  message: string,
  onEvent: (event: ChatEvent) => void,
  options: StreamChatOptions = {},
): Promise<void> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");

  if (!baseUrl) {
    await streamMockReply(onEvent);
    return;
  }

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
