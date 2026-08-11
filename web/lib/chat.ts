import { readSseStream, type ChatEvent, type Citation } from "./sse";

const mockCitation: Citation = {
  document_id: "001-sanada-2025-hallmarks-of-aging-therapeutic-targets",
  page: 3,
  heading_path: ["2 Hallmarks of aging and possible interventions", "Figure 2"],
  bbox: { l: 62.4, t: 498.9, r: 532.9, b: 446.2 },
  snippet:
    "The hallmarks of aging can be categorized into three interconnected layers: primary, antagonistic, and integrative.",
};

const mockCitationSecond: Citation = {
  document_id: "001-sanada-2025-hallmarks-of-aging-therapeutic-targets",
  page: 3,
  heading_path: ["2 Hallmarks of aging and possible interventions", "Table 1"],
  bbox: { l: 53.2, t: 381.6, r: 545.7, b: 201.0 },
  snippet:
    "Table 1 lists therapeutic strategies for each hallmark, including NAD+ boosters, rapamycin, metformin, and senolytics.",
};

const mockEvents: ChatEvent[] = [
  {
    type: "token",
    text: "The hallmarks of aging form **three interconnected layers**. Primary hallmarks reflect accumulating cellular damage. ",
  },
  {
    type: "token",
    text: "Antagonistic hallmarks are compensatory responses that can become harmful, while integrative hallmarks drive systemic decline.\n\n",
  },
  {
    type: "token",
    text: "The review maps interventions to each hallmark. Examples include NAD+ boosters, rapamycin or metformin, and senolytics.",
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
