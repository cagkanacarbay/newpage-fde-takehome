import type { Citation } from "@/lib/sse";

export type UiMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  citations: Citation[];
  error?: string;
  replyTo?: string;
  retryMessage?: string;
  responseStartedAtMs?: number;
  firstTokenSeconds?: number;
  verification?: "verifying" | "verified" | "updated" | "failed";
};

/** The PDF panel target: which citation of which turn is open. */
export type PdfTarget = {
  documentId: string;
  citation: Citation;
  chipIndex: number;
  /** React key of the message that owns the chip, for active-state styling. */
  messageId: string;
};
