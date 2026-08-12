import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { renderToStaticMarkup } from "react-dom/server";

import { MessageList } from "../components/chat/message-list";
import type { UiMessage } from "../components/chat/types";
import type { Citation } from "../lib/sse";

function citation(snippet: string): Citation {
  return {
    document_id: "paper",
    page: 2,
    heading_path: ["Results"],
    bbox: { l: 1, t: 2, r: 3, b: 0 },
    snippet,
  };
}

function renderAssistant(text: string, citations: Citation[]): string {
  const messages: UiMessage[] = [
    { id: "answer", role: "assistant", text, citations },
  ];
  return renderToStaticMarkup(
    <MessageList
      messages={messages}
      streaming={false}
      onOpenCitation={() => undefined}
      onRetry={() => undefined}
    />,
  );
}

describe("MessageList citation controls", () => {
  it("renders fallback controls for markerless stored answers", () => {
    const html = renderAssistant("A saved answer without markers.", [citation("Legacy source")]);

    assert.match(html, /aria-label="Sources"/);
    assert.match(html, /Open citation 1: Legacy source/);
  });

  it("preserves marker indexes for distinct chunks at one location", () => {
    const html = renderAssistant(
      "First claim [1]. Second claim [2].",
      [citation("First chunk"), citation("Second chunk")],
    );

    assert.match(html, /Open citation 1: First chunk/);
    assert.match(html, /Open citation 2: Second chunk/);
    assert.doesNotMatch(html, /aria-label="Sources"/);
  });

  it("keeps punctuation attached to inline citation buttons", () => {
    const html = renderAssistant(
      "One result [1]. Conflicting results [1][2].",
      [citation("First chunk"), citation("Second chunk")],
    );

    assert.match(html, /<\/button>\./);
    assert.match(html, /<\/button><button/);
    assert.doesNotMatch(html, /<\/button>\s+\./);
    assert.doesNotMatch(html, /mx-0\.5/);
  });
});
