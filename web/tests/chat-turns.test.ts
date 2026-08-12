import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  applyAssistantEvent,
  retryFailedTurn,
  withHistoryNotice,
} from "../components/chat/chat-turns";
import type { UiMessage } from "../components/chat/types";

const messages: UiMessage[] = [
  { id: "q1", role: "user", text: "First question", citations: [] },
  {
    id: "a1",
    role: "assistant",
    text: "",
    citations: [],
    error: "First request failed",
    replyTo: "q1",
    retryMessage: "First question",
  },
  { id: "q2", role: "user", text: "Second question", citations: [] },
  { id: "a2", role: "assistant", text: "Second answer", citations: [] },
];

describe("retryFailedTurn", () => {
  it("retries the failed turn without changing later turns", () => {
    const retry = retryFailedTurn(messages, "a1");

    assert.deepEqual(retry, {
      message: "First question",
      messages: [
        { id: "q2", role: "user", text: "Second question", citations: [] },
        { id: "a2", role: "assistant", text: "Second answer", citations: [] },
      ],
    });
  });
});

describe("withHistoryNotice", () => {
  it("keeps visible turns and replaces an earlier notice", () => {
    const result = withHistoryNotice(
      [
        {
          id: "old-notice",
          role: "system",
          text: "Old notice",
          citations: [],
        },
        ...messages,
      ],
      "Earlier messages were dropped to fit the context window",
      "new-notice",
    );

    assert.deepEqual(result, [
      {
        id: "new-notice",
        role: "system",
        text: "Earlier messages were dropped to fit the context window",
        citations: [],
      },
      ...messages,
    ]);
  });
});

describe("applyAssistantEvent", () => {
  it("records first-token time and applies a changed verification result", () => {
    const waiting: UiMessage = {
      id: "answer",
      role: "assistant",
      text: "",
      citations: [],
      responseStartedAtMs: 1_000,
    };

    const streaming = applyAssistantEvent(
      waiting,
      { type: "token", text: "Draft claim [1]." },
      3_340,
    );
    const verifying = applyAssistantEvent(
      streaming,
      { type: "verification", status: "started" },
      4_000,
    );
    const updated = applyAssistantEvent(
      verifying,
      {
        type: "verification",
        status: "complete",
        text: "Supported claim [1].",
        citations: [],
        changed: true,
      },
      5_000,
    );

    assert.equal(streaming.firstTokenSeconds, 2.34);
    assert.equal(streaming.text, "Draft claim [1].");
    assert.equal(verifying.verification, "verifying");
    assert.equal(updated.text, "Supported claim [1].");
    assert.equal(updated.verification, "updated");
  });
});
