import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { retryFailedTurn, withDroppedTurnsNotice } from "../components/chat/chat-turns";
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

describe("withDroppedTurnsNotice", () => {
  it("replaces the oldest complete turns with an inline notice", () => {
    const result = withDroppedTurnsNotice(messages, 1, "notice");

    assert.deepEqual(result, [
      {
        id: "notice",
        role: "system",
        text: "1 earlier turn is not included in the assistant context.",
        citations: [],
      },
      { id: "q2", role: "user", text: "Second question", citations: [] },
      { id: "a2", role: "assistant", text: "Second answer", citations: [] },
    ]);
  });

  it("replaces an earlier notice when the cumulative dropped-turn count grows", () => {
    const threeTurns: UiMessage[] = [
      { id: "q1", role: "user", text: "First question", citations: [] },
      { id: "a1", role: "assistant", text: "First answer", citations: [] },
      { id: "q2", role: "user", text: "Second question", citations: [] },
      { id: "a2", role: "assistant", text: "Second answer", citations: [] },
      { id: "q3", role: "user", text: "Third question", citations: [] },
      { id: "a3", role: "assistant", text: "Third answer", citations: [] },
    ];

    const firstTrim = withDroppedTurnsNotice(threeTurns, 1, "notice-1");
    const secondTrim = withDroppedTurnsNotice(firstTrim, 1, "notice-2");

    assert.deepEqual(secondTrim, [
      {
        id: "notice-2",
        role: "system",
        text: "2 earlier turns are not included in the assistant context.",
        citations: [],
      },
      { id: "q3", role: "user", text: "Third question", citations: [] },
      { id: "a3", role: "assistant", text: "Third answer", citations: [] },
    ]);
  });
});
