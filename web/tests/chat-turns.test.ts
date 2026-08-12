import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { retryFailedTurn, withHistoryNotice } from "../components/chat/chat-turns";
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
