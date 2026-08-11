import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { ConversationSelection } from "../lib/conversation-selection";

describe("ConversationSelection", () => {
  it("rejects a completed load for a conversation the user has left", () => {
    const selection = new ConversationSelection();

    selection.select("conversation-a");
    selection.select("conversation-b");

    assert.equal(selection.isCurrent("conversation-a"), false);
    assert.equal(selection.isCurrent("conversation-b"), true);
  });
});
