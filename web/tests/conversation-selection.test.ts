import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  ConversationListRefresh,
  ConversationSelection,
} from "../lib/conversation-selection";

describe("ConversationSelection", () => {
  it("rejects an earlier load after returning to the same conversation", () => {
    const selection = new ConversationSelection();

    const firstA = selection.select("conversation-a");
    selection.select("conversation-b");
    const secondA = selection.select("conversation-a");

    assert.equal(selection.isCurrent(firstA), false);
    assert.equal(selection.isCurrent(secondA), true);
  });

  it("rejects an earlier conversation-list refresh", () => {
    const refresh = new ConversationListRefresh();

    const initial = refresh.start();
    const afterSend = refresh.start();

    assert.equal(refresh.isCurrent(initial), false);
    assert.equal(refresh.isCurrent(afterSend), true);
  });
});
