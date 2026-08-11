import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { DraftStore } from "../lib/drafts";

describe("DraftStore", () => {
  it("keeps unsent drafts per conversation id", () => {
    const drafts = new DraftStore();
    drafts.set("a", "draft for a");
    drafts.set("b", "draft for b");
    assert.equal(drafts.get("a"), "draft for a");
    assert.equal(drafts.get("b"), "draft for b");
  });

  it("returns empty string for conversations without a draft", () => {
    assert.equal(new DraftStore().get("missing"), "");
  });

  it("clears the draft when set back to empty", () => {
    const drafts = new DraftStore();
    drafts.set("a", "typed something");
    drafts.set("a", "");
    assert.equal(drafts.get("a"), "");
  });

  it("overwrites a previous draft for the same conversation", () => {
    const drafts = new DraftStore();
    drafts.set("a", "first");
    drafts.set("a", "second");
    assert.equal(drafts.get("a"), "second");
  });
});
