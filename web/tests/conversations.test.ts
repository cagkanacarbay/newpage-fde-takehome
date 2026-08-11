import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  createConversationClient,
  titleFromMessage,
} from "../lib/conversations";
import type { ChatEvent } from "../lib/sse";

describe("titleFromMessage", () => {
  it("uses the message as-is when short enough", () => {
    assert.equal(titleFromMessage("What are senolytics?"), "What are senolytics?");
  });

  it("caps long titles at ~60 chars on a word boundary", () => {
    const long =
      "How do epigenetic clocks compare with proteostasis markers when predicting all-cause mortality in older adults?";
    const title = titleFromMessage(long);
    assert.ok(title.length <= 61, `title too long: ${title.length}`);
    assert.ok(title.endsWith("…"));
    assert.ok(!title.slice(0, -1).endsWith(" "), "trailing space before ellipsis");
  });

  it("collapses whitespace and newlines", () => {
    assert.equal(titleFromMessage("  hello\n   world  "), "hello world");
  });
});

describe("MockConversationClient", () => {
  const createMock = () => createConversationClient();

  it("keeps conversations mock-backed when an API base URL is configured", async () => {
    const originalBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    process.env.NEXT_PUBLIC_API_BASE_URL = ".";
    try {
      const client = createConversationClient();
      const list = await client.listConversations();
      assert.ok(list.length >= 3, "configured builds keep the seeded conversations");
    } finally {
      if (originalBaseUrl === undefined) {
        delete process.env.NEXT_PUBLIC_API_BASE_URL;
      } else {
        process.env.NEXT_PUBLIC_API_BASE_URL = originalBaseUrl;
      }
    }
  });

  it("lists seeded conversations sorted by last activity, newest first", async () => {
    const client = createMock();
    const list = await client.listConversations();
    assert.ok(list.length >= 3, "sidebar needs several mock conversations");
    const stamps = list.map((item) => item.updated_at);
    assert.deepEqual(stamps, [...stamps].sort((a, b) => b.localeCompare(a)));
  });

  it("creates an empty conversation", async () => {
    const client = createMock();
    const created = await client.createConversation();
    const conversation = await client.getConversation(created.id);
    assert.equal(conversation.messages.length, 0);
    assert.equal(conversation.title, "New conversation");
  });

  it("deletes a conversation", async () => {
    const client = createMock();
    const created = await client.createConversation();
    await client.deleteConversation(created.id);
    const list = await client.listConversations();
    assert.ok(!list.some((item) => item.id === created.id));
  });

  it("sendMessage on a new conversation emits the conversation event first, then tokens, then done", async () => {
    const client = createMock();
    const events: ChatEvent[] = [];
    await client.sendMessage(
      { conversationId: null, message: "What did the D+Q trial measure?" },
      (event) => events.push(event),
    );

    const first = events[0];
    assert.equal(first.type, "conversation");
    if (first.type === "conversation") {
      assert.equal(first.title, "What did the D+Q trial measure?");
    }
    assert.ok(events.some((event) => event.type === "token"));
    assert.ok(events.some((event) => event.type === "citations"));
    assert.equal(events.at(-1)?.type, "done");
  });

  it("persists the exchange so getConversation returns it", async () => {
    const client = createMock();
    const events: ChatEvent[] = [];
    await client.sendMessage(
      { conversationId: null, message: "Does rapamycin extend lifespan?" },
      (event) => events.push(event),
    );

    const conversationEvent = events[0];
    assert.equal(conversationEvent.type, "conversation");
    if (conversationEvent.type !== "conversation") {
      return;
    }

    const conversation = await client.getConversation(conversationEvent.id);
    assert.deepEqual(
      conversation.messages.map((message) => message.role),
      ["user", "assistant"],
    );
    assert.ok(conversation.messages[1].text.length > 0);
    assert.ok(conversation.messages[1].citations.length > 0);
  });

  it("appends to an existing conversation without changing its title", async () => {
    const client = createMock();
    const created = await client.createConversation();
    await client.sendMessage(
      { conversationId: created.id, message: "First question here" },
      () => {},
    );
    const conversation = await client.getConversation(created.id);
    assert.equal(conversation.title, "New conversation");
    assert.equal(conversation.messages.length, 2);
  });

  it("replaces dropped history with a system notice", async () => {
    const client = createMock();
    const firstEvents: ChatEvent[] = [];
    await client.sendMessage(
      { conversationId: null, message: "word ".repeat(100_001) },
      (event) => firstEvents.push(event),
    );
    const firstConversation = firstEvents[0];
    assert.equal(firstConversation.type, "conversation");
    if (firstConversation.type !== "conversation") {
      return;
    }

    const secondEvents: ChatEvent[] = [];
    await client.sendMessage(
      { conversationId: firstConversation.id, message: "Follow up question" },
      (event) => secondEvents.push(event),
    );

    assert.ok(secondEvents.some((event) => event.type === "dropped"));
    const conversation = await client.getConversation(firstConversation.id);
    assert.equal(conversation.messages[0]?.role, "system");
    assert.match(conversation.messages[0]?.text ?? "", /1 earlier turn is not included/);
    assert.deepEqual(
      conversation.messages.slice(1).map((message) => message.role),
      ["user", "assistant"],
    );
  });

  it("rejects sending to an unknown conversation id", async () => {
    const client = createMock();
    await assert.rejects(
      client.sendMessage({ conversationId: "nope", message: "hi" }, () => {}),
      /Unknown conversation/,
    );
  });
});
