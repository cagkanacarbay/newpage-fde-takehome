import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { createConversationClient } from "../lib/conversations";
import type { ChatEvent } from "../lib/sse";

describe("ConversationClient", () => {
  it("loads conversations from the backend", async () => {
    const originalBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const originalFetch = globalThis.fetch;
    process.env.NEXT_PUBLIC_API_BASE_URL = ".";
    globalThis.fetch = async (input) => {
      assert.equal(input, "./api/conversations");
      return Response.json([
        {
          id: "conversation-1",
          title: "Senolytics evidence",
          updated_at: "2026-08-11T10:00:00Z",
        },
      ]);
    };
    try {
      const client = createConversationClient();
      const list = await client.listConversations();
      assert.deepEqual(list, [
        {
          id: "conversation-1",
          title: "Senolytics evidence",
          updated_at: "2026-08-11T10:00:00Z",
        },
      ]);
    } finally {
      globalThis.fetch = originalFetch;
      if (originalBaseUrl === undefined) {
        delete process.env.NEXT_PUBLIC_API_BASE_URL;
      } else {
        process.env.NEXT_PUBLIC_API_BASE_URL = originalBaseUrl;
      }
    }
  });

  it("creates an empty conversation through the backend", async () => {
    const originalBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const originalFetch = globalThis.fetch;
    process.env.NEXT_PUBLIC_API_BASE_URL = ".";
    globalThis.fetch = async (input, init) => {
      assert.equal(input, "./api/conversations");
      assert.equal(init?.method, "POST");
      return Response.json({ id: "conversation-2", title: "New conversation" });
    };
    try {
      const client = createConversationClient();
      assert.deepEqual(await client.createConversation(), {
        id: "conversation-2",
        title: "New conversation",
      });
    } finally {
      globalThis.fetch = originalFetch;
      if (originalBaseUrl === undefined) {
        delete process.env.NEXT_PUBLIC_API_BASE_URL;
      } else {
        process.env.NEXT_PUBLIC_API_BASE_URL = originalBaseUrl;
      }
    }
  });

  it("loads one conversation and its stored messages", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input) => {
      assert.equal(input, "/api/conversations/conversation-3");
      return Response.json({
        id: "conversation-3",
        title: "Rapamycin evidence",
        messages: [
          {
            role: "user",
            text: "What did the study find?",
            citations: [],
            created_at: "2026-08-11T10:00:00Z",
          },
        ],
      });
    };
    try {
      const conversation = await createConversationClient().getConversation(
        "conversation-3",
      );
      assert.equal(conversation.messages[0].text, "What did the study find?");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("deletes a conversation through the backend", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input, init) => {
      assert.equal(input, "/api/conversations/conversation-4");
      assert.equal(init?.method, "DELETE");
      return new Response(null, { status: 204 });
    };
    try {
      await createConversationClient().deleteConversation("conversation-4");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("streams a message through the backend", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input, init) => {
      assert.equal(input, "/api/chat");
      assert.equal(init?.method, "POST");
      assert.deepEqual(JSON.parse(String(init?.body)), {
        conversation_id: "conversation-5",
        message: "What did the D+Q trial measure?",
      });
      return new Response(
        [
          'data: {"type":"conversation","id":"conversation-5","title":"D+Q trial"}',
          'data: {"type":"token","text":"Evidence"}',
          'data: {"type":"done"}',
          "",
        ].join("\n\n"),
      );
    };
    try {
      const events: ChatEvent[] = [];
      await createConversationClient().sendMessage(
        {
          conversationId: "conversation-5",
          message: "What did the D+Q trial measure?",
        },
        (event) => events.push(event),
      );
      assert.deepEqual(events.map((event) => event.type), [
        "conversation",
        "token",
        "done",
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

});
