import assert from "node:assert/strict";
import test from "node:test";

import { readSseStream, type ChatEvent } from "../lib/sse";

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

test("reads contract events when JSON and event separators span chunks", async () => {
  const received: ChatEvent[] = [];

  await readSseStream(
    streamFrom([
      'data: {"type":"token","text":"A lon',
      'gevity finding"}\n\ndata: {"type":"history_notice","text":"Earlier messages were dropped to fit the context window"}',
      '\n\ndata: {"type":"done"}',
      "\n\n",
    ]),
    (event) => received.push(event),
  );

  assert.deepEqual(received, [
    { type: "token", text: "A longevity finding" },
    {
      type: "history_notice",
      text: "Earlier messages were dropped to fit the context window",
    },
    { type: "done" },
  ]);
});
