import assert from "node:assert/strict";
import test from "node:test";

import { readSseStream, type ChatEvent } from "../lib/sse";
import { streamChat } from "../lib/chat";

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
      'gevity finding"}\n\ndata: {"type":"done"}',
      "\n\n",
    ]),
    (event) => received.push(event),
  );

  assert.deepEqual(received, [
    { type: "token", text: "A longevity finding" },
    { type: "done" },
  ]);
});

test("uses the full contract citation payload in the standalone mock", async () => {
  const received: ChatEvent[] = [];

  await streamChat("What did the first human senolytic study find?", (event) => {
    received.push(event);
  });

  assert.equal(received.at(-1)?.type, "done");
  assert.deepEqual(received.at(-2), {
    type: "citations",
    citations: [
      {
        document_id: "015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human",
        page: 2,
        heading_path: ["Research in context", "1. Introduction"],
        bbox: { l: 310.5, t: 332.6, r: 561.6, b: 300.3 },
        snippet:
          "Dasatinib plus quercetin was evaluated in a first-in-human pilot study.",
      },
      {
        document_id: "013-ivimey-cook-2025-rapamycin-not-metformin-dietary-restriction",
        page: 3,
        heading_path: ["2. Results"],
        bbox: { l: 56.7, t: 640.1, r: 300.2, b: 590.4 },
        snippet:
          "Rapamycin, but not metformin, mirrored the lifespan extension of dietary restriction.",
      },
      {
        document_id: "015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human",
        page: 2,
        heading_path: ["Research in context", "1. Introduction"],
        bbox: { l: 310.5, t: 332.6, r: 561.6, b: 300.3 },
        snippet:
          "Dasatinib plus quercetin was evaluated in a first-in-human pilot study.",
      },
    ],
  });
});
