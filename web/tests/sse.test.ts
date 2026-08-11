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

test("uses fallback citations that match the bundled sample PDF", async () => {
  const received: ChatEvent[] = [];

  await streamChat("What did the first human senolytic study find?", (event) => {
    received.push(event);
  });

  assert.equal(received.at(-1)?.type, "done");
  assert.deepEqual(received.at(-2), {
    type: "citations",
    citations: [
      {
        document_id: "001-sanada-2025-hallmarks-of-aging-therapeutic-targets",
        page: 3,
        heading_path: ["2 Hallmarks of aging and possible interventions", "Figure 2"],
        bbox: { l: 62.4, t: 498.9, r: 532.9, b: 446.2 },
        snippet:
          "The hallmarks of aging can be categorized into three interconnected layers: primary, antagonistic, and integrative.",
      },
      {
        document_id: "001-sanada-2025-hallmarks-of-aging-therapeutic-targets",
        page: 3,
        heading_path: ["2 Hallmarks of aging and possible interventions", "Table 1"],
        bbox: { l: 53.2, t: 381.6, r: 545.7, b: 201.0 },
        snippet:
          "Table 1 lists therapeutic strategies for each hallmark, including NAD+ boosters, rapamycin, metformin, and senolytics.",
      },
      {
        document_id: "001-sanada-2025-hallmarks-of-aging-therapeutic-targets",
        page: 3,
        heading_path: ["2 Hallmarks of aging and possible interventions", "Figure 2"],
        bbox: { l: 62.4, t: 498.9, r: 532.9, b: 446.2 },
        snippet:
          "The hallmarks of aging can be categorized into three interconnected layers: primary, antagonistic, and integrative.",
      },
    ],
  });
});
