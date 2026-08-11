export type Citation = {
  document_id: string;
  page: number;
  heading_path: string[];
  bbox: {
    l: number;
    t: number;
    r: number;
    b: number;
  };
  snippet: string;
};

export type ChatEvent =
  | { type: "conversation"; id: string; title: string }
  | { type: "token"; text: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "dropped"; turns: number }
  | { type: "done" }
  | { type: "error"; message: string };

function isChatEvent(value: unknown): value is ChatEvent {
  if (typeof value !== "object" || value === null || !("type" in value)) {
    return false;
  }

  return ["conversation", "token", "citations", "dropped", "done", "error"].includes(
    (value as { type: unknown }).type as string,
  );
}

export async function readSseStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];

  const dispatch = () => {
    if (dataLines.length === 0) {
      return;
    }

    const payload: unknown = JSON.parse(dataLines.join("\n"));
    dataLines = [];

    if (!isChatEvent(payload)) {
      throw new Error("Received an unknown chat stream event.");
    }

    onEvent(payload);
  };

  const readLine = (line: string) => {
    if (line === "") {
      dispatch();
      return;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      readLine(buffer.slice(0, newlineIndex).replace(/\r$/, ""));
      buffer = buffer.slice(newlineIndex + 1);
      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer !== "") {
    readLine(buffer.replace(/\r$/, ""));
  }
  dispatch();
}
