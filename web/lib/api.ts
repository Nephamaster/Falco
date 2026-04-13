export type ChatStreamCallbacks = {
  onStart?: () => void;
  onDelta: (delta: string) => void;
  onDone?: (answer: string) => void;
};

export async function streamChat(params: {
  apiBase: string;
  threadId: string;
  message: string;
  callbacks: ChatStreamCallbacks;
}) {
  const { apiBase, threadId, message, callbacks } = params;
  const endpoint = `${apiBase.replace(/\/$/, "")}/api/v1/chat/stream`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, message }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalAnswer = "";
  let started = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const lines = chunk.split("\n");
      const event = lines.find((line) => line.startsWith("event:"))?.replace("event:", "").trim();
      const dataLine = lines.find((line) => line.startsWith("data:"))?.replace("data:", "").trim();
      if (!event || !dataLine) {
        continue;
      }
      const data = JSON.parse(dataLine) as { content?: string; answer?: string };

      if (event === "start" && !started) {
        callbacks.onStart?.();
        started = true;
      }
      if (event === "delta" && data.content) {
        finalAnswer += data.content;
        callbacks.onDelta(data.content);
      }
      if (event === "done") {
        const answer = data.answer ?? finalAnswer;
        callbacks.onDone?.(answer);
      }
    }
  }
}
