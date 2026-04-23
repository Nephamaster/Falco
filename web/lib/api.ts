export type ChatStreamCallbacks = {
  onStart?: () => void;
  onDelta: (delta: string) => void;
  onDone?: (answer: string) => void;
};

type ApiBaseParam = {
  apiBase: string;
};

export type HealthResponse = {
  status: string;
  service: string;
};

export type MCPCatalogResponse = {
  result: string;
};

export type RAGSearchResponse = {
  result: string;
};

export type RAGIndexResponse = {
  message: string;
};

function buildEndpoint(apiBase: string, path: string) {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchHealth({ apiBase }: ApiBaseParam) {
  const response = await fetch(buildEndpoint(apiBase, "/api/v1/health"), {
    method: "GET",
    cache: "no-store",
  });
  return parseJson<HealthResponse>(response);
}

export async function fetchMCPCatalog({ apiBase }: ApiBaseParam) {
  const response = await fetch(buildEndpoint(apiBase, "/api/v1/mcp/catalog"), {
    method: "GET",
    cache: "no-store",
  });
  return parseJson<MCPCatalogResponse>(response);
}

export async function searchRAG(params: { apiBase: string; query: string; topK: number }) {
  const response = await fetch(buildEndpoint(params.apiBase, "/api/v1/rag/search"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: params.query, top_k: params.topK }),
  });
  return parseJson<RAGSearchResponse>(response);
}

export async function indexRAG(params: { apiBase: string; path: string; dropOld: boolean }) {
  const response = await fetch(buildEndpoint(params.apiBase, "/api/v1/rag/index"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: params.path, drop_old: params.dropOld }),
  });
  return parseJson<RAGIndexResponse>(response);
}

export async function streamChat(params: {
  apiBase: string;
  threadId: string;
  message: string;
  userResponsePreference: string;
  resume?: boolean;
  callbacks: ChatStreamCallbacks;
}) {
  const { apiBase, threadId, message, userResponsePreference, resume = false, callbacks } = params;
  const endpoint = buildEndpoint(apiBase, "/api/v1/chat/stream");
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      message,
      user_response_preference: userResponsePreference,
      resume,
    }),
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
