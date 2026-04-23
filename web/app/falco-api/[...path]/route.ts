import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const upstreamBase = (process.env.FALCO_API_PROXY_TARGET || "http://127.0.0.1:8000").replace(/\/$/, "");
const hopByHopHeaders = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function buildUpstreamUrl(pathParts: string[] | undefined, requestUrl: string) {
  const path = (pathParts || []).join("/");
  const incoming = new URL(requestUrl);
  const upstream = new URL(`${upstreamBase}/${path}`);
  upstream.search = incoming.search;
  return upstream;
}

function copyRequestHeaders(request: NextRequest) {
  const headers = new Headers();

  request.headers.forEach((value, key) => {
    if (!hopByHopHeaders.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });

  return headers;
}

function copyResponseHeaders(headers: Headers) {
  const nextHeaders = new Headers();

  headers.forEach((value, key) => {
    if (!hopByHopHeaders.has(key.toLowerCase())) {
      nextHeaders.set(key, value);
    }
  });

  return nextHeaders;
}

function isEventStream(headers: Headers) {
  return (headers.get("content-type") || "").toLowerCase().includes("text/event-stream");
}

function isIgnorableStreamTermination(error: unknown) {
  if (!(error instanceof Error)) {
    return false;
  }
  const text = `${error.name} ${error.message}`.toLowerCase();
  return text.includes("terminated") || text.includes("socket") || text.includes("pipe");
}

type RouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

async function proxy(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const upstreamUrl = buildUpstreamUrl(path, request.url);
  const headers = copyRequestHeaders(request);

  const init: RequestInit = {
    method: request.method,
    headers,
    redirect: "manual",
    cache: "no-store",
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    // Required by Node fetch when streaming request bodies.
    (init as RequestInit & { duplex: "half" }).duplex = "half";
  }

  try {
    const upstreamResponse = await fetch(upstreamUrl, init);
    const responseHeaders = copyResponseHeaders(upstreamResponse.headers);

    if (upstreamResponse.body && isEventStream(upstreamResponse.headers)) {
      const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();

      void (async () => {
        const reader = upstreamResponse.body!.getReader();
        const writer = writable.getWriter();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            if (value) {
              await writer.write(value);
            }
          }
        } catch (error) {
          if (!isIgnorableStreamTermination(error)) {
            console.error("Falco SSE proxy failed", error);
            try {
              await writer.abort(error);
            } catch {}
            return;
          }
        } finally {
          try {
            await writer.close();
          } catch {}
          reader.releaseLock();
        }
      })();

      return new Response(readable, {
        status: upstreamResponse.status,
        statusText: upstreamResponse.statusText,
        headers: responseHeaders,
      });
    }

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown proxy error";
    return Response.json({ error: message }, { status: 502 });
  }
}

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function OPTIONS(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}
