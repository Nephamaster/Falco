"use client";

import React, { useMemo, useState } from "react";

import { streamChat } from "../lib/api";

type ChatItem = {
  role: "user" | "assistant";
  content: string;
};

const defaultApiBase = process.env.NEXT_PUBLIC_FALCO_API_BASE || "http://127.0.0.1:8000";

export default function HomePage() {
  const [apiBase, setApiBase] = useState(defaultApiBase);
  const [threadId, setThreadId] = useState("default");
  const [draft, setDraft] = useState("");
  const [items, setItems] = useState<ChatItem[]>([]);
  const [pending, setPending] = useState(false);

  const canSend = useMemo(() => !pending && draft.trim().length > 0, [pending, draft]);

  const handleSend = async () => {
    const message = draft.trim();
    if (!message || pending) {
      return;
    }

    setDraft("");
    setPending(true);
    setItems((prev) => [...prev, { role: "user", content: message }, { role: "assistant", content: "" }]);

    try {
      await streamChat({
        apiBase,
        threadId,
        message,
        callbacks: {
          onDelta: (delta) => {
            setItems((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (!last || last.role !== "assistant") {
                return prev;
              }
              next[next.length - 1] = { role: "assistant", content: last.content + delta };
              return next;
            });
          },
          onDone: (answer) => {
            setItems((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (!last || last.role !== "assistant") {
                return prev;
              }
              next[next.length - 1] = { role: "assistant", content: answer };
              return next;
            });
          },
        },
      });
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown error";
      setItems((prev) => [...prev, { role: "assistant", content: `请求失败: ${text}` }]);
    } finally {
      setPending(false);
    }
  };

  return (
    <main className="app">
      <section className="container">
        <header className="header">
          <h1 className="title">Falco Control Panel</h1>
          <div className="row">
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="API Base URL" />
            <input value={threadId} onChange={(e) => setThreadId(e.target.value)} placeholder="thread_id" />
          </div>
        </header>

        <section className="messages">
          {items.map((item, idx) => (
            <article key={`${item.role}-${idx}`} className={`msg ${item.role}`}>
              {item.content || (item.role === "assistant" ? "..." : "")}
            </article>
          ))}
        </section>

        <section className="composer">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="输入你的任务目标，Falco 将进行编排并调用工具..."
          />
          <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
            <span className="hint">当前线程: {threadId}</span>
            <button disabled={!canSend} onClick={handleSend}>
              {pending ? "处理中..." : "发送"}
            </button>
          </div>
        </section>
      </section>
    </main>
  );
}
