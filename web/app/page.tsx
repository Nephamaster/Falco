"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

import { fetchHealth, fetchMCPCatalog, streamChat } from "../lib/api";

type ChatItem = {
  role: "user" | "assistant";
  content: string;
};

type ChatSession = {
  id: string;
  title: string;
  threadId: string;
  items: ChatItem[];
  updatedAt: number;
};

type ConnectionState = "idle" | "checking" | "online" | "offline";

const defaultApiBase = process.env.NEXT_PUBLIC_FALCO_API_BASE || "/falco-api";
const storageKey = "falco-web-sessions";
const activeSessionKey = "falco-web-active-session";
const initialSession = createSession({ id: "default", title: "默认会话", threadId: "default" });
const starterPrompts = [
  "帮我整理当前知识库的重点，并给出下一步行动建议。",
  "基于已有技能和工具，帮我规划一个可执行的工作流。",
  "读取当前线程上下文，帮我总结状态并提出后续任务。",
];

function createSession(seed?: Partial<ChatSession>): ChatSession {
  const timestamp = Date.now();
  const id = seed?.id || `session-${timestamp}`;
  return {
    id,
    title: seed?.title || "新会话",
    threadId: seed?.threadId || id,
    items: seed?.items || [
      {
        role: "assistant",
        content: "Falco workspace 已就绪。你可以直接开始对话，也可以先在右侧验证系统与索引状态。",
      },
    ],
    updatedAt: seed?.updatedAt || timestamp,
  };
}

function summarizeSession(items: ChatItem[]) {
  const lastUser = [...items].reverse().find((item) => item.role === "user" && item.content.trim());
  if (!lastUser) {
    return "等待第一条消息";
  }
  return lastUser.content.trim();
}

function buildSessionTitle(message: string) {
  const trimmed = message.trim();
  if (!trimmed) {
    return "新会话";
  }
  return trimmed.slice(0, 18) + (trimmed.length > 18 ? "..." : "");
}

export default function HomePage() {
  const [apiBase, setApiBase] = useState("");
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>(() => [initialSession]);
  const [activeSessionId, setActiveSessionId] = useState("default");

  const [healthStatus, setHealthStatus] = useState<ConnectionState>("idle");
  const [healthMessage, setHealthMessage] = useState("等待检查");
  const [catalogText, setCatalogText] = useState("尚未加载 MCP catalog。");
  const [catalogPending, setCatalogPending] = useState(false);

  const messageViewportRef = useRef<HTMLDivElement | null>(null);
  const hydratedRef = useRef(false);

  const activeSession =
    sessions.find((session) => session.id === activeSessionId) ||
    sessions[0] ||
    initialSession;

  const items = activeSession.items;
  const threadId = activeSession.threadId;
  const canSend = !pending && draft.trim().length > 0;
  const assistantMessageCount = items.filter((item) => item.role === "assistant").length;
  const orderedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions],
  );

  useEffect(() => {
    if (!messageViewportRef.current) {
      return;
    }
    messageViewportRef.current.scrollTop = messageViewportRef.current.scrollHeight;
  }, [items, activeSessionId]);

  useEffect(() => {
    setApiBase(defaultApiBase);
  }, []);

  useEffect(() => {
    try {
      const rawSessions = window.localStorage.getItem(storageKey);
      const rawActive = window.localStorage.getItem(activeSessionKey);
      if (!rawSessions) {
        hydratedRef.current = true;
        return;
      }

      const parsed = JSON.parse(rawSessions) as ChatSession[];
      if (!Array.isArray(parsed) || parsed.length === 0) {
        hydratedRef.current = true;
        return;
      }

      const normalized = parsed.map((session) =>
        createSession({
          ...session,
          items: Array.isArray(session.items) && session.items.length > 0 ? session.items : undefined,
        }),
      );
      setSessions(normalized);
      setActiveSessionId(rawActive && normalized.some((session) => session.id === rawActive) ? rawActive : normalized[0].id);
    } catch {
      // Ignore broken local storage data and fall back to defaults.
    } finally {
      hydratedRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (!hydratedRef.current) {
      return;
    }
    window.localStorage.setItem(storageKey, JSON.stringify(sessions));
    window.localStorage.setItem(activeSessionKey, activeSessionId);
  }, [sessions, activeSessionId]);

  useEffect(() => {
    if (!apiBase.trim()) {
      return;
    }
    void refreshSystemState();
  }, [apiBase]);

  function updateActiveSession(updater: (session: ChatSession) => ChatSession) {
    setSessions((prev) =>
      prev.map((session) => (session.id === activeSession.id ? updater(session) : session)),
    );
  }

  function handleCreateSession() {
    const session = createSession({ title: "新会话" });
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setDraft("");
  }

  function handleDeleteSession(sessionId: string) {
    setSessions((prev) => {
      if (prev.length === 1) {
        const replacement = createSession({ title: "新会话" });
        setActiveSessionId(replacement.id);
        return [replacement];
      }
      const next = prev.filter((session) => session.id !== sessionId);
      if (sessionId === activeSessionId) {
        setActiveSessionId(next[0].id);
      }
      return next;
    });
  }

  async function refreshSystemState() {
    setHealthStatus("checking");
    setHealthMessage("正在连接服务...");
    setCatalogPending(true);

    try {
      const [health, catalog] = await Promise.all([
        fetchHealth({ apiBase }),
        fetchMCPCatalog({ apiBase }),
      ]);
      setHealthStatus("online");
      setHealthMessage(`${health.service} · ${health.status}`);
      setCatalogText(catalog.result || "MCP registry 返回为空。");
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown error";
      setHealthStatus("offline");
      setHealthMessage(`连接失败 · ${text}`);
      setCatalogText("无法读取 MCP catalog。请确认后端服务已经启动。");
    } finally {
      setCatalogPending(false);
    }
  }

  async function handleSend() {
    const message = draft.trim();
    if (!message || pending) {
      return;
    }

    const nextTitle = activeSession.items.length <= 1 ? buildSessionTitle(message) : activeSession.title;

    setDraft("");
    setPending(true);
    updateActiveSession((session) => ({
      ...session,
      title: nextTitle,
      updatedAt: Date.now(),
      items: [...session.items, { role: "user", content: message }, { role: "assistant", content: "" }],
    }));

    try {
      await streamChat({
        apiBase,
        threadId,
        message,
        callbacks: {
          onDelta: (delta) => {
            setSessions((prev) =>
              prev.map((session) => {
                if (session.id !== activeSession.id) {
                  return session;
                }
                const next = [...session.items];
                const last = next[next.length - 1];
                if (!last || last.role !== "assistant") {
                  return session;
                }
                next[next.length - 1] = { role: "assistant", content: `${last.content}${delta}` };
                return { ...session, items: next, updatedAt: Date.now() };
              }),
            );
          },
          onDone: (answer) => {
            setSessions((prev) =>
              prev.map((session) => {
                if (session.id !== activeSession.id) {
                  return session;
                }
                const next = [...session.items];
                const last = next[next.length - 1];
                if (!last || last.role !== "assistant") {
                  return session;
                }
                next[next.length - 1] = { role: "assistant", content: answer || "Falco 没有返回内容。" };
                return { ...session, items: next, updatedAt: Date.now() };
              }),
            );
          },
        },
      });
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown error";
      setSessions((prev) =>
        prev.map((session) => {
          if (session.id !== activeSession.id) {
            return session;
          }
          const next = [...session.items];
          const last = next[next.length - 1];
          if (last?.role === "assistant" && last.content === "") {
            next[next.length - 1] = { role: "assistant", content: `请求失败：${text}` };
          } else {
            next.push({ role: "assistant", content: `请求失败：${text}` });
          }
          return { ...session, items: next, updatedAt: Date.now() };
        }),
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="workspace-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <section className="workspace-grid">
        <aside className="panel panel-left">
          <div className="queue-header">
            <div>
              <p className="eyebrow">Chat Queue</p>
              <h2>会话队列</h2>
            </div>
            <button className="ghost-button" onClick={handleCreateSession} type="button">
              新建会话
            </button>
          </div>

          <div className="queue-summary">
            <article className="stat-card">
              <span>Sessions</span>
              <strong>{String(sessions.length).padStart(2, "0")}</strong>
            </article>
            <article className="stat-card">
              <span>Messages</span>
              <strong>{String(items.length).padStart(2, "0")}</strong>
            </article>
          </div>

          <section className="session-list">
            {orderedSessions.map((session) => {
              const active = session.id === activeSessionId;
              return (
                <article
                  className={`session-card ${active ? "active" : ""}`}
                  key={session.id}
                  onClick={() => setActiveSessionId(session.id)}
                >
                  <div className="session-card-top">
                    <div>
                      <h3>{session.title}</h3>
                      <p>{session.threadId}</p>
                    </div>
                    <button
                      className="session-delete"
                      onClick={(event) => {
                        event.stopPropagation();
                        handleDeleteSession(session.id);
                      }}
                      type="button"
                    >
                      删除
                    </button>
                  </div>
                  <span>{summarizeSession(session.items)}</span>
                </article>
              );
            })}
          </section>
        </aside>

        <section className="panel panel-main">
          <header className="chat-toolbar">
            <div>
              <p className="eyebrow">Conversation</p>
              <h2>Agent Console</h2>
            </div>

            <div className="toolbar-fields">
              <label>
                <span>API Base</span>
                <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} placeholder="/falco-api" />
              </label>
              <label>
                <span>Thread ID</span>
                <input
                  value={threadId}
                  onChange={(event) =>
                    updateActiveSession((session) => ({ ...session, threadId: event.target.value, updatedAt: Date.now() }))
                  }
                  placeholder="default"
                />
              </label>
              <button className="ghost-button" onClick={() => void refreshSystemState()} type="button">
                刷新状态
              </button>
            </div>
          </header>

          <div className="prompt-strip">
            {starterPrompts.map((prompt) => (
              <button className="prompt-chip" key={prompt} onClick={() => setDraft(prompt)} type="button">
                {prompt}
              </button>
            ))}
          </div>

          <section className="messages-panel" ref={messageViewportRef}>
            {items.map((item, index) => (
              <article className={`message-card ${item.role}`} key={`${item.role}-${index}`}>
                <div className="message-meta">
                  <span>{item.role === "user" ? "Operator" : "Falco"}</span>
                </div>
                <div className="message-content">{item.content || (item.role === "assistant" ? "..." : "")}</div>
              </article>
            ))}
          </section>

          <section className="composer-panel">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                  event.preventDefault();
                  void handleSend();
                }
              }}
              placeholder="输入你的任务、问题或调度目标。按 Ctrl/Cmd + Enter 发送。"
            />
            <div className="composer-footer">
              <span>当前线程：{threadId}</span>
              <button disabled={!canSend} onClick={() => void handleSend()} type="button">
                {pending ? "处理中..." : "发送到 Falco"}
              </button>
            </div>
          </section>
        </section>

        <aside className="panel panel-right">
          <section className="status-card">
            <div className="status-heading">
              <div>
                <p className="eyebrow">System</p>
                <h2>Service Status</h2>
              </div>
              <span className={`status-pill ${healthStatus}`}>{healthStatus}</span>
            </div>
            <p className="status-line">{healthMessage}</p>
          </section>

          <section className="tool-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Registry</p>
                <h2>MCP Catalog</h2>
              </div>
              <button className="ghost-button" onClick={() => void refreshSystemState()} type="button">
                {catalogPending ? "加载中..." : "重新加载"}
              </button>
            </div>
            <pre className="result-block catalog-block">{catalogText}</pre>
          </section>
        </aside>
      </section>
    </main>
  );
}
