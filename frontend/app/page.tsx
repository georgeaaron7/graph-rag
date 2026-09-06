"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import {
  Citation,
  checkHealth,
  ingestPdf,
  retrieverColor,
  retrieverLabel,
  streamChat,
  API_BASE,
} from "./lib/api";
import ProvenanceGraph from "./components/ProvenanceGraph";

type Role = "user" | "assistant";
interface Msg {
  role: Role;
  text: string;
  citations: Citation[];
  streaming?: boolean;
  error?: string;
}

interface Corpus {
  docs: string[];
  chunks: number;
  entities: number;
  relations: number;
}

const EXAMPLES = [
  "How are the two main approaches in this document related?",
  "What did the person who introduced X go on to do?",
  "Summarize the trade-offs discussed, with sources.",
];

export default function Page() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const [corpus, setCorpus] = useState<Corpus>({ docs: [], chunks: 0, entities: 0, relations: 0 });
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const [apiUp, setApiUp] = useState<boolean | null>(null);

  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    checkHealth().then(setApiUp);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming]);

  const updateLast = useCallback((patch: (m: Msg) => Msg) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const copy = prev.slice();
      copy[copy.length - 1] = patch(copy[copy.length - 1]);
      return copy;
    });
  }, []);

  const send = useCallback(
    async (raw: string) => {
      const q = raw.trim();
      if (!q || streaming) return;
      setInput("");
      setActiveIndex(null);
      setMessages((prev) => [
        ...prev,
        { role: "user", text: q, citations: [] },
        { role: "assistant", text: "", citations: [], streaming: true },
      ]);
      setStreaming(true);

      await streamChat(q, {
        onCitations: (c) => updateLast((m) => ({ ...m, citations: c })),
        onToken: (t) => updateLast((m) => ({ ...m, text: m.text + t })),
        onError: (e) =>
          updateLast((m) => ({ ...m, error: e.message, streaming: false })),
        onDone: () => updateLast((m) => ({ ...m, streaming: false })),
      });
      setStreaming(false);
      if (apiUp === false) setApiUp(true);
      taRef.current?.focus();
    },
    [streaming, updateLast, apiUp]
  );

  const doIngest = useCallback(async (file: File) => {
    setIngestError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setIngestError("Only PDF files are supported.");
      return;
    }
    setIngesting(true);
    try {
      const res = await ingestPdf(file);
      setCorpus((c) => ({
        docs: [...c.docs, res.source],
        chunks: c.chunks + res.chunks,
        entities: c.entities + res.entities,
        relations: c.relations + res.relations,
      }));
      setApiUp(true);
    } catch (e) {
      setIngestError((e as Error).message);
    } finally {
      setIngesting(false);
    }
  }, []);

  // The aside reflects the most recent answer that produced citations.
  const lastCited = [...messages].reverse().find((m) => m.role === "assistant" && m.citations.length > 0);
  const citations = lastCited?.citations ?? [];

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };
  const autosize = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  return (
    <div className="app">
      {/* -------- Left rail -------- */}
      <aside className="panel rail">
        <div className="brand">
          <BrandMark />
          <div>
            <div className="brand-name">GraphRAG</div>
            <div className="brand-sub">Knowledge Assistant</div>
          </div>
        </div>

        <div>
          <p className="eyebrow">Corpus</p>
          <label
            className={`drop ${dragging ? "drag" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) doIngest(f);
            }}
          >
            {ingesting ? (
              <div className="drop-busy">Building the graph…</div>
            ) : (
              <>
                <div className="drop-title">Add a PDF</div>
                <div className="drop-hint">Drop a file, or click to browse</div>
              </>
            )}
            <input
              type="file"
              accept="application/pdf,.pdf"
              disabled={ingesting}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) doIngest(f);
                e.currentTarget.value = "";
              }}
            />
          </label>

          {ingestError && (
            <div className="msg-error" style={{ marginTop: 10, fontSize: 12 }}>
              {ingestError}
            </div>
          )}

          <div className="corpus">
            <div className="stat">
              <div className="stat-num">{corpus.chunks}</div>
              <div className="stat-label">Chunks</div>
            </div>
            <div className="stat">
              <div className="stat-num">{corpus.entities}</div>
              <div className="stat-label">Entities</div>
            </div>
            <div className="stat">
              <div className="stat-num">{corpus.relations}</div>
              <div className="stat-label">Relations</div>
            </div>
          </div>

          {corpus.docs.length > 0 && (
            <ul className="doclist">
              {corpus.docs.map((d, i) => (
                <li key={i} title={d}>
                  {d}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <p className="eyebrow">How an answer is found</p>
          <div className="how">
            <div className="how-step">
              <span className="dot" style={{ background: "var(--vector)", color: "var(--vector)" }} />
              <div>
                <strong>Vector arm</strong>
                <p>Embeds your question and finds semantically similar chunks.</p>
              </div>
            </div>
            <div className="how-step">
              <span className="dot" style={{ background: "var(--graph)", color: "var(--graph)" }} />
              <div>
                <strong>Graph arm</strong>
                <p>Walks entity relationships in Neo4j to reach connected facts.</p>
              </div>
            </div>
            <div className="how-step">
              <span className="dot" style={{ background: "var(--hybrid)", color: "var(--hybrid)" }} />
              <div>
                <strong>Fusion</strong>
                <p>Reciprocal Rank Fusion merges both; chunks found by both rank highest.</p>
              </div>
            </div>
          </div>
        </div>

        <div className={`status ${apiUp === true ? "up" : apiUp === false ? "down" : ""}`}>
          <span className="pulse" />
          {apiUp === true ? "API connected" : apiUp === false ? "API offline" : "Checking API…"}
          <span className="mono" style={{ marginLeft: "auto", fontSize: 10.5 }}>
            {API_BASE.replace(/^https?:\/\//, "")}
          </span>
        </div>
      </aside>

      {/* -------- Center: conversation -------- */}
      <main className="panel center">
        {apiUp === false && (
          <div className="banner">
            Can&apos;t reach the API at <code>{API_BASE}</code>. Start the backend with{" "}
            <code>uvicorn app.main:app --reload</code>, then reload.
          </div>
        )}

        <div className="thread">
          <div className="thread-inner">
            {messages.length === 0 ? (
              <div className="empty">
                <BrandMark large />
                <h1>Ask your documents anything</h1>
                <p>
                  Every answer is grounded in a knowledge graph over your PDFs and cited back to the
                  exact page. Add a document, then try a question.
                </p>
                <div className="examples">
                  {EXAMPLES.map((ex, i) => (
                    <button key={i} className="example" onClick={() => send(ex)} disabled={streaming}>
                      <span>{String(i + 1).padStart(2, "0")}</span>
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i} className="msg-user">
                    {m.text}
                  </div>
                ) : (
                  <div key={i} className="msg-assistant">
                    <div className="assistant-head">
                      <span className="assistant-glyph" />
                      <span className="assistant-who">Assistant</span>
                    </div>
                    {m.error ? (
                      <div className="msg-error">{m.error}</div>
                    ) : (
                      <div className="answer">
                        <InlineAnswer
                          text={m.text}
                          citations={m.citations}
                          activeIndex={activeIndex}
                          onSelect={setActiveIndex}
                        />
                        {m.streaming && <span className="caret" />}
                      </div>
                    )}
                  </div>
                )
              )
            )}
            <div ref={endRef} />
          </div>
        </div>

        <div className="composer-wrap">
          <div className="composer">
            <textarea
              ref={taRef}
              value={input}
              placeholder={corpus.chunks === 0 ? "Add a PDF first, then ask…" : "Ask about your documents…"}
              rows={1}
              onChange={(e) => {
                setInput(e.target.value);
                autosize(e.target);
              }}
              onKeyDown={onKeyDown}
              disabled={streaming}
            />
            <button
              className="send"
              onClick={() => send(input)}
              disabled={streaming || !input.trim()}
              aria-label="Send question"
            >
              <SendIcon />
            </button>
          </div>
          <p className="composer-hint">
            Enter to send · Shift+Enter for a new line · answers cite the source page
          </p>
        </div>
      </main>

      {/* -------- Right: provenance -------- */}
      <aside className="panel aside">
        <div className="aside-inner">
          <p className="eyebrow">Provenance</p>
          <ProvenanceGraph citations={citations} activeIndex={activeIndex} onSelect={setActiveIndex} />

          {citations.length > 0 && (
            <>
              <div className="legend">
                <LegendRow color="var(--vector)" name="Vector" note="semantic match" />
                <LegendRow color="var(--graph)" name="Graph" note="relationship hop" />
                <LegendRow color="var(--hybrid)" name="Hybrid" note="found by both" />
              </div>
              <p className="eyebrow" style={{ marginTop: 8 }}>
                Sources
              </p>
              <div className="sources">
                {citations.map((c) => (
                  <SourceCard
                    key={c.index}
                    c={c}
                    active={c.index === activeIndex}
                    onClick={() => setActiveIndex(c.index)}
                  />
                ))}
              </div>
            </>
          )}
          {citations.length === 0 && (
            <p className="aside-empty">No sources yet — ask a question to see where the answer comes from.</p>
          )}
        </div>
      </aside>
    </div>
  );
}

/* ---------- Sub-components ---------- */

function InlineAnswer({
  text,
  citations,
  activeIndex,
  onSelect,
}: {
  text: string;
  citations: Citation[];
  activeIndex: number | null;
  onSelect: (i: number) => void;
}) {
  const byIndex = new Map(citations.map((c) => [c.index, c]));
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (m) {
          const idx = Number(m[1]);
          const cit = byIndex.get(idx);
          const col = cit ? retrieverColor(cit.retriever) : "var(--accent)";
          return (
            <button
              key={i}
              className={`cite ${idx === activeIndex ? "cite-active" : ""}`}
              style={{ color: col, ["--cite-color" as string]: col } as CSSProperties}
              onClick={() => onSelect(idx)}
              title={cit ? `${cit.source ?? "source"} · ${retrieverLabel(cit.retriever)}` : `Source ${idx}`}
            >
              {idx}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function SourceCard({ c, active, onClick }: { c: Citation; active: boolean; onClick: () => void }) {
  const col = retrieverColor(c.retriever);
  return (
    <button
      className={`source ${active ? "source-active" : ""}`}
      style={{ borderLeftColor: col, textAlign: "left" }}
      onClick={onClick}
    >
      <div className="source-top">
        <span className="source-idx" style={{ background: col }}>
          {c.index}
        </span>
        <span className="source-meta">
          {(c.source ?? "document") + (c.page != null ? ` · p.${c.page}` : "")}
        </span>
        <span className="tag" style={{ background: col }}>
          {retrieverLabel(c.retriever)}
        </span>
      </div>
      {c.snippet && (
        <div 
          className="source-snippet" 
          style={active ? { WebkitLineClamp: "unset", display: "block" } : undefined}
        >
          {c.snippet}
        </div>
      )}
    </button>
  );
}

function LegendRow({ color, name, note }: { color: string; name: string; note: string }) {
  return (
    <div className="legend-row">
      <span className="dot" style={{ background: color, color }} />
      <span>
        <b>{name}</b> · {note}
      </span>
    </div>
  );
}

function BrandMark({ large }: { large?: boolean }) {
  const s = large ? 46 : 34;
  return (
    <svg
      className="brand-mark"
      width={s}
      height={s}
      viewBox="0 0 34 34"
      fill="none"
      aria-hidden="true"
      style={{ width: s, height: s }}
    >
      <line x1="9" y1="9" x2="25" y2="12" stroke="#0d9488" strokeWidth="1.6" />
      <line x1="9" y1="9" x2="12" y2="25" stroke="#c2820a" strokeWidth="1.6" />
      <line x1="25" y1="12" x2="12" y2="25" stroke="#4f46e5" strokeWidth="1.6" />
      <circle cx="9" cy="9" r="4.2" fill="#0d9488" />
      <circle cx="25" cy="12" r="4.2" fill="#c2820a" />
      <circle cx="12" cy="25" r="4.2" fill="#4f46e5" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
