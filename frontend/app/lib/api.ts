// Typed client for the GraphRAG backend.
// Endpoints: GET /health, POST /ingest (multipart), POST /chat, POST /chat/stream (SSE).

export type Retriever = "hybrid" | "vector" | "graph";

export interface Citation {
  index: number;
  chunk_id: string;
  source?: string | null;
  page?: number | null;
  retriever: Retriever;
  snippet: string;
}

export interface IngestResult {
  source: string;
  chunks: number;
  entities: number;
  relations: number;
}

export const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function detail(res: Response): Promise<string> {
  try {
    const j = await res.json();
    return typeof j?.detail === "string" ? j.detail : "";
  } catch {
    return "";
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

export async function ingestPdf(file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${API_BASE}/ingest`, { method: "POST", body: form });
  if (!r.ok) {
    throw new Error((await detail(r)) || `Ingest failed (${r.status})`);
  }
  return r.json();
}

export interface StreamHandlers {
  onCitations?: (c: Citation[]) => void;
  onToken?: (t: string) => void;
  onDone?: () => void;
  onError?: (e: Error) => void;
  signal?: AbortSignal;
}

// Parse one SSE frame ("event: X\ndata: Y") and dispatch it.
function dispatch(frame: string, h: StreamHandlers) {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  const payload = data.join("\n");
  if (!payload) return;
  try {
    if (event === "citations") h.onCitations?.(JSON.parse(payload) as Citation[]);
    else if (event === "token") h.onToken?.(String(JSON.parse(payload).t ?? ""));
  } catch {
    /* ignore a malformed frame rather than killing the stream */
  }
}

// POST /chat/stream and read the SSE body incrementally.
// (EventSource only speaks GET, so we stream the fetch body by hand.)
export async function streamChat(query: string, h: StreamHandlers): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: h.signal,
    });
  } catch {
    h.onError?.(new Error(`Can't reach the API at ${API_BASE}. Is the backend running?`));
    return;
  }
  if (!res.ok || !res.body) {
    h.onError?.(new Error((await detail(res)) || `Request failed (${res.status})`));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        dispatch(buffer.slice(0, sep), h);
        buffer = buffer.slice(sep + 2);
      }
    }
  } catch (e) {
    if ((e as Error).name !== "AbortError") h.onError?.(e as Error);
    return;
  }
  h.onDone?.();
}

export function retrieverColor(r: Retriever): string {
  if (r === "vector") return "var(--vector)";
  if (r === "graph") return "var(--graph)";
  return "var(--hybrid)";
}

export function retrieverLabel(r: Retriever): string {
  if (r === "vector") return "vector";
  if (r === "graph") return "graph";
  return "hybrid";
}
