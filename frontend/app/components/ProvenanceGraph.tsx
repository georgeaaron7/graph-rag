"use client";

import { Citation, retrieverColor } from "../lib/api";

/**
 * The signature element: a radial "provenance graph". A central query hub is
 * connected by an edge to every cited source, and each edge/node is colored by
 * the retrieval arm that surfaced it (teal = vector, amber = graph, indigo =
 * both). This makes the whole point of the project — graph-augmented, traceable
 * retrieval — visible at a glance.
 */
export default function ProvenanceGraph({
  citations,
  activeIndex,
  onSelect,
}: {
  citations: Citation[];
  activeIndex: number | null;
  onSelect: (index: number) => void;
}) {
  const SIZE = 320;
  const c = SIZE / 2;
  const hubR = 22;
  const n = citations.length;
  const nodeR = n > 8 ? 15 : 18;
  const ring = 112;

  const nodes = citations.map((cit, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / Math.max(1, n);
    return {
      cit,
      x: c + ring * Math.cos(angle),
      y: c + ring * Math.sin(angle),
      color: retrieverColor(cit.retriever),
    };
  });

  if (n === 0) {
    return (
      <div className="graph-empty" aria-hidden="true">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width="100%" role="img">
          <circle cx={c} cy={c} r={ring} className="graph-empty-ring" />
          <circle cx={c} cy={c} r={hubR} className="graph-empty-hub" />
        </svg>
        <p className="graph-empty-note">
          Sources appear here, wired to your question — one node per cited chunk.
        </p>
      </div>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      width="100%"
      className="graph"
      role="group"
      aria-label="Provenance graph of cited sources"
    >
      {/* edges: colored via style so CSS var() resolves (presentation attrs don't) */}
      {nodes.map((nd, i) => (
        <line
          key={`e${i}`}
          x1={c}
          y1={c}
          x2={nd.x}
          y2={nd.y}
          className="edge"
          style={{ stroke: nd.color, animationDelay: `${i * 70}ms` }}
        />
      ))}

      <circle cx={c} cy={c} r={hubR} className="hub" />
      <text x={c} y={c} className="hub-label" dominantBaseline="central" textAnchor="middle">
        Q
      </text>

      {/* outer <g> holds the position; inner <g> runs the pop animation so the
          animated transform never clobbers the translate */}
      {nodes.map((nd, i) => {
        const active = nd.cit.index === activeIndex;
        return (
          <g key={`n${i}`} transform={`translate(${nd.x} ${nd.y})`}>
            <g
              className="node"
              style={{ animationDelay: `${i * 70 + 120}ms` }}
              role="button"
              tabIndex={0}
              aria-label={`Source ${nd.cit.index}: ${nd.cit.source ?? "document"}${
                nd.cit.page != null ? `, page ${nd.cit.page}` : ""
              } (${nd.cit.retriever})`}
              onClick={() => onSelect(nd.cit.index)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(nd.cit.index);
                }
              }}
            >
              <circle
                r={nodeR}
                className="node-circle"
                style={{ stroke: nd.color, fill: active ? nd.color : "var(--paper)" }}
              />
              <text
                className="node-label"
                dominantBaseline="central"
                textAnchor="middle"
                style={{ fill: active ? "var(--paper)" : nd.color }}
              >
                {nd.cit.index}
              </text>
              <title>
                {(nd.cit.source ?? "document") +
                  (nd.cit.page != null ? ` · p.${nd.cit.page}` : "") +
                  ` · ${nd.cit.retriever}`}
              </title>
            </g>
          </g>
        );
      })}
    </svg>
  );
}
