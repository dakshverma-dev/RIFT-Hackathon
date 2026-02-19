import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";

export default function GraphView({ result }) {
    const containerRef = useRef(null);
    const cyRef = useRef(null);
    const [selected, setSelected] = useState(null);

    useEffect(() => {
        if (!result || !containerRef.current) return;

        const suspiciousMap = {};
        const patternMap = {};
        result.suspicious_accounts.forEach((a) => {
            suspiciousMap[a.account_id] = a.suspicion_score;
            patternMap[a.account_id] = a.detected_patterns;
        });

        const ringMemberSet = new Set(result.fraud_rings.flatMap((r) => r.member_accounts));
        const suspiciousSet = new Set(Object.keys(suspiciousMap));

        const nodeSet = new Set();
        const edges = [];

        result.fraud_rings.forEach((ring) => {
            const m = ring.member_accounts;
            m.forEach((id) => nodeSet.add(id));
            if (ring.pattern_type.startsWith("cycle")) {
                for (let i = 0; i < m.length; i++)
                    edges.push({ s: m[i], t: m[(i + 1) % m.length] });
            } else {
                for (let i = 1; i < m.length; i++)
                    edges.push({ s: m[i], t: m[0] });
            }
        });
        result.suspicious_accounts.forEach((a) => nodeSet.add(a.account_id));

        const totalNodes = nodeSet.size;
        const hideLabels = totalNodes > 300;

        const elements = [];
        nodeSet.forEach((id) => {
            const isSusp = suspiciousSet.has(id);
            const isRing = ringMemberSet.has(id);
            const score = suspiciousMap[id] || 0;
            const type = isSusp && isRing ? "ring" : isSusp ? "suspicious" : "normal";
            elements.push({ data: { id, label: id, score, type, patterns: patternMap[id] || [] } });
        });

        const edgeIds = new Set();
        edges.forEach((e) => {
            const eid = `${e.s}->${e.t}`;
            if (!edgeIds.has(eid)) {
                edgeIds.add(eid);
                const isFlagged = suspiciousSet.has(e.s) && suspiciousSet.has(e.t);
                elements.push({ data: { id: eid, source: e.s, target: e.t, flagged: isFlagged } });
            }
        });

        if (cyRef.current) cyRef.current.destroy();

        cyRef.current = cytoscape({
            container: containerRef.current,
            elements,
            style: [
                {
                    selector: "node",
                    style: {
                        label: hideLabels ? "" : "data(label)",
                        "text-valign": "bottom",
                        "text-margin-y": 6,
                        "font-size": 9,
                        "font-family": "'SF Mono', 'Fira Code', monospace",
                        color: "#9CA3AF",
                        "background-color": "#E5E7EB",
                        width: 20,
                        height: 20,
                        "border-width": 1.5,
                        "border-color": "#D1D5DB",
                    },
                },
                {
                    selector: 'node[type="suspicious"]',
                    style: {
                        "background-color": "rgba(230,57,70,0.12)",
                        "border-color": "#E63946",
                        "border-width": 2,
                        width: "mapData(score, 0, 100, 24, 40)",
                        height: "mapData(score, 0, 100, 24, 40)",
                    },
                },
                {
                    selector: 'node[type="ring"]',
                    style: {
                        "background-color": "rgba(29,53,87,0.12)",
                        "border-color": "#1D3557",
                        "border-width": 2,
                        width: 24,
                        height: 24,
                    },
                },
                {
                    selector: "edge",
                    style: {
                        width: 1,
                        "line-color": "#D1D5DB",
                        "target-arrow-color": "#D1D5DB",
                        "target-arrow-shape": "triangle",
                        "arrow-scale": 0.7,
                        "curve-style": "bezier",
                    },
                },
                {
                    selector: "edge[flagged]",
                    style: {
                        "line-color": "rgba(230,57,70,0.35)",
                        "target-arrow-color": "rgba(230,57,70,0.35)",
                        width: 1.5,
                    },
                },
            ],
            layout: {
                name: "cose",
                animate: true,
                animationDuration: 600,
                nodeRepulsion: () => 6000,
                idealEdgeLength: () => 100,
                nodeDimensionsIncludeLabels: true,
            },
            minZoom: 0.3,
            maxZoom: 3,
        });

        cyRef.current.on("tap", "node", (e) => {
            const d = e.target.data();
            setSelected(d);
        });
        cyRef.current.on("tap", (e) => {
            if (e.target === cyRef.current) setSelected(null);
        });

        return () => { if (cyRef.current) cyRef.current.destroy(); };
    }, [result]);

    const ringId = selected
        ? result?.fraud_rings.find((r) => r.member_accounts.includes(selected.id))?.ring_id
        : null;

    const scoreColor = (s) =>
        s >= 70 ? "var(--accent)" : s >= 40 ? "var(--amber)" : "var(--green)";

    return (
        <div className="graph-section">
            <div className="graph-card">
                <div className="graph-header">
                    <span className="graph-title">Transaction Network</span>
                    <div className="graph-legend">
                        <span className="legend-item">
                            <span className="legend-dot" style={{ background: "#D1D5DB" }} /> Normal
                        </span>
                        <span className="legend-item">
                            <span className="legend-dot" style={{ background: "var(--accent)" }} /> Suspicious
                        </span>
                        <span className="legend-item">
                            <span className="legend-dot" style={{ background: "var(--accent2)" }} /> Fraud Ring
                        </span>
                    </div>
                </div>
                <div className="graph-area">
                    {!result ? (
                        <div className="graph-empty">Upload a CSV to render the network graph</div>
                    ) : (
                        <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
                    )}
                    {selected && (
                        <div className="node-panel">
                            <button className="node-panel-close" onClick={() => setSelected(null)}>
                                &times;
                            </button>
                            <div className="node-panel-id">{selected.id}</div>
                            <div className="node-panel-row">
                                <span className="node-panel-label">Suspicion Score</span>
                                <span className="node-panel-score" style={{ color: scoreColor(selected.score) }}>
                                    {selected.score}
                                </span>
                            </div>
                            <div className="node-panel-patterns">
                                <p>Detected Patterns</p>
                                {selected.patterns.length > 0 ? (
                                    selected.patterns.map((p) => (
                                        <div key={p} className="pattern-item">&middot; {p}</div>
                                    ))
                                ) : (
                                    <div className="pattern-item" style={{ color: "var(--muted)" }}>None detected</div>
                                )}
                            </div>
                            {ringId && <div className="node-panel-ring">{ringId}</div>}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
