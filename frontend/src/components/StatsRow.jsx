import { useEffect, useState, useRef } from "react";

function AnimatedValue({ target, duration = 1200, suffix = "", color = "var(--text)" }) {
    const [value, setValue] = useState(0);
    const rafRef = useRef(null);

    useEffect(() => {
        const num = typeof target === "number" ? target : parseFloat(target) || 0;
        const start = performance.now();

        function tick(now) {
            const t = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - t, 3);
            setValue(eased * num);
            if (t < 1) rafRef.current = requestAnimationFrame(tick);
        }

        rafRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafRef.current);
    }, [target, duration]);

    const display = Number.isInteger(target) ? Math.round(value) : value.toFixed(1);

    return (
        <div className="stat-value" style={{ color }}>
            {display}{suffix}
        </div>
    );
}

export default function StatsRow({ summary }) {
    if (!summary) return null;

    const total = summary.total_accounts_analyzed;
    const suspicious = summary.suspicious_accounts_flagged;
    const pct = total > 0 ? ((suspicious / total) * 100).toFixed(1) : "0";

    return (
        <div className="stats-row" id="stats">
            <div className="stat-card">
                <div className="stat-label">Accounts Analyzed</div>
                <AnimatedValue target={total} color="var(--text)" />
                <div className="stat-sub">unique nodes in graph</div>
            </div>
            <div className="stat-card">
                <div className="stat-label">Suspicious Flagged</div>
                <AnimatedValue target={suspicious} color="var(--accent)" />
                <div className="stat-sub">{pct}% of total</div>
            </div>
            <div className="stat-card">
                <div className="stat-label">Fraud Rings</div>
                <AnimatedValue target={summary.fraud_rings_detected} color="var(--accent2)" />
                <div className="stat-sub">cycles + smurfing</div>
            </div>
            <div className="stat-card">
                <div className="stat-label">Processing Time</div>
                <AnimatedValue target={summary.processing_time_seconds} suffix="s" color="var(--text)" />
                <div className="stat-sub">server-side analysis</div>
            </div>
        </div>
    );
}
