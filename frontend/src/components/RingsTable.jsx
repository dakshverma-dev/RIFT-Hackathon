export default function RingsTable({ rings }) {
    if (!rings || rings.length === 0) return null;

    const sorted = [...rings].sort((a, b) => b.risk_score - a.risk_score);

    function riskClass(score) {
        if (score > 80) return "risk-high";
        if (score >= 50) return "risk-med";
        return "risk-low";
    }

    function patternStyle(type) {
        if (type.startsWith("cycle")) return "cycle";
        if (type.includes("fan")) return "fan";
        return "shell";
    }

    function accountsText(accounts) {
        const str = accounts.join(", ");
        return str.length > 60 ? str.slice(0, 57) + "..." : str;
    }

    return (
        <div className="table-section">
            <div className="table-section-label">Fraud Rings</div>
            <div className="table-card">
                <div className="table-header">
                    <span className="table-header-text">
                        {sorted.length} ring{sorted.length !== 1 ? "s" : ""} detected
                    </span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Ring ID</th>
                            <th>Pattern</th>
                            <th>Members</th>
                            <th>Risk Score</th>
                            <th>Accounts</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((ring) => (
                            <tr key={ring.ring_id}>
                                <td className="mono muted" style={{ fontSize: 13 }}>{ring.ring_id}</td>
                                <td>
                                    <span className={`pattern-indicator ${patternStyle(ring.pattern_type)}`}>
                                        {ring.pattern_type}
                                    </span>
                                </td>
                                <td>{ring.member_accounts.length}</td>
                                <td>
                                    <span className={riskClass(ring.risk_score)}>{ring.risk_score}</span>
                                </td>
                                <td>
                                    <span
                                        className="muted truncate mono"
                                        style={{ fontSize: 12 }}
                                        title={ring.member_accounts.join(", ")}
                                    >
                                        {accountsText(ring.member_accounts)}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
