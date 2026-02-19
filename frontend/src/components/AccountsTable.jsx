export default function AccountsTable({ accounts }) {
    if (!accounts || accounts.length === 0) return null;

    const sorted = [...accounts].sort((a, b) => b.suspicion_score - a.suspicion_score);

    function scoreClass(score) {
        if (score >= 70) return "risk-high";
        if (score >= 40) return "risk-med";
        return "risk-low";
    }

    return (
        <div className="table-section">
            <div className="table-section-label">Suspicious Accounts</div>
            <div className="table-card">
                <div className="table-header">
                    <span className="table-header-text">
                        {sorted.length} account{sorted.length !== 1 ? "s" : ""} flagged
                    </span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Account ID</th>
                            <th>Score</th>
                            <th>Patterns</th>
                            <th>Ring</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((acct) => (
                            <tr key={acct.account_id}>
                                <td className="mono" style={{ fontSize: 13 }}>{acct.account_id}</td>
                                <td>
                                    <span className={scoreClass(acct.suspicion_score)}>
                                        {acct.suspicion_score}
                                    </span>
                                </td>
                                <td style={{ color: "var(--muted)", fontSize: 13 }}>
                                    {acct.detected_patterns.join(", ")}
                                </td>
                                <td className="mono muted" style={{ fontSize: 11 }}>
                                    {acct.ring_id || "—"}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
