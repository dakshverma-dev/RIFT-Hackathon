"""
Money Muling Detection Backend
==============================
A Flask application that accepts CSV transaction data, builds a directed
transaction graph with NetworkX, and runs three fraud-detection algorithms:
  1. Cycle Detection (DFS, length 3-5)
  2. Smurfing / Fan-in & Fan-out
  3. Shell Network Detection

Each flagged account receives a suspicion score (0-100) and is grouped into
fraud rings where applicable.
"""

import io
import time
import json
from datetime import timedelta

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# In-memory store for the last analysis result
_last_result = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "FinForensics API is running 🚀",
        "endpoints": ["/ping", "/upload", "/download-json"]
    })


# ===========================================================================
# 1. GRAPH CONSTRUCTION
# ===========================================================================

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Construct a directed graph from the transaction DataFrame.

    Each unique account (sender or receiver) becomes a node.
    Each transaction becomes a directed edge carrying amount and timestamp.

    Time complexity: O(E) where E = number of transaction rows.
    Uses vectorized edge construction instead of iterrows for speed.
    """
    G = nx.DiGraph()
    edges = list(zip(df["sender_id"], df["receiver_id"]))
    G.add_edges_from(edges)
    return G


# ===========================================================================
# 2. CYCLE DETECTION  (length 3-5, DFS-based)
# ===========================================================================

def detect_cycles(G: nx.DiGraph) -> list[list[str]]:
    """
    Find all simple cycles of length 3, 4, or 5 using DFS.

    NetworkX's simple_cycles uses Johnson's algorithm internally.
    We pass length_bound=5 so the algorithm ONLY enumerates cycles up to
    length 5, avoiding the combinatorial explosion of longer cycles.

    Time complexity: O((V + E) * (C + 1))  where C = number of elementary
    circuits of length ≤ 5.  The length_bound makes this tractable even for
    graphs with thousands of nodes.

    Returns a list of cycles, each cycle being a list of account IDs.
    """
    cycles = []
    for cycle in nx.simple_cycles(G, length_bound=5):
        if len(cycle) >= 3:
            cycles.append(cycle)
    return cycles


# ===========================================================================
# 3. SMURFING DETECTION  (Fan-in / Fan-out within 72 hours)
# ===========================================================================

def detect_smurfing(df: pd.DataFrame) -> dict:
    """
    Detect fan-in and fan-out patterns using Pandas timestamp windowing.

    Fan-in:  ≥10 distinct senders  → 1 receiver within any 72-hour window.
    Fan-out: 1 sender → ≥10 distinct receivers within any 72-hour window.

    Approach:
      • Sort by timestamp.
      • For each receiver (fan-in) or sender (fan-out), use a 72-hour
        rolling window and count distinct counterparties.

    Time complexity: O(E * log E) for sorting + O(E) for the groupby scans,
    so overall O(E log E) where E = number of transactions.

    Returns {"fan_in_nodes": set, "fan_out_nodes": set}.
    """
    WINDOW = timedelta(hours=72)
    fan_in_nodes: set[str] = set()
    fan_out_nodes: set[str] = set()

    df_sorted = df.sort_values("timestamp")

    # --- Fan-in: many senders → one receiver ---
    for receiver, group in df_sorted.groupby("receiver_id"):
        timestamps = group["timestamp"].values
        senders = group["sender_id"].values
        n = len(group)
        left = 0
        for right in range(n):
            while pd.Timestamp(timestamps[right]) - pd.Timestamp(timestamps[left]) > WINDOW:
                left += 1
            unique_senders = set(senders[left : right + 1])
            if len(unique_senders) >= 10:
                fan_in_nodes.add(receiver)
                break  # already flagged

    # --- Fan-out: one sender → many receivers ---
    for sender, group in df_sorted.groupby("sender_id"):
        timestamps = group["timestamp"].values
        receivers = group["receiver_id"].values
        n = len(group)
        left = 0
        for right in range(n):
            while pd.Timestamp(timestamps[right]) - pd.Timestamp(timestamps[left]) > WINDOW:
                left += 1
            unique_receivers = set(receivers[left : right + 1])
            if len(unique_receivers) >= 10:
                fan_out_nodes.add(sender)
                break

    return {"fan_in_nodes": fan_in_nodes, "fan_out_nodes": fan_out_nodes}


# ===========================================================================
# 4. SHELL NETWORK DETECTION  (chains ≥3 hops, low-activity intermediates)
# ===========================================================================

def detect_shell_networks(G: nx.DiGraph) -> set[str]:
    """
    Identify shell-network nodes: accounts that appear as intermediate hops
    in chains of length ≥3, where each intermediate node has a total degree
    (in + out) of only 2 or 3 transactions.

    Algorithm (optimised DFS chain-walk):
      1. Identify candidate "shell" nodes (degree 2 or 3).
      2. Build subgraph of shell candidates only.
      3. From each candidate with in-degree 0 in the subgraph (chain start),
         do a single DFS forward walk (max depth 6).  Any path ≥ 3 nodes
         flags all its members.
      This avoids the O(N²) all-pairs enumeration of the previous version.

    Time complexity: O(V + E) for degree filtering + O(S * D) where
    S = number of chain-start nodes and D = max depth (6).

    Returns a set of account IDs flagged as shell intermediaries.
    """
    shell_candidates = set()
    for node in G.nodes():
        total_degree = G.in_degree(node) + G.out_degree(node)
        if 2 <= total_degree <= 3:
            shell_candidates.add(node)

    if not shell_candidates:
        return set()

    sub = G.subgraph(shell_candidates)
    shell_nodes: set[str] = set()

    # DFS chain walk from each candidate (only start from sources or
    # nodes whose predecessors are NOT in the subgraph, to avoid
    # redundant walks).  Also walk from every node to catch cycles.
    starts = [n for n in sub.nodes() if sub.in_degree(n) == 0] or list(sub.nodes())

    def _dfs(node, path, visited):
        if len(path) >= 3:
            shell_nodes.update(path)
        if len(path) >= 6:
            return  # depth cap
        for nxt in sub.successors(node):
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                _dfs(nxt, path, visited)
                path.pop()
                visited.discard(nxt)

    for start in starts:
        _dfs(start, [start], {start})

    return shell_nodes


# ===========================================================================
# 5. HIGH VELOCITY CHECK
# ===========================================================================

def detect_high_velocity(df: pd.DataFrame) -> set[str]:
    """
    Flag accounts with >10 transactions (sent or received) within any
    24-hour window.

    Time complexity: O(E log E) for sorting + O(E) for the sliding-window
    scan per account.

    Returns a set of high-velocity account IDs.
    """
    WINDOW = timedelta(hours=24)
    high_vel: set[str] = set()
    df_sorted = df.sort_values("timestamp")

    # Check both sender and receiver activity
    for role in ["sender_id", "receiver_id"]:
        for account, group in df_sorted.groupby(role):
            timestamps = group["timestamp"].values
            n = len(group)
            left = 0
            for right in range(n):
                while pd.Timestamp(timestamps[right]) - pd.Timestamp(timestamps[left]) > WINDOW:
                    left += 1
                if (right - left + 1) > 10:
                    high_vel.add(account)
                    break

    return high_vel


# ===========================================================================
# 6. FALSE POSITIVE PROTECTION — Merchant Filter
# ===========================================================================

def get_merchant_accounts(df: pd.DataFrame) -> set[str]:
    """
    Accounts with >50 total transactions (as sender or receiver) are
    considered legitimate merchants and excluded from flagging.

    Time complexity: O(E) to count transactions per account.
    Uses vectorized value_counts instead of iterrows for speed.
    """
    send_counts = df["sender_id"].value_counts()
    recv_counts = df["receiver_id"].value_counts()
    total = send_counts.add(recv_counts, fill_value=0)
    return set(total[total > 50].index)


# ===========================================================================
# 7. SUSPICION SCORING  (0-100, additive, capped)
# ===========================================================================

def compute_scores(
    all_accounts: set[str],
    cycle_accounts: set[str],
    fan_in_out_accounts: set[str],
    shell_accounts: set[str],
    high_vel_accounts: set[str],
    merchants: set[str],
) -> dict[str, float]:
    """
    Suspicion score formula (additive, capped at 100):
      • In a cycle:            +40
      • Fan-in / fan-out node: +30
      • Shell network node:    +20
      • High velocity:         +10

    Merchants (>50 txns) are skipped entirely and receive score 0.

    Time complexity: O(V) where V = number of unique accounts.
    """
    scores: dict[str, float] = {}
    for acct in all_accounts:
        if acct in merchants:
            continue  # skip merchants
        score = 0.0
        if acct in cycle_accounts:
            score += 40
        if acct in fan_in_out_accounts:
            score += 30
        if acct in shell_accounts:
            score += 20
        if acct in high_vel_accounts:
            score += 10
        score = min(score, 100)
        if score > 0:
            scores[acct] = score
    return scores


# ===========================================================================
# 8. MAIN ANALYSIS PIPELINE
# ===========================================================================

def run_analysis(df: pd.DataFrame) -> dict:
    """
    Orchestrate all detection algorithms and build the output JSON.
    """
    start = time.time()

    # --- Parse timestamps ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Build graph ---
    G = build_graph(df)
    all_accounts = set(G.nodes())

    # --- Detection ---
    cycles = detect_cycles(G)
    smurfing = detect_smurfing(df)
    shell_nodes = detect_shell_networks(G)
    high_vel = detect_high_velocity(df)
    merchants = get_merchant_accounts(df)

    # Flatten cycle accounts and build ring mappings
    cycle_accounts: set[str] = set()
    # Map account → list of ring IDs it belongs to
    account_rings: dict[str, list[str]] = {}
    fraud_rings: list[dict] = []

    for idx, cycle in enumerate(cycles, start=1):
        ring_id = f"RING_{idx:03d}"
        ring_members = [str(a) for a in cycle]

        # Compute ring risk score = average individual score of members
        # (we'll update this after scoring)
        fraud_rings.append(
            {
                "ring_id": ring_id,
                "member_accounts": ring_members,
                "pattern_type": f"cycle_length_{len(cycle)}",
                "risk_score": 0.0,  # placeholder, computed below
            }
        )
        for acct in cycle:
            cycle_accounts.add(acct)
            account_rings.setdefault(acct, []).append(ring_id)

    fan_in_out_accounts = smurfing["fan_in_nodes"] | smurfing["fan_out_nodes"]

    # --- Scoring ---
    scores = compute_scores(
        all_accounts, cycle_accounts, fan_in_out_accounts,
        shell_nodes, high_vel, merchants,
    )

    # --- Update ring risk scores (average member score) ---
    for ring in fraud_rings:
        member_scores = [scores.get(m, 0.0) for m in ring["member_accounts"]]
        ring["risk_score"] = round(
            sum(member_scores) / max(len(member_scores), 1), 1
        )

    # --- Build suspicious accounts list ---
    suspicious_accounts: list[dict] = []
    for acct, score in sorted(scores.items(), key=lambda x: -x[1]):
        patterns: list[str] = []
        # Which cycle lengths?
        for cycle in cycles:
            if acct in cycle:
                patterns.append(f"cycle_length_{len(cycle)}")
        if acct in smurfing["fan_in_nodes"]:
            patterns.append("fan_in")
        if acct in smurfing["fan_out_nodes"]:
            patterns.append("fan_out")
        if acct in shell_nodes:
            patterns.append("shell_network")
        if acct in high_vel:
            patterns.append("high_velocity")
        # Deduplicate while preserving order
        patterns = list(dict.fromkeys(patterns))

        entry: dict = {
            "account_id": str(acct),
            "suspicion_score": score,
            "detected_patterns": patterns,
        }
        # Attach first ring_id if the account is in a ring
        if acct in account_rings:
            entry["ring_id"] = account_rings[acct][0]
        suspicious_accounts.append(entry)

    # --- Also add smurfing-based rings (fan-in / fan-out clusters) ---
    ring_counter = len(fraud_rings)

    # Fan-in rings: each receiver with ≥10 senders forms a ring
    for node in smurfing["fan_in_nodes"]:
        if node in merchants:
            continue
        ring_counter += 1
        ring_id = f"RING_{ring_counter:03d}"
        # Collect the senders that sent to this node
        senders = df[df["receiver_id"] == node]["sender_id"].unique().tolist()
        members = [str(node)] + [str(s) for s in senders if s not in merchants]
        member_scores = [scores.get(str(m), 0.0) for m in members]
        fraud_rings.append({
            "ring_id": ring_id,
            "member_accounts": members,
            "pattern_type": "fan_in",
            "risk_score": round(
                sum(member_scores) / max(len(member_scores), 1), 1
            ),
        })

    # Fan-out rings
    for node in smurfing["fan_out_nodes"]:
        if node in merchants:
            continue
        ring_counter += 1
        ring_id = f"RING_{ring_counter:03d}"
        receivers = df[df["sender_id"] == node]["receiver_id"].unique().tolist()
        members = [str(node)] + [str(r) for r in receivers if r not in merchants]
        member_scores = [scores.get(str(m), 0.0) for m in members]
        fraud_rings.append({
            "ring_id": ring_id,
            "member_accounts": members,
            "pattern_type": "fan_out",
            "risk_score": round(
                sum(member_scores) / max(len(member_scores), 1), 1
            ),
        })

    elapsed = round(time.time() - start, 1)

    result = {
        "suspicious_accounts": suspicious_accounts,
        "fraud_rings": fraud_rings,
        "summary": {
            "total_accounts_analyzed": len(all_accounts),
            "suspicious_accounts_flagged": len(suspicious_accounts),
            "fraud_rings_detected": len(fraud_rings),
            "processing_time_seconds": elapsed,
        },
    }
    return result


# ===========================================================================
# FLASK ENDPOINTS
# ===========================================================================

@app.route("/ping", methods=["GET"])
def ping():
    """Health-check / keep-alive endpoint."""
    return jsonify({"status": "alive"})


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a CSV file via multipart/form-data.

    Expected CSV columns:
      transaction_id, sender_id, receiver_id, amount, timestamp

    Returns the full analysis JSON.
    """
    global _last_result

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    try:
        stream = io.StringIO(file.stream.read().decode("utf-8"))
        df = pd.read_csv(stream)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

    required_cols = {"transaction_id", "sender_id", "receiver_id", "amount", "timestamp"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    result = run_analysis(df)
    _last_result = result
    return jsonify(result)


@app.route("/download-json", methods=["GET"])
def download_json():
    """Return the last analysis result as a downloadable JSON file."""
    if _last_result is None:
        return jsonify({"error": "No analysis has been run yet. Upload a CSV first."}), 404

    return Response(
        json.dumps(_last_result, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=analysis_result.json"},
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("🚀  Money Muling Detection API running on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
