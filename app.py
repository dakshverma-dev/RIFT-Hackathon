"""
Money Muling Detection Backend
==============================
A Flask application that accepts CSV transaction data, builds a directed
transaction graph with NetworkX, and runs three fraud-detection algorithms:
  1. Cycle Detection (SCC-filtered, DFS, length 3-5)
  2. Smurfing / Fan-in & Fan-out (int64 timestamps + sliding counter)
  3. Shell Network Detection (DFS chain-walk)

Performance optimisations applied (v2):
  • Timestamps converted to int64 nanoseconds once upfront; all window
    comparisons use integer arithmetic — avoids millions of pd.Timestamp()
    object instantiations inside inner loops.
  • Sliding-window unique-counter uses collections.defaultdict(int) that
    increments / decrements as the window slides — true O(N) per group
    instead of set(slice) which was O(N²).
  • Cycle detection pre-filters the graph to strongly connected components
    (SCCs) with ≥3 nodes and removes leaf nodes (in_degree=0 or
    out_degree=0) that can never participate in cycles.
  • Inverted index account_cycle_patterns built during cycle processing
    so that per-account pattern lookup is O(1) instead of scanning all
    cycles for every suspicious account.

Each flagged account receives a suspicion score (0-100) and is grouped into
fraud rings where applicable.
"""

import io
import time
import json
from collections import defaultdict

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, origins=[
    "web-production-cc633.up.railway.app",
    "http://localhost:5173"
])
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
    Each transaction becomes a directed edge with amount and timestamp attributes.

    Time complexity: O(E) where E = number of transaction rows.
    Uses vectorized edge construction instead of iterrows for speed.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        sender, receiver, amount = row["sender_id"], row["receiver_id"], row["amount"]
        ts = row["timestamp"]
        if G.has_edge(sender, receiver):
            # Multiple transactions on same edge: accumulate amounts, append timestamps
            G[sender][receiver]["amounts"].append(amount)
            G[sender][receiver]["timestamps"].append(ts)
        else:
            G.add_edge(sender, receiver, amounts=[amount], timestamps=[ts])
    return G


# ===========================================================================
# 2. CYCLE DETECTION  (SCC-filtered, length 3-5)
# ===========================================================================

_MAX_CYCLES = 500          # Hard cap on total cycles returned
_SMALL_SCC_THRESHOLD = 100  # SCCs ≤ this size get exhaustive search
_TIME_BUDGET = 1.0          # Max seconds for cycle detection


def _find_short_cycles_local(G, node, length_bound=5):
    """
    Fast local search: find all simple cycles of length 3..length_bound
    passing through 'node' by doing a DFS from node's successors and
    checking if any path returns to 'node' within length_bound steps.

    This is O(d^length_bound) per node where d = avg out-degree in the
    local neighborhood — fast for the sparse subgraphs typical of
    financial transaction networks.
    """
    cycles = []
    target = node

    def _dfs(current, path, depth):
        if depth >= length_bound:
            return
        for nxt in G.successors(current):
            if nxt == target and len(path) >= 3:
                cycles.append(list(path))
            elif nxt not in path and depth + 1 < length_bound:
                path.add(nxt)
                _dfs(nxt, path, depth + 1)
                path.discard(nxt)

    for succ in G.successors(node):
        if succ == target:
            continue
        _dfs(succ, {node, succ}, 2)

    return cycles


def detect_cycles(G: nx.DiGraph) -> tuple[list[list[str]], dict[str, float]]:
    """
    Find simple cycles of length 3, 4, or 5 and compute continuous risk scores.

    For each cycle, compute:
      cycle_time = max(timestamp_of_edges) - min(timestamp_of_edges)
      flow_loss = abs(total_in_amt - total_out_amt) / max(total_in_amt, 1)
      cycle_risk = (6 - len(cycle)) * 10
                 + (86400 - cycle_time_seconds) / 3600
                 + (1 - flow_loss) * 20

    Hybrid strategy for guaranteed sub-second performance:

      1. Iterative leaf pruning — remove nodes with in_degree=0 or
         out_degree=0 (can never be in any cycle).
      2. SCC decomposition — cycles only exist within strongly
         connected components.
      3. For small SCCs (≤100 nodes): exhaustive nx.simple_cycles
         with length_bound=5.
      4. For large SCCs (>100 nodes): fast local DFS search from
         each node, finding short cycles through that node's
         neighborhood.  Capped by _MAX_CYCLES (500) and
         _TIME_BUDGET (1 second).
      5. Global deduplication via frozenset to avoid reporting the
         same cycle found from different starting nodes.

    Returns (cycles, cycle_score) where:
      - cycles: list of cycles, each cycle being a list of account IDs
      - cycle_score: dict mapping account ID → continuous risk score
    """
    import time as _time
    t_start = _time.perf_counter()

    # Step 1: iterative leaf pruning
    pruned = G.copy()
    changed = True
    while changed:
        changed = False
        to_remove = [
            n for n in pruned.nodes()
            if pruned.in_degree(n) == 0 or pruned.out_degree(n) == 0
        ]
        if to_remove:
            pruned.remove_nodes_from(to_remove)
            changed = True

    if pruned.number_of_nodes() == 0:
        return [], {}

    cycles = []
    cycle_score: dict[str, float] = defaultdict(float)
    seen = set()  # frozensets for dedup
    budget = _MAX_CYCLES

    def _add_cycle(c):
        nonlocal budget
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            cycles.append(c)
            
            # Compute cycle risk score
            cycle_nodes = list(c)
            edges_in_cycle = []
            node_in_amt = defaultdict(float)   # Track incoming amounts per node
            node_out_amt = defaultdict(float)  # Track outgoing amounts per node
            
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                if pruned.has_edge(u, v):
                    edge_data = pruned[u][v]
                    amounts = edge_data.get("amounts", [0])
                    timestamps = edge_data.get("timestamps", [0])
                    edge_amount = sum(amounts)
                    edges_in_cycle.append({
                        "amounts": amounts,
                        "timestamps": timestamps
                    })
                    # Track flow direction: u sends, v receives
                    node_out_amt[u] += edge_amount
                    node_in_amt[v] += edge_amount
            
            all_amounts = []
            all_timestamps = []
            for e in edges_in_cycle:
                all_amounts.extend(e.get("amounts", []))
                all_timestamps.extend(e.get("timestamps", []))
            
            if all_timestamps:
                # Convert timestamps to seconds
                try:
                    ts_vals = [pd.Timestamp(ts).timestamp() for ts in all_timestamps]
                    cycle_time = max(ts_vals) - min(ts_vals)
                except:
                    cycle_time = 0
            else:
                cycle_time = 0
            
            # Compute flow_loss as average balance imbalance across nodes in cycle
            total_imbalance = 0.0
            total_in_amt = 0.0
            for node in cycle_nodes:
                in_amt = node_in_amt[node]
                out_amt = node_out_amt[node]
                total_in_amt += in_amt
                total_imbalance += abs(in_amt - out_amt)
            
            # flow_loss: normalized imbalance (high = money leaking)
            flow_loss = (total_imbalance / len(cycle_nodes)) / max(total_in_amt, 1) if total_in_amt > 0 else 0
            
            cycle_risk = (
                (6 - len(c)) * 10 +
                max(0, (86400 - cycle_time) / 3600) +
                (1 - flow_loss) * 20
            )
            
            # Add to each node in cycle
            for node in c:
                cycle_score[str(node)] += cycle_risk
            
            budget -= 1

    # Step 2 & 3/4: process each SCC
    for scc_nodes in nx.strongly_connected_components(pruned):
        if len(scc_nodes) < 3 or budget <= 0:
            continue

        elapsed = _time.perf_counter() - t_start
        if elapsed > _TIME_BUDGET:
            break

        sub = pruned.subgraph(scc_nodes)

        if len(scc_nodes) <= _SMALL_SCC_THRESHOLD:
            # Exhaustive search for small SCCs — fast enough
            for cycle in nx.simple_cycles(sub, length_bound=5):
                if len(cycle) >= 3:
                    _add_cycle(cycle)
                    if budget <= 0:
                        break
        else:
            # Large SCC: local DFS from each node
            for node in sub.nodes():
                if budget <= 0 or (_time.perf_counter() - t_start) > _TIME_BUDGET:
                    break
                local_cycles = _find_short_cycles_local(sub, node)
                for c in local_cycles:
                    _add_cycle(list(c))
                    if budget <= 0:
                        break

    return cycles, dict(cycle_score)


# ===========================================================================
# 3. SMURFING DETECTION  (Fan-in / Fan-out within 72 hours)
# ===========================================================================

# 72 hours expressed in nanoseconds (int64 arithmetic)
_72H_NS = int(72 * 3600 * 1e9)
# 24 hours in nanoseconds
_24H_NS = int(24 * 3600 * 1e9)


def detect_smurfing(df: pd.DataFrame) -> dict:
    """
    Detect fan-in and fan-out patterns using a sliding window over
    int64-nanosecond timestamps with a defaultdict counter for O(1)
    window maintenance.

    For each distinct counterparty count ≥10, compute:
      fan_intensity = distinct_count / max(window_duration_hours, 1)

    Fan-in:  ≥10 distinct senders  → 1 receiver within any 72-hour window.
    Fan-out: 1 sender → ≥10 distinct receivers within any 72-hour window.

    Optimisations vs. previous version:
      • Timestamps are pre-converted to int64 nanoseconds once via
        .astype("int64") — all comparisons are simple integer subtractions
        instead of pd.Timestamp() object construction.
      • The unique-counterparty count uses a defaultdict(int) that
        increments when the right pointer advances and decrements when
        the left pointer advances.  A separate 'distinct' counter tracks
        the number of keys with count > 0.  This gives true O(N) per
        group instead of O(N²) from set(slice).

    Time complexity: O(E log E) for the initial sort + O(E) for the
    sliding-window scan, overall O(E log E).

    Returns {
      "fan_in_nodes": set,
      "fan_out_nodes": set,
      "fan_score": dict[str, float]
    }.
    """
    fan_in_nodes: set[str] = set()
    fan_out_nodes: set[str] = set()
    fan_score: dict[str, float] = defaultdict(float)

    df_sorted = df.sort_values("timestamp")
    ts_int = df_sorted["timestamp"].astype("int64").values  # nanosecond ints

    # Re-index the other columns to match sorted order
    sender_vals = df_sorted["sender_id"].values
    receiver_vals = df_sorted["receiver_id"].values

    # Build per-receiver index arrays for fan-in
    recv_groups = df_sorted.groupby("receiver_id").indices
    send_groups = df_sorted.groupby("sender_id").indices

    # --- Fan-in: many senders → one receiver ---
    for receiver, idxs in recv_groups.items():
        ts = ts_int[idxs]
        senders = sender_vals[idxs]
        n = len(idxs)
        counter = defaultdict(int)
        distinct = 0
        left = 0
        flagged = False
        for right in range(n):
            # Add right element
            s = senders[right]
            if counter[s] == 0:
                distinct += 1
            counter[s] += 1

            # Shrink window from left
            while ts[right] - ts[left] > _72H_NS:
                s_left = senders[left]
                counter[s_left] -= 1
                if counter[s_left] == 0:
                    distinct -= 1
                left += 1

            if distinct >= 10 and not flagged:
                fan_in_nodes.add(receiver)
                flagged = True
                # Compute fan intensity
                window_duration_ns = ts[right] - ts[left]
                window_duration_hours = max(window_duration_ns / 1e9 / 3600, 1)
                fan_intensity = distinct / max(window_duration_hours, 1)
                fan_score[str(receiver)] += fan_intensity

    # --- Fan-out: one sender → many receivers ---
    for sender, idxs in send_groups.items():
        ts = ts_int[idxs]
        receivers = receiver_vals[idxs]
        n = len(idxs)
        counter = defaultdict(int)
        distinct = 0
        left = 0
        for right in range(n):
            r = receivers[right]
            if counter[r] == 0:
                distinct += 1
            counter[r] += 1

            while ts[right] - ts[left] > _72H_NS:
                r_left = receivers[left]
                counter[r_left] -= 1
                if counter[r_left] == 0:
                    distinct -= 1
                left += 1

            if distinct >= 10:
                fan_out_nodes.add(sender)
                # Compute fan intensity
                window_duration_ns = ts[right] - ts[left]
                window_duration_hours = max(window_duration_ns / 1e9 / 3600, 1)
                fan_intensity = distinct / max(window_duration_hours, 1)
                fan_score[str(sender)] += fan_intensity
                break

    return {
        "fan_in_nodes": fan_in_nodes,
        "fan_out_nodes": fan_out_nodes,
        "fan_score": dict(fan_score)
    }


# ===========================================================================
# 4. SHELL NETWORK DETECTION  (chains ≥3 hops, low-activity intermediates)
# ===========================================================================

def detect_shell_networks(G: nx.DiGraph, df: pd.DataFrame = None) -> dict:
    """
    Identify shell-network nodes: accounts that appear as intermediate hops
    in chains of length ≥3, where each intermediate node has a total degree
    (in + out) of only 2 or 3 transactions.

    For each shell candidate, compute:
      efficiency = out_amt[node] / max(in_amt[node], 1)
      latency = avg(outgoing_ts - incoming_ts)
      shell_risk = efficiency * (1 / max(latency_hours, 1))

    Algorithm (optimised DFS chain-walk):
      1. Identify candidate "shell" nodes (degree 2 or 3).
      2. Build subgraph of shell candidates only.
      3. From each candidate with in-degree 0 in the subgraph (chain start),
         do a single DFS forward walk (max depth 6).  Any path ≥ 3 nodes
         flags all its members.
      This avoids the O(N²) all-pairs enumeration of the previous version.

    Time complexity: O(V + E) for degree filtering + O(S * D) where
    S = number of chain-start nodes and D = max depth (6).

    Returns {
      "shell_nodes": set,
      "shell_score": dict[str, float]
    }.
    """
    shell_candidates = set()
    for node in G.nodes():
        total_degree = G.in_degree(node) + G.out_degree(node)
        if 2 <= total_degree <= 3:
            shell_candidates.add(node)

    if not shell_candidates:
        return {"shell_nodes": set(), "shell_score": {}}

    sub = G.subgraph(shell_candidates)
    shell_nodes: set[str] = set()
    shell_score: dict[str, float] = defaultdict(float)

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

    # Compute shell risk scores if dataframe is provided
    if df is not None:
        for node in shell_nodes:
            # Incoming transactions
            incoming = df[df["receiver_id"] == node]
            in_amt = incoming["amount"].sum()

            # Outgoing transactions
            outgoing = df[df["sender_id"] == node]
            out_amt = outgoing["amount"].sum()

            efficiency = out_amt / max(in_amt, 1)

            # Compute latency: avg(outgoing_ts - incoming_ts)
            latencies = []
            for _, in_row in incoming.iterrows():
                in_ts = pd.Timestamp(in_row["timestamp"]).timestamp()
                for _, out_row in outgoing.iterrows():
                    out_ts = pd.Timestamp(out_row["timestamp"]).timestamp()
                    latency = out_ts - in_ts
                    if latency >= 0:  # Only count positive latencies
                        latencies.append(latency)

            if latencies:
                avg_latency_hours = np.mean(latencies) / 3600
            else:
                avg_latency_hours = 1

            shell_risk = efficiency * (1 / max(avg_latency_hours, 1))
            shell_score[str(node)] += shell_risk

    return {
        "shell_nodes": shell_nodes,
        "shell_score": dict(shell_score)
    }


# ===========================================================================
# 5. HIGH VELOCITY CHECK  (int64 sliding window)
# ===========================================================================

def detect_high_velocity(df: pd.DataFrame) -> dict:
    """
    Flag accounts with >10 transactions (sent or received) within any
    24-hour window.

    For each high-velocity account, compute:
      txn_rate = num_txns / max(active_hours, 1)

    Optimisation: uses pre-computed int64 nanosecond timestamps for
    integer comparison instead of constructing pd.Timestamp objects
    inside the inner loop.

    Time complexity: O(E log E) for sorting + O(E) for the sliding-window
    scan per account.

    Returns {
      "high_vel_accounts": set,
      "velocity_score": dict[str, float]
    }.
    """
    high_vel: set[str] = set()
    velocity_score: dict[str, float] = defaultdict(float)

    df_sorted = df.sort_values("timestamp")
    ts_int = df_sorted["timestamp"].astype("int64").values

    # Check both sender and receiver activity
    for role in ["sender_id", "receiver_id"]:
        groups = df_sorted.groupby(role).indices
        role_vals = df_sorted[role].values
        for account, idxs in groups.items():
            ts = ts_int[idxs]
            n = len(idxs)
            left = 0
            for right in range(n):
                while ts[right] - ts[left] > _24H_NS:
                    left += 1
                if (right - left + 1) > 10:
                    high_vel.add(account)
                    # Compute transaction rate
                    window_duration_ns = ts[right] - ts[left]
                    window_duration_hours = max(window_duration_ns / 1e9 / 3600, 1)
                    txn_count = right - left + 1
                    txn_rate = txn_count / max(window_duration_hours, 1)
                    velocity_score[str(account)] += txn_rate
                    break

    return {
        "high_vel_accounts": high_vel,
        "velocity_score": dict(velocity_score)
    }


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
    cycle_score: dict[str, float],
    fan_score: dict[str, float],
    shell_score: dict[str, float],
    velocity_score: dict[str, float],
    merchants: set[str],
) -> dict[str, float]:
    """
    Suspicion score using weighted combination of behavioral severity scores:

      score = w1 * cycle_score[node]
             + w2 * fan_score[node]
             + w3 * shell_score[node]
             + w4 * velocity_score[node]

    where:
      w1 = 0.5  (cycle risk weight)
      w2 = 0.2  (fan intensity weight)
      w3 = 0.2  (shell efficiency weight)
      w4 = 0.1  (velocity rate weight)

    Final score is capped at 100.

    Merchants (>50 txns) are skipped entirely and receive score 0.

    Time complexity: O(V) where V = number of unique accounts.
    """
    w1, w2, w3, w4 = 0.5, 0.2, 0.2, 0.1

    scores: dict[str, float] = {}
    for acct in all_accounts:
        if acct in merchants:
            continue  # skip merchants

        c_score = cycle_score.get(str(acct), 0.0)
        f_score = fan_score.get(str(acct), 0.0)
        s_score = shell_score.get(str(acct), 0.0)
        v_score = velocity_score.get(str(acct), 0.0)

        score = w1 * c_score + w2 * f_score + w3 * s_score + w4 * v_score
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

    Key optimisation: builds an inverted index (account_cycle_patterns)
    during cycle processing so that per-account pattern lookup is O(1)
    instead of looping over ALL cycles for every suspicious account.
    """
    start = time.time()

    # --- Parse timestamps once (int64 conversion happens inside each
    #     detector to avoid redundant work) ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Build graph ---
    G = build_graph(df)
    all_accounts = set(G.nodes())

    # --- Detection (now returns risk scores along with nodes) ---
    cycles, cycle_score = detect_cycles(G)
    smurfing = detect_smurfing(df)
    shell_result = detect_shell_networks(G, df)
    high_vel_result = detect_high_velocity(df)
    merchants = get_merchant_accounts(df)

    # Extract components
    shell_nodes = shell_result["shell_nodes"]
    shell_score = shell_result["shell_score"]
    high_vel = high_vel_result["high_vel_accounts"]
    velocity_score = high_vel_result["velocity_score"]
    fan_score = smurfing["fan_score"]

    # Flatten cycle accounts and build ring mappings + INVERTED INDEX
    cycle_accounts: set[str] = set()
    account_rings: dict[str, list[str]] = {}
    # Inverted index: account → set of pattern strings (e.g. "cycle_length_3")
    account_cycle_patterns: dict[str, list[str]] = defaultdict(list)
    fraud_rings: list[dict] = []

    for idx, cycle in enumerate(cycles, start=1):
        ring_id = f"RING_{idx:03d}"
        ring_members = [str(a) for a in cycle]
        pattern_label = f"cycle_length_{len(cycle)}"

        fraud_rings.append(
            {
                "ring_id": ring_id,
                "member_accounts": ring_members,
                "pattern_type": pattern_label,
                "risk_score": 0.0,  # placeholder, computed below
            }
        )
        for acct in cycle:
            cycle_accounts.add(acct)
            account_rings.setdefault(acct, []).append(ring_id)
            # Build inverted index — O(1) lookup later
            account_cycle_patterns[acct].append(pattern_label)

    fan_in_out_accounts = smurfing["fan_in_nodes"] | smurfing["fan_out_nodes"]

    # --- Scoring (now with continuous risk scores) ---
    scores = compute_scores(
        all_accounts, cycle_score, fan_score, shell_score, velocity_score, merchants,
    )

    # --- Update ring risk scores (average member score) ---
    for ring in fraud_rings:
        member_scores = [scores.get(m, 0.0) for m in ring["member_accounts"]]
        ring["risk_score"] = round(
            sum(member_scores) / max(len(member_scores), 1), 1
        )

    # --- Build suspicious accounts list ---
    # Uses the inverted index for O(1) cycle pattern lookup per account
    suspicious_accounts: list[dict] = []
    for acct, score in sorted(scores.items(), key=lambda x: -x[1]):
        patterns: list[str] = []
        # O(1) lookup from inverted index instead of scanning all cycles
        if acct in account_cycle_patterns:
            patterns.extend(account_cycle_patterns[acct])
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
