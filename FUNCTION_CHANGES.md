# Function Signature Changes - Behavioral Severity Scoring Upgrade

## 1. Graph Construction

### `build_graph(df: pd.DataFrame) -> nx.DiGraph`
**Change**: Added edge attributes for temporal and financial data

```python
# Edge attributes now include:
G[sender][receiver]["amounts"] = [amt1, amt2, ...]  # Transaction amounts
G[sender][receiver]["timestamps"] = [ts1, ts2, ...]  # Transaction timestamps
```

---

## 2. Cycle Detection

### Before:
```python
def detect_cycles(G: nx.DiGraph) -> list[list[str]]:
    # Returns only the cycles
    return cycles  # [[a,b,c], [d,e,f], ...]
```

### After:
```python
def detect_cycles(G: nx.DiGraph) -> tuple[list[list[str]], dict[str, float]]:
    # Returns cycles AND continuous risk scores for each node
    return cycles, cycle_score
    # cycles: [[a,b,c], [d,e,f], ...]
    # cycle_score: {"a": 45.2, "b": 45.2, "c": 45.2, "d": 38.1, ...}
```

**New computation inside loop:**
```python
cycle_time = max(timestamps) - min(timestamps)
flow_loss = abs(in_amt - out_amt) / max(in_amt, 1)
cycle_risk = (6 - len(cycle)) * 10 + (86400 - cycle_time) / 3600 + (1 - flow_loss) * 20

for node in cycle:
    cycle_score[str(node)] += cycle_risk
```

---

## 3. Smurfing Detection

### Before:
```python
def detect_smurfing(df: pd.DataFrame) -> dict:
    return {
        "fan_in_nodes": set(...),
        "fan_out_nodes": set(...)
    }
```

### After:
```python
def detect_smurfing(df: pd.DataFrame) -> dict:
    return {
        "fan_in_nodes": set(...),
        "fan_out_nodes": set(...),
        "fan_score": dict[str, float]  # NEW
    }
```

**New computation when distinct_count ≥ 10:**
```python
window_duration_hours = (max_ts - min_ts) / (1e9 * 3600)  # nanoseconds to hours
fan_intensity = distinct_count / max(window_duration_hours, 1)
fan_score[str(node)] += fan_intensity
```

---

## 4. Shell Network Detection

### Before:
```python
def detect_shell_networks(G: nx.DiGraph) -> set[str]:
    return shell_nodes  # {"a", "b", "c", ...}
```

### After:
```python
def detect_shell_networks(G: nx.DiGraph, df: pd.DataFrame = None) -> dict:
    return {
        "shell_nodes": set(...),
        "shell_score": dict[str, float]  # NEW
    }
```

**New computation for each shell node:**
```python
out_amt = df[df["sender_id"] == node]["amount"].sum()
in_amt = df[df["receiver_id"] == node]["amount"].sum()

efficiency = out_amt / max(in_amt, 1)
avg_latency = mean(outgoing_ts - incoming_ts)  # in seconds
avg_latency_hours = avg_latency / 3600

shell_risk = efficiency * (1 / max(avg_latency_hours, 1))
shell_score[str(node)] += shell_risk
```

---

## 5. High Velocity Detection

### Before:
```python
def detect_high_velocity(df: pd.DataFrame) -> set[str]:
    return high_vel  # {"a", "b", "c", ...}
```

### After:
```python
def detect_high_velocity(df: pd.DataFrame) -> dict:
    return {
        "high_vel_accounts": set(...),
        "velocity_score": dict[str, float]  # NEW
    }
```

**New computation when >10 transactions in 24-hour window:**
```python
window_duration_hours = (max_ts - min_ts) / (1e9 * 3600)
txn_count = right - left + 1
txn_rate = txn_count / max(window_duration_hours, 1)
velocity_score[str(account)] += txn_rate
```

---

## 6. Suspicion Scoring

### Before:
```python
def compute_scores(
    all_accounts: set[str],
    cycle_accounts: set[str],           # Binary detection
    fan_in_out_accounts: set[str],      # Binary detection
    shell_accounts: set[str],           # Binary detection
    high_vel_accounts: set[str],        # Binary detection
    merchants: set[str],
) -> dict[str, float]:
    # Binary scoring: +40, +30, +20, +10
    score = 0
    if acct in cycle_accounts: score += 40
    if acct in fan_in_out_accounts: score += 30
    if acct in shell_accounts: score += 20
    if acct in high_vel_accounts: score += 10
    score = min(score, 100)
    return scores
```

### After:
```python
def compute_scores(
    all_accounts: set[str],
    cycle_score: dict[str, float],      # Continuous scores
    fan_score: dict[str, float],        # Continuous scores
    shell_score: dict[str, float],      # Continuous scores
    velocity_score: dict[str, float],   # Continuous scores
    merchants: set[str],
) -> dict[str, float]:
    # Weighted combination: 0.5 + 0.2 + 0.2 + 0.1 = 1.0
    w1, w2, w3, w4 = 0.5, 0.2, 0.2, 0.1
    
    c_score = cycle_score.get(str(acct), 0.0)
    f_score = fan_score.get(str(acct), 0.0)
    s_score = shell_score.get(str(acct), 0.0)
    v_score = velocity_score.get(str(acct), 0.0)
    
    score = w1 * c_score + w2 * f_score + w3 * s_score + w4 * v_score
    score = min(score, 100)
    return scores
```

---

## 7. Main Analysis Pipeline

### Before:
```python
def run_analysis(df: pd.DataFrame) -> dict:
    cycles = detect_cycles(G)
    smurfing = detect_smurfing(df)
    shell_nodes = detect_shell_networks(G)
    high_vel = detect_high_velocity(df)
    merchants = get_merchant_accounts(df)
    
    scores = compute_scores(
        all_accounts, cycle_accounts, fan_in_out_accounts,
        shell_nodes, high_vel, merchants
    )
    # ... rest unchanged
```

### After:
```python
def run_analysis(df: pd.DataFrame) -> dict:
    cycles, cycle_score = detect_cycles(G)  # CHANGED: capture score
    smurfing = detect_smurfing(df)
    shell_result = detect_shell_networks(G, df)  # CHANGED: pass df, get dict
    high_vel_result = detect_high_velocity(df)  # CHANGED: get dict
    merchants = get_merchant_accounts(df)  # Unchanged
    
    shell_nodes = shell_result["shell_nodes"]
    shell_score = shell_result["shell_score"]
    high_vel = high_vel_result["high_vel_accounts"]
    velocity_score = high_vel_result["velocity_score"]
    fan_score = smurfing["fan_score"]
    
    scores = compute_scores(
        all_accounts, cycle_score, fan_score, shell_score, velocity_score, merchants
    )
    # ... rest UNCHANGED (output format identical)
```

---

## Backward Compatibility

| Item | Status | Impact |
|------|--------|--------|
| Flask routes | ✅ Unchanged | No API changes |
| JSON output schema | ✅ Unchanged | Clients see same structure |
| Suspicious accounts format | ✅ Unchanged | `{account_id, suspicion_score, detected_patterns, ring_id}` |
| Fraud rings structure | ✅ Unchanged | `{ring_id, member_accounts, pattern_type, risk_score}` |
| Merchant filter logic | ✅ Unchanged | Still excludes >50 txns accounts |
| Pattern labels | ✅ Unchanged | `cycle_length_X`, `fan_in`, `fan_out`, `shell_network`, `high_velocity` |
| Performance | ✅ Maintained | O(E log E), still sub-2 seconds |

---

## Score Range Examples

**Old Method:**
- Cycle only: 40
- Fan-in only: 30
- Shell only: 20
- Velocity only: 10
- All four + more: 100 (capped immediately)

**New Method (w1=0.5, w2=0.2, w3=0.2, w4=0.1):**
- Mild cycle (risk=20) + no others: **0.5 × 20 = 10**
- Strong cycle (risk=60) + mild fan (intensity=5): **0.5 × 60 + 0.2 × 5 = 31**
- Strong cycle (risk=80) + strong fan (intensity=50) + shell (risk=30) + velocity (rate=15):
  - **0.5 × 80 + 0.2 × 50 + 0.2 × 30 + 0.1 × 15 = 40 + 10 + 6 + 1.5 = 57.5**
- Extreme case (all maxed): **min(0.5×100 + 0.2×100 + 0.2×100 + 0.1×100, 100) = 100**

---

## Migration Path

1. **Deploy** the new `app.py`
2. **No data migration** needed (pure logic change)
3. **Reprocess** historical data if needed for comparison
4. **Monitor** new score distribution (will be lower, more granular)
5. **Adjust thresholds** if using alerts (recommend recalibration)
