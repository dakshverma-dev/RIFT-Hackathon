"""
Synthetic Transaction Data Generator for Money Muling Detection Testing
=======================================================================
Generates a 10,000-row CSV with planted fraud patterns and an answer key.
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# --- Configuration ---
OUTPUT_FILE = "test_10k.csv"
TOTAL_ROWS = 10_000
TS_START = datetime(2024, 1, 1)
TS_END = datetime(2024, 3, 1)
AMOUNT_MIN, AMOUNT_MAX = 500, 75_000

rows = []
tx_counter = 0
answer_key = {
    "cycle_3": [],
    "cycle_4": [],
    "cycle_5": [],
    "fan_out": [],
    "fan_in": [],
    "shell": [],
    "merchants": [],
}


def next_tx_id():
    global tx_counter
    tx_counter += 1
    return f"TX_{tx_counter:06d}"


def rand_ts(base=None, window_hours=48):
    if base:
        return base + timedelta(hours=random.uniform(0, window_hours))
    delta = TS_END - TS_START
    return TS_START + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def rand_amount():
    return round(random.uniform(AMOUNT_MIN, AMOUNT_MAX), 2)


def add_row(sender, receiver, ts=None, amount=None):
    rows.append({
        "transaction_id": next_tx_id(),
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": amount or rand_amount(),
        "timestamp": (ts or rand_ts()).strftime("%Y-%m-%d %H:%M:%S"),
    })


# =====================================================================
# 1. CYCLES OF LENGTH 3  (5 cycles × 3 accounts = CYC3_001..CYC3_015)
# =====================================================================
print("Planting 5 cycles of length 3...")
for i in range(5):
    a, b, c = [f"CYC3_{i*3+j+1:03d}" for j in range(3)]
    answer_key["cycle_3"].extend([a, b, c])
    base = rand_ts()
    add_row(a, b, base, rand_amount())
    add_row(b, c, base + timedelta(hours=2), rand_amount())
    add_row(c, a, base + timedelta(hours=4), rand_amount())

# =====================================================================
# 2. CYCLES OF LENGTH 4  (3 cycles × 4 accounts = CYC4_001..CYC4_012)
# =====================================================================
print("Planting 3 cycles of length 4...")
for i in range(3):
    accts = [f"CYC4_{i*4+j+1:03d}" for j in range(4)]
    answer_key["cycle_4"].extend(accts)
    base = rand_ts()
    for k in range(4):
        add_row(accts[k], accts[(k+1) % 4], base + timedelta(hours=k*3), rand_amount())

# =====================================================================
# 3. CYCLES OF LENGTH 5  (2 cycles × 5 accounts = CYC5_001..CYC5_010)
# =====================================================================
print("Planting 2 cycles of length 5...")
for i in range(2):
    accts = [f"CYC5_{i*5+j+1:03d}" for j in range(5)]
    answer_key["cycle_5"].extend(accts)
    base = rand_ts()
    for k in range(5):
        add_row(accts[k], accts[(k+1) % 5], base + timedelta(hours=k*2), rand_amount())

# =====================================================================
# 4. FAN-OUT NODES  (4 nodes, each → 12 receivers in 48 hrs)
# =====================================================================
print("Planting 4 fan-out hubs...")
for i in range(4):
    hub = f"FAN_OUT_{i+1:03d}"
    answer_key["fan_out"].append(hub)
    base = rand_ts()
    for j in range(12):
        receiver = f"FO_RCV_{i+1:03d}_{j+1:02d}"
        add_row(hub, receiver, rand_ts(base, 48), rand_amount())

# =====================================================================
# 5. FAN-IN NODES  (3 nodes, each ← 15 senders in 60 hrs)
# =====================================================================
print("Planting 3 fan-in collectors...")
for i in range(3):
    collector = f"FAN_IN_{i+1:03d}"
    answer_key["fan_in"].append(collector)
    base = rand_ts()
    for j in range(15):
        sender = f"FI_SND_{i+1:03d}_{j+1:02d}"
        add_row(sender, collector, rand_ts(base, 60), rand_amount())

# =====================================================================
# 6. SHELL CHAINS  (3 chains of 4 hops each)
# =====================================================================
print("Planting 3 shell chains (4 hops each)...")
for i in range(3):
    chain = [f"SHELL_{i+1:03d}_{hop}" for hop in "ABCDE"]
    answer_key["shell"].extend(chain)
    base = rand_ts()
    for k in range(4):
        add_row(chain[k], chain[k+1], base + timedelta(hours=k*6), rand_amount())
    # Give intermediate nodes (B, C, D) only 1 extra tx each so total = 2-3
    for mid in chain[1:4]:
        add_row(mid, f"SHELL_EXT_{random.randint(1000,9999)}", rand_ts(), rand_amount())

# =====================================================================
# 7. LEGITIMATE MERCHANTS  (5 merchants, 60+ txns each)
# =====================================================================
print("Planting 5 legitimate merchants (60+ txns each)...")
for i in range(5):
    merch = f"MERCH_{i+1:03d}"
    answer_key["merchants"].append(merch)
    num_txns = random.randint(60, 75)
    for _ in range(num_txns):
        rand_acct = f"ACC_{random.randint(1000, 5000)}"
        if random.random() < 0.5:
            add_row(rand_acct, merch, rand_ts(), rand_amount())
        else:
            add_row(merch, rand_acct, rand_ts(), rand_amount())

# =====================================================================
# 8. RANDOM LEGITIMATE TRANSACTIONS  (fill to 10,000)
# =====================================================================
remaining = TOTAL_ROWS - len(rows)
print(f"Filling {remaining} random legitimate transactions...")
for _ in range(remaining):
    sender = f"ACC_{random.randint(1000, 5000)}"
    receiver = f"ACC_{random.randint(1000, 5000)}"
    while receiver == sender:
        receiver = f"ACC_{random.randint(1000, 5000)}"
    add_row(sender, receiver, rand_ts(), rand_amount())

# --- Shuffle all rows for realism ---
random.shuffle(rows)

# --- Write CSV ---
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Generated {len(rows)} transactions → {OUTPUT_FILE}")

# =====================================================================
# ANSWER KEY
# =====================================================================
print("\n" + "=" * 60)
print("📋  ANSWER KEY — Planted Fraud Patterns")
print("=" * 60)

print(f"\n🔴 CYCLES (length 3) — {len(answer_key['cycle_3'])} accounts:")
for a in answer_key["cycle_3"]:
    print(f"   {a}")

print(f"\n🟠 CYCLES (length 4) — {len(answer_key['cycle_4'])} accounts:")
for a in answer_key["cycle_4"]:
    print(f"   {a}")

print(f"\n🟡 CYCLES (length 5) — {len(answer_key['cycle_5'])} accounts:")
for a in answer_key["cycle_5"]:
    print(f"   {a}")

print(f"\n📤 FAN-OUT hubs — {len(answer_key['fan_out'])} accounts:")
for a in answer_key["fan_out"]:
    print(f"   {a}")

print(f"\n📥 FAN-IN collectors — {len(answer_key['fan_in'])} accounts:")
for a in answer_key["fan_in"]:
    print(f"   {a}")

print(f"\n🐚 SHELL chain nodes — {len(answer_key['shell'])} accounts:")
for a in answer_key["shell"]:
    print(f"   {a}")

print(f"\n🏪 MERCHANTS (should NOT be flagged) — {len(answer_key['merchants'])} accounts:")
for a in answer_key["merchants"]:
    print(f"   {a}")

all_fraud = set(
    answer_key["cycle_3"] + answer_key["cycle_4"] + answer_key["cycle_5"]
    + answer_key["fan_out"] + answer_key["fan_in"] + answer_key["shell"]
)
print(f"\n🎯 Total unique fraud accounts to detect: {len(all_fraud)}")
print(f"🏪 Total merchants to skip:               {len(answer_key['merchants'])}")
print("=" * 60)
