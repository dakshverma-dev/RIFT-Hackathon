# FinForensics - Money Muling Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![React](https://img.shields.io/badge/React-Vite-61DAFB) ![Flask](https://img.shields.io/badge/Flask-Backend-000000) ![NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange)

FinForensics is a high-performance, graph-based financial crime detection system designed to identify money muling patterns in transaction data. It leverages advanced graph algorithms to detect complex fraud topologies such as cyclic flows, smurfing (structuring), and shell account networks.

**[Live Demo (Placeholder)](#)** | **[Video Demo](#)**

---

##  Tech Stack

### Backend
*   **Language**: Python 3.10+
*   **Framework**: Flask (REST API)
*   **Graph/Data Processing**: NetworkX, Pandas, NumPy
*   **Optimizations**: Vectorized operations, Int64 timestamp arithmetic, SCC decomposition

### Frontend
*   **Framework**: React (Vite)
*   **Styling**: Tailwind CSS
*   **Visualization**: Cytoscape.js (Interactive Graph)
*   **HTTP Client**: Axios

---

##  System Architecture

The system follows a stateless, decoupled architecture:

1.  **Data Ingestion**: User uploads a CSV file (`transaction_id`, `sender_id`, `receiver_id`, `amount`, `timestamp`).
2.  **Preprocessing**: Timestamps are converted to Int64 nanoseconds for O(1) comparison; DataFrames are indexed.
3.  **Graph Construction**: A directed graph `G(V, E)` is built using NetworkX.
4.  **Detection Engine**:
    *   **Cycle Detection**: Finds circular money flows.
    *   **Smurfing Detection**: Identifies fan-in/fan-out patterns.
    *   **Shell Detection**: Finds high-efficiency pass-through nodes.
    *   **Velocity Check**: Flags high-frequency transaction bursts.
5.  **Scoring & Aggregation**: Each account is assigned a suspicion score (0-100) based on weighted behaviors.
6.  **Visualization**: The React frontend renders the fraud rings and high-risk accounts.

---

##  Algorithm Approach & Complexity

### 1. Cycle Detection (The "Money Loop")
Detects funds returning to the originator (e.g., A -> B -> C -> A), a classic money laundering pattern.
*   **Method**: 
    1.  Prunes the graph to Strongly Connected Components (SCCs) to filter out acyclic nodes.
    2.  Uses Depth First Search (DFS) with a depth limit (3-5 hops) to find simple cycles.
*   **Optimization**: Large SCCs (>100 nodes) use degree-capped local search to prevent explosion.
*   **Complexity**: **O(V + E)** for SCCs. Cycle finding is bounded by specific depth/degree limits, ensuring sub-second execution on typical datasets.

### 2. Smurfing Detection (Fan-In / Fan-Out)
Detects "structuring" where funds are split into small amounts or aggregated from many sources.
*   **Method**: Sliding time window (72 hours) over sorted transactions.
*   **Optimization**: Uses a `defaultdict` counter that updates incrementally as the window slides (Two-Pointer approach), avoiding re-computation.
*   **Complexity**: **O(E log E)** (due to sorting). Sliding window scan is **O(E)**.

### 3. Shell Network Detection
Identifies "mule" accounts that act as mere conduits (High pass-through efficiency, low latency).
*   **Method**: DFS chain-walk on accounts with low degree (2-3) but high flow efficiency (Incoming ≈ Outgoing).
*   **Criteria**: Pass-through ratio > 0.7, clear temporal ordering (`t_in < t_out`).
*   **Complexity**: **O(V_shell * Depth)**, where `V_shell` is the number of candidate shell nodes.

### 4. Suspicion Scoring Model
A weighted additive model produces a continuous risk score **[0, 100]**.
*   **Formula**:
    ```python
    Score = (1.0 * Cycle_Score) + 
            (0.5 * Fan_Score) + 
            (0.5 * Shell_Score) + 
            (0.3 * Velocity_Score)
    ```
*   **Clamping**: Final score is capped at 100. A single strong indicator (like a verified cycle) can trigger a max score.

---

##  Installation & Setup

### Prerequisites
*   Python 3.10+
*   Node.js 18+

### 1. Backend Setup
```bash
# Clone repository
git clone https://github.com/dakshverma-dev/RIFT-Hackathon.git
cd RIFT-Hackathon

# Install dependencies
pip install -r requirements.txt
# (or just: pip install flask flask-cors pandas networkx numpy)

# Run Server
python app.py
```
*Server runs on `http://127.0.0.1:5000`*

### 2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run Development Server
npm run dev
```
*App runs on `http://localhost:5173`*

---

## 📖 Usage Instructions

1.  **Upload CSV**: Drag and drop a transaction CSV file.
    *   *Required Columns*: `transaction_id`, `sender_id`, `receiver_id`, `amount`, `timestamp`
2.  **Analyze**: The system processes the file (~1-2 seconds for 10k rows).
3.  **View Results**:
    *   **Graph**: Interactive view of fraud rings. Red nodes = High Risk.
    *   **Stats**: Summary of total money laundered estimate.
4.  **Download Report**: Click "Download JSON" to get the full forensic report.

---

##  Known Limitations

*   **In-Memory Processing**: Graph is built in RAM. Extremely large datasets (>1M rows) may require batching or a graph DB (Neo4j).
*   **Static Rules**: Thresholds (e.g., "72h window") are currently hardcoded but can be parameterized.
*   **Single Layout**: The graph visualization uses a force-directed layout which may be cluttered for very dense networks.

---

##  Team

*   **[Member 1 Name]** - Backend & Algorithms
*   **[Member 2 Name]** - Frontend & Visualization
*   **[Member 3 Name]** - Data Analysis & Testing
