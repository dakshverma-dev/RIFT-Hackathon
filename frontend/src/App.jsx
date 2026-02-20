import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

import UploadZone from "./components/UploadZone";
import StatsRow from "./components/StatsRow";
import GraphView from "./components/GraphView";
import RingsTable from "./components/RingsTable";
import AccountsTable from "./components/AccountsTable";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const statsRef = useRef(null);

  /* Health check on mount */
  useEffect(() => {
    axios.get(`${API_BASE}/ping`).catch(() => { });
  }, []);

  /* Upload & analyze */
  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await axios.post(`${API_BASE}/upload`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
      setTimeout(() => {
        statsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      setError(err.response?.data?.error || "Analysis failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  /* Download JSON */
  const handleDownload = async () => {
    try {
      const res = await axios.get(`${API_BASE}/download-json`, { responseType: "blob" });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement("a");
      a.href = url;
      a.download = "fraud_report.json";
      a.click();
      window.URL.revokeObjectURL(url);
    } catch {
      setError("Download failed. Run an analysis first.");
    }
  };

  return (
    <>
      {/* NAV */}
      <nav className="nav">
        <span className="nav-brand">FinForensics</span>
        <span className="nav-sub">Graph Analysis Engine</span>
      </nav>

      {/* HERO */}
      <section className="hero">
        <p className="hero-label">Financial Crime Detection &middot; RIFT 2026</p>
        <h1 className="hero-heading">
          <span className="hero-heading-line1">Detect money muling</span>
          <span className="hero-heading-line2">before it disappears.</span>
        </h1>
        <p className="hero-sub">
          Upload a transaction CSV. Our graph engine surfaces fraud rings,
          smurfing patterns, and shell networks — automatically.
        </p>
      </section>

      {/* UPLOAD */}
      <UploadZone
        file={file}
        setFile={setFile}
        onAnalyze={handleAnalyze}
        loading={loading}
        error={error}
      />

      {/* RESULTS — only shown after analysis */}
      {result && (
        <>
          <div ref={statsRef}>
            <StatsRow summary={result.summary} />
          </div>
          <GraphView result={result} />
          <RingsTable rings={result.fraud_rings} />
          <AccountsTable accounts={result.suspicious_accounts} />

          {/* DOWNLOAD */}
          <div className="download-section">
            <button className="btn-download" onClick={handleDownload}>
              ↓ Download fraud_report.json
            </button>
          </div>
        </>
      )}

      {/* FOOTER */}
      <footer className="footer">
        FinForensics &middot; RIFT 2026 &middot; Graph Theory Track
      </footer>
    </>
  );
}
