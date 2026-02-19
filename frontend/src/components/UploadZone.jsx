import { useState, useCallback, useRef } from "react";

export default function UploadZone({ file, setFile, onAnalyze, loading, error }) {
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
    }, [setFile]);

    const handleChange = (e) => {
        if (e.target.files?.[0]) setFile(e.target.files[0]);
    };

    const openFilePicker = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="upload-wrap">
            <div className="upload-card">
                {/* Drop zone — only covers the top area, not the button */}
                <div
                    className={`upload-dropzone${dragActive ? " drag-over" : ""}`}
                    onClick={openFilePicker}
                    onDragEnter={handleDrag}
                    onDragOver={handleDrag}
                    onDragLeave={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv"
                        onChange={handleChange}
                        style={{ display: "none" }}
                    />
                    <p className="upload-text">Drop your transaction CSV here</p>
                    <p className="upload-hint">
                        or click to browse — requires transaction_id, sender_id, receiver_id, amount, timestamp
                    </p>
                </div>
                {file && <p className="upload-filename">{file.name}</p>}
                <div className="upload-actions">
                    <button
                        className="btn-analyze"
                        onClick={onAnalyze}
                        disabled={loading || !file}
                    >
                        {loading ? (
                            <>
                                <span className="spinner" />
                                Analyzing...
                            </>
                        ) : (
                            "Analyze"
                        )}
                    </button>
                    {error && <span className="error-msg">{error}</span>}
                </div>
            </div>
        </div>
    );
}
