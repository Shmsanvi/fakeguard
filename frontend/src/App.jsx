// frontend/src/App.jsx
import { useState, useRef } from "react";

const API_BASE = "http://localhost:8000";

const VERDICT_CONFIG = {
  REAL: {
    label: "REAL",
    color: "#1a7a4a",
    bg: "#e8f7ef",
    border: "#a3d9b8",
    bar: "#2ecc71",
  },
  FAKE: {
    label: "FAKE",
    color: "#b81c1c",
    bg: "#fdecea",
    border: "#f5b8b8",
    bar: "#e74c3c",
  },
  UNCERTAIN: {
    label: "UNCERTAIN",
    color: "#7d5a00",
    bg: "#fdf6e3",
    border: "#f0d080",
    bar: "#f39c12",
  },
};

export default function App() {
  const [mode, setMode] = useState("url"); // "url" | "upload"
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [headline, setHeadline] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef();

  const handleImageSelect = (file) => {
    if (!file) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      let res;
      if (mode === "url") {
        if (!url.trim()) throw new Error("Please enter a URL.");
        res = await fetch(`${API_BASE}/analyze/url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: url.trim(), include_heatmap: true }),
        });
      } else {
        if (!text.trim()) throw new Error("Please enter article text.");
        const form = new FormData();
        form.append("text", text.trim());
        form.append("headline", headline.trim());
        form.append("include_heatmap", "true");
        if (imageFile) form.append("image", imageFile);
        res = await fetch(`${API_BASE}/analyze/upload`, { method: "POST", body: form });
      }

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed.");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const cfg = result ? VERDICT_CONFIG[result.verdict] : null;

  return (
    <div style={{ minHeight: "100vh", background: "#f5f4f0", fontFamily: "'Georgia', serif" }}>
      {/* Header */}
      <div style={{ background: "#1a1a2e", color: "#fff", padding: "20px 40px", display: "flex", alignItems: "center", gap: 16 }}>
        <div style={{ width: 36, height: 36, background: "#e74c3c", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 18 }}>F</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 20, letterSpacing: 1 }}>FakeGuard</div>
          <div style={{ fontSize: 12, color: "#aaa", fontFamily: "monospace", letterSpacing: 2 }}>HYBRID MISINFORMATION DETECTOR</div>
        </div>
      </div>

      <div style={{ maxWidth: 860, margin: "40px auto", padding: "0 24px" }}>

        {/* Mode toggle */}
        <div style={{ display: "flex", background: "#e0dfd8", borderRadius: 10, padding: 4, marginBottom: 28, width: "fit-content" }}>
          {["url", "upload"].map((m) => (
            <button key={m} onClick={() => { setMode(m); setResult(null); setError(null); }}
              style={{
                padding: "8px 28px", borderRadius: 8, border: "none", cursor: "pointer",
                background: mode === m ? "#1a1a2e" : "transparent",
                color: mode === m ? "#fff" : "#555",
                fontFamily: "monospace", fontSize: 13, fontWeight: 600, letterSpacing: 1,
                transition: "all .2s",
              }}>
              {m === "url" ? "ARTICLE URL" : "PASTE TEXT"}
            </button>
          ))}
        </div>

        {/* Input card */}
        <div style={{ background: "#fff", borderRadius: 16, padding: 32, boxShadow: "0 2px 12px rgba(0,0,0,0.07)", marginBottom: 24 }}>

          {mode === "url" ? (
            <div>
              <label style={labelStyle}>Article URL</label>
              <input value={url} onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/news/article"
                style={inputStyle}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
              <div style={{ fontSize: 12, color: "#888", marginTop: 6 }}>
                Paste any news article URL — we'll scrape the text and headline image automatically.
              </div>
            </div>
          ) : (
            <div>
              <label style={labelStyle}>Headline (optional)</label>
              <input value={headline} onChange={(e) => setHeadline(e.target.value)}
                placeholder="Article headline..."
                style={{ ...inputStyle, marginBottom: 16 }}
              />
              <label style={labelStyle}>Article body</label>
              <textarea value={text} onChange={(e) => setText(e.target.value)}
                placeholder="Paste the full article text here..."
                style={{ ...inputStyle, minHeight: 140, resize: "vertical" }}
              />
              <div style={{ marginTop: 16 }}>
                <label style={labelStyle}>Image (optional)</label>
                <div
                  onClick={() => fileRef.current.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); handleImageSelect(e.dataTransfer.files[0]); }}
                  style={{
                    border: "2px dashed #ccc", borderRadius: 10, padding: 24, textAlign: "center",
                    cursor: "pointer", color: "#888", fontSize: 14, background: "#fafaf8",
                    transition: "border-color .2s",
                  }}>
                  {imagePreview
                    ? <img src={imagePreview} alt="preview" style={{ maxHeight: 160, borderRadius: 8 }} />
                    : "Click or drag an image here"}
                </div>
                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                  onChange={(e) => handleImageSelect(e.target.files[0])} />
              </div>
            </div>
          )}

          <button onClick={handleSubmit} disabled={loading}
            style={{
              marginTop: 24, padding: "12px 36px", background: loading ? "#888" : "#1a1a2e",
              color: "#fff", border: "none", borderRadius: 10, fontSize: 15, fontWeight: 700,
              cursor: loading ? "not-allowed" : "pointer", letterSpacing: 1,
              fontFamily: "monospace", transition: "background .2s",
            }}>
            {loading ? "ANALYZING..." : "ANALYZE"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "#fdecea", border: "1px solid #f5b8b8", borderRadius: 10, padding: "12px 20px", color: "#b81c1c", marginBottom: 20 }}>
            {error}
          </div>
        )}

        {/* Results */}
        {result && cfg && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Verdict banner */}
            <div style={{ background: cfg.bg, border: `2px solid ${cfg.border}`, borderRadius: 16, padding: "28px 32px", display: "flex", alignItems: "center", gap: 24 }}>
              <div style={{ background: cfg.color, color: "#fff", borderRadius: 12, padding: "10px 22px", fontFamily: "monospace", fontWeight: 700, fontSize: 22, letterSpacing: 3 }}>
                {cfg.label}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, color: "#666", marginBottom: 6, fontFamily: "monospace" }}>CONFIDENCE</div>
                <div style={{ height: 10, background: "#e0e0e0", borderRadius: 5, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${result.confidence_pct}%`, background: cfg.bar, borderRadius: 5, transition: "width 1s ease" }} />
                </div>
                <div style={{ fontSize: 13, color: cfg.color, fontWeight: 700, marginTop: 4, fontFamily: "monospace" }}>
                  {result.confidence_pct.toFixed(1)}%
                </div>
              </div>
              <div style={{ fontSize: 12, color: "#aaa", textAlign: "right", fontFamily: "monospace" }}>
                {result.processing_time_ms}ms
              </div>
            </div>

            {/* Score breakdown */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
              <ScoreCard label="Text Score" value={result.fusion.text_score} desc="RoBERTa + metadata" />
              <ScoreCard label="Image Score" value={result.fusion.image_score} desc="ELA + FFT + CNN" />
              <ScoreCard label="Consistency" value={1 - result.fusion.consistency_score} desc="Image ↔ text match" invert />
            </div>

            {/* Text detail */}
            {result.text_analysis && (
              <DetailCard title="Text Pipeline">
                <Row label="RoBERTa" value={result.text_analysis.roberta_score} />
                <Row label="Metadata signals" value={result.text_analysis.metadata_score} />
                <Row label="Source credibility" value={1 - result.text_analysis.source_credibility} invert />
                <Row label="Sentiment mismatch" value={result.text_analysis.sentiment_mismatch} />
                <Row label="NER inconsistency" value={result.text_analysis.ner_consistency} />
                {result.text_analysis.top_tokens.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 12, color: "#888", fontFamily: "monospace", marginBottom: 6 }}>KEY TOKENS</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {result.text_analysis.top_tokens.map((tok, i) => (
                        <span key={i} style={{ background: "#f0ede8", borderRadius: 6, padding: "3px 10px", fontSize: 13, fontFamily: "monospace", color: "#333" }}>{tok}</span>
                      ))}
                    </div>
                  </div>
                )}
              </DetailCard>
            )}

            {/* Image detail */}
            {result.image_analysis && (
              <DetailCard title="Image Pipeline">
                <Row label="ELA (splice detection)" value={result.image_analysis.ela_score} />
                <Row label="FFT/DCT (GAN artifacts)" value={result.image_analysis.frequency_score} />
                <Row label="EfficientNet (AI-gen)" value={result.image_analysis.efficientnet_score} />
                {result.image_analysis.ela_heatmap_b64 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 12, color: "#888", fontFamily: "monospace", marginBottom: 6 }}>ELA HEATMAP (bright = suspect regions)</div>
                    <img src={`data:image/png;base64,${result.image_analysis.ela_heatmap_b64}`}
                      alt="ELA heatmap" style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid #e0e0e0" }} />
                  </div>
                )}
              </DetailCard>
            )}

            {/* Warnings */}
            {result.warnings.length > 0 && (
              <div style={{ background: "#fdf6e3", border: "1px solid #f0d080", borderRadius: 10, padding: "12px 20px" }}>
                <div style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: "#7d5a00", marginBottom: 6 }}>WARNINGS</div>
                {result.warnings.map((w, i) => <div key={i} style={{ fontSize: 13, color: "#7d5a00" }}>• {w}</div>)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ScoreCard({ label, value, desc, invert }) {
  const v = invert ? value : value;
  const isNeg = v < 0;
  const display = isNeg ? "N/A" : `${(v * 100).toFixed(0)}%`;
  const color = isNeg ? "#aaa" : v > 0.6 ? "#c0392b" : v > 0.4 ? "#e67e22" : "#27ae60";
  return (
    <div style={{ background: "#fff", borderRadius: 12, padding: "18px 20px", boxShadow: "0 1px 6px rgba(0,0,0,0.06)" }}>
      <div style={{ fontSize: 11, fontFamily: "monospace", color: "#aaa", letterSpacing: 1 }}>{label.toUpperCase()}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: "monospace", margin: "6px 0" }}>{display}</div>
      <div style={{ fontSize: 11, color: "#999" }}>{desc}</div>
    </div>
  );
}

function DetailCard({ title, children }) {
  return (
    <div style={{ background: "#fff", borderRadius: 12, padding: "22px 24px", boxShadow: "0 1px 6px rgba(0,0,0,0.06)" }}>
      <div style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: "#555", letterSpacing: 2, marginBottom: 16 }}>{title.toUpperCase()}</div>
      {children}
    </div>
  );
}

function Row({ label, value, invert }) {
  const isNeg = value < 0;
  const display = isNeg ? "N/A" : `${(value * 100).toFixed(1)}%`;
  const color = isNeg ? "#ccc" : value > 0.6 ? "#c0392b" : value > 0.35 ? "#e67e22" : "#27ae60";
  return (
    <div style={{ display: "flex", alignItems: "center", marginBottom: 8, gap: 10 }}>
      <div style={{ fontSize: 13, color: "#555", flex: 1 }}>{label}</div>
      {!isNeg && (
        <div style={{ width: 120, height: 5, background: "#eee", borderRadius: 3, overflow: "hidden" }}>
          <div style={{ width: `${value * 100}%`, height: "100%", background: color, borderRadius: 3 }} />
        </div>
      )}
      <div style={{ fontSize: 13, fontFamily: "monospace", fontWeight: 700, color, minWidth: 44, textAlign: "right" }}>{display}</div>
    </div>
  );
}

const labelStyle = { fontSize: 12, fontFamily: "monospace", color: "#888", letterSpacing: 1, display: "block", marginBottom: 6 };
const inputStyle = {
  width: "100%", padding: "10px 14px", border: "1px solid #ddd", borderRadius: 8,
  fontSize: 14, fontFamily: "inherit", boxSizing: "border-box",
  outline: "none", background: "#fafaf8",
};