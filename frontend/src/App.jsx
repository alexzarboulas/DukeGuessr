import { useState, useCallback, useEffect, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5001";

const LANDMARK_META = {
  "Duke Chapel":           { key: "chapel" },
  "Main Quad":             { key: "main_quad" },
  "Perkins Library":       { key: "perkins" },
  "Campus Dr / Bus Stop":  { key: "bus_stop" },
  "Sarah P. Duke Gardens": { key: "gardens" },
  "Wannamaker Benches":    { key: "wannamaker_benches" },
  "Unknown Location":      { key: "other" },
};

const FACTS = {
  chapel:             "Duke Chapel's tower stands 210 feet tall and is visible from miles away. Its 5,000-pipe organ is one of the finest in the Southeast.",
  main_quad:          "The Main Quad's Gothic architecture was modeled after Oxford and Cambridge. Duke founder James Duke wanted it to feel timeless.",
  perkins:            "Perkins Library holds over 6 million volumes, making it the largest research library in the Research Triangle.",
  bus_stop:           "Duke Transit logs over 1 million passenger trips per year, connecting West and East Campus with routes across Durham.",
  gardens:            "Sarah P. Duke Gardens spans 55 acres and welcomes over 300,000 visitors annually, making it one of Duke's most visited spots.",
  wannamaker_benches: "Wannamaker dorm on East Campus was built in 1927 and housed Duke's first women students when the Women's College opened.",
  other:              "Duke's campus spans over 8,600 acres, one of the largest contiguous university campuses in the United States.",
};

const MODELS = [
  {
    key:   "finetuned_clip",
    label: "Fine-Tuned CLIP",
    badge: "STATE OF THE ART",
    accent: true,
    stats: [
      { label: "Test Accuracy", value: "100.0%" },
      { label: "Architecture", value: "ViT-B/32" },
      { label: "Training",     value: "30 epochs · lr=1e-5" },
    ],
    description:
      "CLIP ViT-B/32 with the image encoder fine-tuned on 350 Duke campus photos. Text encoder is frozen: 7 landmark paragraphs are pre-encoded into fixed 512-d anchors. Classification is cosine similarity between image embedding and each anchor, scaled by logit_scale, then softmax and cross-entropy loss. Adam, weight_decay=1e-4, early stopping patience=5.",
  },
  {
    key:   "zeroshot_clip",
    label: "Zero-Shot CLIP",
    badge: "BASELINE",
    accent: false,
    stats: [
      { label: "Test Accuracy", value: "32.1%" },
      { label: "Architecture", value: "ViT-B/32" },
      { label: "Training",     value: "None (zero-shot)" },
    ],
    description:
      "Same CLIP ViT-B/32, same text anchors, but zero training on Duke photos. Pretrained on 400M internet image-text pairs. Scores 32.1% vs. 49.1% for short labels, because CLIP's 77-token limit truncates the 200-word paragraphs and CLIP was pretrained on short captions, not long prose.",
  },
  {
    key:   "finetuned_vit",
    label: "Fine-Tuned ViT",
    badge: "COMPARISON",
    accent: false,
    stats: [
      { label: "Test Accuracy", value: "100.0%" },
      { label: "Architecture", value: "ViT-B/16" },
      { label: "Training",     value: "30 epochs · lr=1e-4" },
    ],
    description:
      "torchvision ViT-B/16 (ImageNet pretrained) with the head replaced by nn.Linear(768, 7). Standard CrossEntropyLoss over integer class labels with no language involved. Unlike CLIP, adding a new class requires retraining; there are no semantic text anchors, just learned visual features mapped to indices.",
  },
];

// ── SVG components ─────────────────────────────────────────────────────────

function CameraIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
      <circle cx="12" cy="13" r="4"/>
    </svg>
  );
}

function Spinner() {
  return (
    <div style={{
      width: 36, height: 36,
      border: "3px solid rgba(0,163,224,0.2)",
      borderTop: "3px solid #00a3e0",
      borderRadius: "50%",
      animation: "spin 0.8s linear infinite",
    }} />
  );
}

/* Nested Gothic arches — positioned absolutely in the header */
function GothicArch() {
  return (
    <svg
      aria-hidden="true"
      width="130"
      height="160"
      viewBox="0 0 130 160"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{
        position:      "absolute",
        right:         "1.75rem",
        top:           0,
        opacity:       0.07,
        pointerEvents: "none",
        userSelect:    "none",
      }}
    >
      <path d="M8,160 L8,75 Q8,8 65,8 Q122,8 122,75 L122,160"
            stroke="white" strokeWidth="3" />
      <path d="M20,160 L20,80 Q20,24 65,24 Q110,24 110,80 L110,160"
            stroke="white" strokeWidth="2" opacity="0.65" />
      <path d="M32,160 L32,88 Q32,42 65,42 Q98,42 98,88 L98,160"
            stroke="white" strokeWidth="1.5" opacity="0.38" />
      <path d="M44,160 L44,95 Q44,60 65,60 Q86,60 86,95 L86,160"
            stroke="white" strokeWidth="1" opacity="0.18" />
    </svg>
  );
}

/* Replaces the 💡 emoji — clean SVG info icon */
function InfoIcon() {
  return (
    <svg
      width="15" height="15" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round"
      style={{ flexShrink: 0, marginTop: 2, color: "rgba(0,163,224,0.65)" }}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  );
}

// ── Result panel ───────────────────────────────────────────────────────────

function ResultPanel({ result, isVisible, showFact }) {
  const [barWidths, setBarWidths] = useState([]);

  useEffect(() => {
    if (!result) return;
    setBarWidths(result.all_predictions.map(() => 0));
    const t = setTimeout(() => {
      setBarWidths(result.all_predictions.map(p => p.confidence * 100));
    }, 80);
    return () => clearTimeout(t);
  }, [result]);

  if (!result) return null;

  const topKey = LANDMARK_META[result.landmark]?.key || "other";
  const fact   = FACTS[topKey];

  return (
    <div style={{
      ...s.results,
      opacity:   isVisible ? 1 : 0,
      transform: isVisible ? "translateY(0)" : "translateY(24px)",
      transition: "opacity 0.4s ease, transform 0.4s ease",
    }}>
      {result.is_other ? (
        <div style={s.otherCard}>
          <div style={s.otherQ}>?</div>
          <p style={s.otherMsg}>This doesn't match any known Duke landmark.</p>
          <p style={s.factText}>{FACTS.other}</p>
        </div>
      ) : (
        <>
          <div style={s.predictedLabel}>PREDICTED LOCATION</div>

          {/* key forces remount → re-triggers the CSS animation on every new prediction */}
          <div
            key={`${result.landmark}-${result.confidence}`}
            className="landmark-name"
            style={s.predictedName}
          >
            {result.landmark}
          </div>

          <div style={s.confidenceBadge}>
            {(result.confidence * 100).toFixed(1)}% confidence
          </div>

          <div style={s.barsSection}>
            {result.all_predictions.map((p, i) => {
              const isTop = p.landmark === result.landmark;
              const w = barWidths[i] ?? 0;
              return (
                <div key={p.landmark} style={s.barRow}>
                  <span
                    className="bar-label"
                    style={{ color: isTop ? "#00a3e0" : "rgba(255,255,255,0.55)" }}
                  >
                    {p.landmark}
                  </span>
                  <div style={s.barTrack}>
                    <div style={{
                      ...s.barFill,
                      width: `${w}%`,
                      background: isTop
                        ? "linear-gradient(90deg, #003087, #00a3e0)"
                        : "#1e3a5f",
                      boxShadow: isTop ? "0 0 10px rgba(0,163,224,0.6)" : "none",
                      transition: "width 0.65s cubic-bezier(0.4,0,0.2,1)",
                    }} />
                  </div>
                  <span style={{
                    ...s.barPct,
                    color: isTop ? "#00a3e0" : "rgba(255,255,255,0.4)",
                  }}>
                    {(p.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>

          {showFact && fact && (
            <div style={s.factCard}>
              <InfoIcon />
              <p style={s.factText}>{fact}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────

export default function App() {
  const [allResults, setAllResults]       = useState(null);
  const [activeModel, setActiveModel]     = useState("finetuned_clip");
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);
  const [brightnessMap, setBrightnessMap] = useState(null);
  const [resultVisible, setResultVisible] = useState(false);
  const [dropHover, setDropHover]         = useState(false);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    setError(null);
    setAllResults(null);
    setBrightnessMap(null);
    setResultVisible(false);
    setLoading(true);
    setActiveModel("finetuned_clip");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setAllResults(data);
      setBrightnessMap(data.finetuned_clip?.brightness_map ?? null);
      setTimeout(() => setResultVisible(true), 80);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDropHover(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const activeResult = allResults?.[activeModel] ?? null;

  return (
    <div style={s.page}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header style={s.header} className="page-header">
        <div style={s.headerInner}>
          <div>
            <h1 style={s.title}>DUKEGUESSR</h1>
            <div style={s.titleRule} />
            <p style={s.subtitle}>Where on Duke's campus is this?</p>
          </div>
        </div>
        <GothicArch />
      </header>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <main className="page-main">

        {/* Upload zone — always shows upload UI, never an image preview */}
        <div
          style={{
            ...s.dropzone,
            ...(dropHover ? s.dropzoneHover   : {}),
            ...(loading   ? s.dropzoneLoading : {}),
          }}
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDropHover(true); }}
          onDragLeave={() => setDropHover(false)}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <div style={s.uploadPrompt}>
            <div style={s.cameraIcon}><CameraIcon /></div>
            <p style={s.dropLabel}>
              {allResults ? "Upload another photo" : "Drop a Duke campus photo to identify it"}
            </p>
            <p style={s.dropSub}>or click to browse</p>
          </div>
        </div>

        <input
          id="fileInput" type="file" accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {/* Loading */}
        {loading && (
          <div style={s.loadingRow}>
            <Spinner />
            <span style={s.loadingText}>Running all three models…</span>
          </div>
        )}

        {/* Error */}
        {error && <p style={s.errorText}>{error}</p>}

        {/* Model tabs + results */}
        {allResults && !loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

            {/* Brightness map — shown above tabs, generated from fine-tuned CLIP attention */}
            {brightnessMap && (
              <div style={s.brightnessCard}>
                <div style={s.brightnessHeader}>
                  <span style={s.brightnessTitle}>ATTENTION MAP</span>
                  <span style={s.brightnessSub}>Fine-Tuned CLIP · what the model focused on</span>
                </div>
                <img
                  src={`data:image/png;base64,${brightnessMap}`}
                  alt="CLIP attention brightness map"
                  style={s.brightnessImg}
                />
                <p style={s.brightnessExplain}>
                  Brighter areas show where the model concentrated its attention when identifying this location. Each bright region corresponds to a patch of the image that most influenced the prediction. Darker areas had less effect on the result.
                </p>
              </div>
            )}

            {/* Tab bar — horizontal scroll, no wrapping */}
            <div className="tab-bar">
              {MODELS.map((m) => (
                <button
                  key={m.key}
                  className="tab-btn"
                  style={{
                    ...s.tab,
                    ...(activeModel === m.key
                      ? (m.accent ? s.tabActiveAccent : s.tabActive)
                      : {}),
                  }}
                  onClick={() => setActiveModel(m.key)}
                >
                  {m.accent && activeModel === m.key && (
                    <span style={s.tabBadge}>{m.badge}</span>
                  )}
                  <span style={s.tabLabel}>{m.label}</span>
                  {allResults[m.key] && (
                    <span style={{
                      ...s.tabConf,
                      color: activeModel === m.key && m.accent
                        ? "#00a3e0"
                        : "rgba(255,255,255,0.4)",
                    }}>
                      {(allResults[m.key].confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Model description card */}
            {(() => {
              const m = MODELS.find(x => x.key === activeModel);
              return (
                <div style={s.modelDescCard}>
                  <div style={s.modelStatsRow}>
                    {m.stats.map(stat => (
                      <div key={stat.label} style={s.statChip}>
                        <span style={s.statLabel}>{stat.label}</span>
                        <span style={{
                          ...s.statValue,
                          color: m.accent ? "#00a3e0" : "rgba(255,255,255,0.85)",
                        }}>
                          {stat.value}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="model-desc-para" style={s.modelDescPara}>
                    {m.description}
                  </p>
                </div>
              );
            })()}

            {/* Result panel — key forces remount on tab switch, re-animating bars */}
            <ResultPanel
              key={activeModel}
              result={activeResult}
              isVisible={resultVisible}
              showFact={activeModel === "finetuned_clip"}
            />
          </div>
        )}
      </main>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────
const s = {
  page: {
    minHeight: "100vh",
    background: "transparent",
    display: "flex",
    flexDirection: "column",
  },

  header: {
    background: "linear-gradient(135deg, #001a4d 0%, #003087 100%)",
    borderBottom: "1px solid rgba(0,163,224,0.3)",
    padding: "1.75rem 2rem",
  },
  headerInner: {
    maxWidth: 720,
    margin: "0 auto",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: {
    margin: 0,
    fontSize: "clamp(1.8rem, 5vw, 2.8rem)",
    fontWeight: 900,
    letterSpacing: "0.12em",
    color: "#fff",
  },
  titleRule: {
    height: 2,
    width: "100%",
    background: "linear-gradient(90deg, #00a3e0, transparent)",
    boxShadow: "0 0 8px rgba(0,163,224,0.8)",
    margin: "0.4rem 0 0.6rem",
    borderRadius: 2,
  },
  subtitle: {
    margin: 0,
    fontSize: "0.9rem",
    color: "rgba(255,255,255,0.65)",
    letterSpacing: "0.04em",
    fontWeight: 400,
  },

  dropzone: {
    border: "2px dashed rgba(0,163,224,0.35)",
    borderRadius: 12,
    minHeight: 110,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(0,163,224,0.04)",
    boxShadow: "0 0 18px rgba(0,163,224,0.12)",
    transition: "box-shadow 0.2s, border-color 0.2s",
  },
  dropzoneHover: {
    borderColor: "rgba(0,163,224,0.8)",
    boxShadow: "0 0 32px rgba(0,163,224,0.4)",
    background: "rgba(0,163,224,0.08)",
  },
  dropzoneLoading: {
    animation: "borderGlow 1.5s ease-in-out infinite",
  },
  uploadPrompt: {
    textAlign: "center",
    padding: "1.25rem 2rem",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "0.35rem",
  },
  cameraIcon: { color: "rgba(0,163,224,0.5)", marginBottom: "0.2rem" },
  dropLabel:  { margin: 0, fontSize: "0.9rem", fontWeight: 500, color: "rgba(255,255,255,0.75)" },
  dropSub:    { margin: 0, fontSize: "0.75rem", color: "rgba(255,255,255,0.3)" },

  // Brightness map card
  brightnessCard: {
    borderRadius: 12,
    overflow: "hidden",
    border: "1px solid rgba(0,163,224,0.2)",
    background: "#0d1525",
  },
  brightnessHeader: {
    padding: "0.55rem 1rem",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
  },
  brightnessTitle: {
    fontSize: "0.62rem",
    letterSpacing: "0.14em",
    color: "rgba(0,163,224,0.75)",
    fontWeight: 700,
  },
  brightnessSub: {
    fontSize: "0.65rem",
    color: "rgba(255,255,255,0.3)",
    fontWeight: 400,
  },
  brightnessImg: {
    width: "100%",
    display: "block",
    maxHeight: 420,
    objectFit: "contain",
    background: "#000",
  },
  brightnessExplain: {
    margin: 0,
    padding: "0.7rem 1rem",
    fontSize: "0.75rem",
    color: "rgba(255,255,255,0.4)",
    lineHeight: 1.6,
    fontStyle: "italic",
    borderTop: "1px solid rgba(255,255,255,0.05)",
  },

  loadingRow:  { display: "flex", alignItems: "center", justifyContent: "center", gap: "0.75rem" },
  loadingText: { color: "rgba(0,163,224,0.8)", fontSize: "0.9rem", letterSpacing: "0.06em", fontWeight: 500 },
  errorText:   { color: "#f87171", textAlign: "center", fontSize: "0.875rem", margin: 0 },

  // Tabs
  tab: {
    background: "#111827",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    padding: "0.65rem 1rem",
    cursor: "pointer",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "0.2rem",
    transition: "border-color 0.2s, background 0.2s",
    color: "#fff",
  },
  tabActive: {
    background: "#1a2640",
    borderColor: "rgba(255,255,255,0.25)",
  },
  tabActiveAccent: {
    background: "rgba(0,163,224,0.1)",
    borderColor: "#00a3e0",
    boxShadow: "0 0 14px rgba(0,163,224,0.2)",
  },
  tabBadge: {
    fontSize: "0.55rem",
    letterSpacing: "0.14em",
    color: "#00a3e0",
    fontWeight: 700,
  },
  tabLabel: {
    fontSize: "0.78rem",
    fontWeight: 600,
    whiteSpace: "nowrap",
  },
  tabConf: {
    fontSize: "0.85rem",
    fontWeight: 700,
    fontVariantNumeric: "tabular-nums",
  },

  // Model description card
  modelDescCard: {
    background: "#111827",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 12,
    padding: "1.1rem 1.4rem",
    display: "flex",
    flexDirection: "column",
    gap: "0.9rem",
  },
  modelStatsRow: { display: "flex", gap: "0.6rem", flexWrap: "wrap" },
  statChip: {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 8,
    padding: "0.35rem 0.75rem",
    display: "flex",
    flexDirection: "column",
    gap: "0.1rem",
  },
  statLabel: {
    fontSize: "0.6rem",
    letterSpacing: "0.12em",
    color: "rgba(255,255,255,0.35)",
    fontWeight: 600,
    textTransform: "uppercase",
  },
  statValue: {
    fontSize: "0.82rem",
    fontWeight: 700,
    fontVariantNumeric: "tabular-nums",
  },
  modelDescPara: {
    margin: 0,
    fontSize: "0.78rem",
    color: "rgba(255,255,255,0.5)",
    lineHeight: 1.65,
  },

  // Results
  results:        { display: "flex", flexDirection: "column", gap: "1rem" },
  predictedLabel: {
    fontSize: "0.7rem",
    letterSpacing: "0.18em",
    color: "rgba(0,163,224,0.7)",
    fontWeight: 600,
  },
  predictedName: {
    fontFamily: "'Playfair Display', Georgia, 'Times New Roman', serif",
    fontSize: "clamp(1.7rem, 5.5vw, 2.5rem)",
    fontWeight: 800,
    color: "#fff",
    letterSpacing: "-0.01em",
    lineHeight: 1.1,
  },
  confidenceBadge: {
    display: "inline-block",
    background: "rgba(0,163,224,0.12)",
    border: "1px solid rgba(0,163,224,0.3)",
    borderRadius: 20,
    padding: "0.25rem 0.85rem",
    fontSize: "0.8rem",
    color: "#00a3e0",
    fontWeight: 600,
    letterSpacing: "0.04em",
    width: "fit-content",
  },

  barsSection: {
    background: "#111827",
    borderRadius: 12,
    padding: "1.25rem 1.5rem",
    border: "1px solid rgba(255,255,255,0.06)",
    display: "flex",
    flexDirection: "column",
    gap: "0.7rem",
  },
  barRow:  { display: "flex", alignItems: "center", gap: "0.75rem" },
  barTrack: {
    flex: 1,
    height: 8,
    background: "rgba(255,255,255,0.06)",
    borderRadius: 4,
    overflow: "hidden",
  },
  barFill: { height: "100%", borderRadius: 4, minWidth: 2 },
  barPct:  {
    width: 44,
    fontSize: "0.75rem",
    textAlign: "right",
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    flexShrink: 0,
  },

  factCard: {
    background: "rgba(0,163,224,0.06)",
    border: "1px solid rgba(0,163,224,0.18)",
    borderRadius: 10,
    padding: "0.9rem 1.1rem",
    display: "flex",
    gap: "0.65rem",
    alignItems: "flex-start",
  },
  factText: {
    margin: 0,
    fontSize: "0.82rem",
    color: "rgba(255,255,255,0.6)",
    lineHeight: 1.6,
    fontStyle: "italic",
  },

  otherCard: {
    background: "rgba(251,191,36,0.07)",
    border: "1px solid rgba(251,191,36,0.25)",
    borderRadius: 14,
    padding: "2rem",
    textAlign: "center",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "0.5rem",
  },
  otherQ:   { fontSize: "3rem", color: "#f59e0b", fontWeight: 900, lineHeight: 1 },
  otherMsg: { margin: 0, fontSize: "1rem", color: "rgba(255,255,255,0.7)", fontWeight: 500 },
};
