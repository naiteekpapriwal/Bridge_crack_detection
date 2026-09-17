import { useCallback, useRef, useState, type CSSProperties, type DragEvent, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  CloudUpload,
  FileImage,
  Gauge,
  HardHat,
  Image as ImageIcon,
  Info,
  Loader2,
  Menu,
  Minus,
  PanelRight,
  RefreshCw,
  Ruler,
  ScanLine,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import { ThemeProvider } from "./contexts/ThemeContext";
import { Link, Route, Switch } from "wouter";

type DetectionResult = {
  max_width_mm?: number;
  avg_width_mm?: number;
  processed_image_base64?: string;
};

type ScanState = "idle" | "loading" | "success" | "error";

const API_ENDPOINT = "http://localhost:8000/detect";

function formatMetric(value: number | null) {
  return value === null || Number.isNaN(value) ? "—" : value.toFixed(2);
}

function base64ToDataUrl(value: string) {
  if (value.startsWith("data:")) return value;
  return `data:image/png;base64,${value}`;
}

function App() {
  return (
    <ThemeProvider defaultTheme="dark">
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: "#111923",
            color: "#eef6f0",
            border: "1px solid rgba(138, 255, 155, 0.18)",
          },
        }}
      />
      <Switch>
        <Route path="/" component={LandingPage} />
        <Route path="/console" component={BridgeDashboard} />
        <Route component={LandingPage} />
      </Switch>
    </ThemeProvider>
  );
}

function LandingPage() {
  return (
    <div className="landing-page">
      <header className="landing-nav">
        <Link href="/" className="landing-brand">
          <span className="brand-mark logo-emblem" aria-hidden="true"><svg viewBox="0 0 36 36" role="img"><path d="M18 3 30 10v14l-12 7L6 24V10l12-7Z" /><path d="m15 9 4 6-4 4 5 8" /><path d="M10 18h4M22 13h4" /></svg></span>
          <span><small>STRUCTURAL AI</small><strong>CRACK<span>/</span>SCAN</strong></span>
        </Link>
        <nav className="landing-links"><a href="#how-it-works">How it works</a><a href="#capabilities">Capabilities</a><Link href="/console">Open console <ArrowUpRight size={14} /></Link></nav>
      </header>

      <main>
        <section className="landing-hero">
          <div className="hero-copy">
            <div className="hero-eyebrow"><span className="hero-eyebrow-dot" /> COMPUTER VISION FOR STRUCTURAL SAFETY</div>
            <h1>See the smallest<br /><em>signs of stress.</em></h1>
            <p className="hero-description">Crack/Scan turns a single bridge surface image into a clear, measurable inspection in seconds — helping engineering teams move from visual uncertainty to confident action.</p>
            <div className="hero-actions"><Link href="/console" className="primary-cta">Start an inspection <ArrowUpRight size={16} /></Link><a href="#how-it-works" className="secondary-cta">Explore the workflow <ChevronRight size={15} /></a></div>
            <div className="hero-proof"><div className="proof-avatars"><span>AS</span><span>ME</span><span>+4</span></div><div><strong>Built for the field</strong><small>Clear output for every inspection team</small></div></div>
          </div>
          <div className="hero-visual" aria-label="Illustration of a bridge inspection scan">
            <div className="hero-visual-glow" />
            <div className="visual-grid" />
            <div className="visual-topline"><span>LIVE STRUCTURAL MODEL</span><span><span className="hero-live-dot" /> READY</span></div>
            <div className="bridge-scene photo-scene">
              <img src="https://images.unsplash.com/photo-1513360408428-02be46e29788?q=80&w=1400&auto=format&fit=crop" alt="Engineer inspecting a concrete bridge" />
              <div className="photo-tint" /><div className="scan-beam" /><div className="scan-target target-one"><span /><b>0.18 mm</b></div><div className="scan-target target-two"><span /><b>0.42 mm</b></div>
            </div>
            <div className="visual-footer"><span>FRAME 001 / CONCRETE SURFACE</span><strong>AI ANALYSIS</strong></div>
            <div className="hero-floating-card floating-status"><span className="floating-icon mint"><ShieldCheck size={15} /></span><span><small>STRUCTURAL STATUS</small><strong>Within threshold</strong></span></div>
            <div className="hero-floating-card floating-measure"><span className="floating-icon peach"><Ruler size={15} /></span><span><small>MAX CRACK WIDTH</small><strong>0.18 <i>mm</i></strong></span></div>
          </div>
        </section>

        <section className="metrics-strip"><div><strong>01</strong><span>Upload one image</span></div><div><strong>02</strong><span>Detect surface cracks</span></div><div><strong>03</strong><span>Measure with confidence</span></div><div className="metrics-strip-note"><span className="hero-live-dot" /> MODEL ONLINE / v1.0.4</div></section>

        <section className="landing-section" id="how-it-works">
          <div className="landing-section-heading"><div className="hero-eyebrow"><span className="kicker-line" /> THE WORKFLOW</div><h2>From image to<br /><em>inspection signal.</em></h2></div>
          <div className="workflow-cards"><div className="workflow-card"><span className="workflow-number">01</span><CloudUpload size={21} /><h3>Upload</h3><p>Drop a concrete surface image into the inspection console. No specialist setup required.</p></div><div className="workflow-card highlighted"><span className="workflow-number">02</span><ScanLine size={21} /><h3>Analyze</h3><p>The vision model segments crack geometry and measures its width against your scale.</p></div><div className="workflow-card"><span className="workflow-number">03</span><ShieldCheck size={21} /><h3>Act</h3><p>Review a clear status and quantitative output that helps prioritize your next step.</p></div></div>
        </section>

        <section className="capability-section" id="capabilities"><div><div className="hero-eyebrow"><span className="kicker-line" /> BUILT FOR DECISIONS</div><h2>Quietly precise.<br /><em>Operationally clear.</em></h2></div><div className="capability-copy"><p>Crack/Scan is designed for the first pass: the moment a field image needs to become a useful engineering signal. The interface keeps the important things visible — image context, measured width, and threshold status.</p><Link href="/console" className="text-link">Open the inspection console <ArrowUpRight size={15} /></Link></div></section>
      </main>
      <footer className="landing-footer"><span>CRACK/SCAN <i>STRUCTURAL VISION LAB</i></span><span>MEASUREMENT ENGINEERING / 2026</span></footer>
    </div>
  );
}

function BridgeDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [processedPreview, setProcessedPreview] = useState<string | null>(null);
  const [scanState, setScanState] = useState<ScanState>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [maxWidth, setMaxWidth] = useState<number | null>(null);
  const [avgWidth, setAvgWidth] = useState<number | null>(null);
  const [scale, setScale] = useState(0.05);
  const [isDragging, setIsDragging] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const analyzeImage = useCallback(
    async (selectedFile: File) => {
      setFile(selectedFile);
      setOriginalPreview(URL.createObjectURL(selectedFile));
      setProcessedPreview(null);
      setMaxWidth(null);
      setAvgWidth(null);
      setErrorMessage("");
      setScanState("loading");

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("pixel_to_mm_ratio", String(scale));
      formData.append("use_mock", "false"); // Using real model weights

      try {
        const response = await fetch(API_ENDPOINT, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Detection service returned ${response.status}`);
        }

        const result = await response.json();
        const measurements = result.detections?.[0]?.measurements;
        
        setMaxWidth(typeof measurements?.max_width_mm === "number" ? measurements.max_width_mm : null);
        setAvgWidth(typeof measurements?.mean_width_mm === "number" ? measurements.mean_width_mm : null);
        
        // The backend directly returns a data URI now, so we can use it directly
        setProcessedPreview(result.annotated_image || null);
        setScanState("success");
        toast.success("Analysis complete", { description: "Crack measurements are ready to review." });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to reach the detection service.";
        setErrorMessage(message.includes("Failed to fetch") ? "API offline — check that localhost:8000 is running." : message);
        setScanState("error");
        toast.error("Analysis unavailable", { description: "The image is loaded, but the detection API could not be reached." });
      }
    },
    [scale],
  );

  const handleFile = useCallback(
    (nextFile?: File) => {
      if (!nextFile) return;
      if (!nextFile.type.startsWith("image/")) {
        toast.error("Unsupported file", { description: "Please upload a JPG, PNG, WEBP, or TIFF image." });
        return;
      }
      void analyzeImage(nextFile);
    },
    [analyzeImage],
  );

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const statusIsWarning = maxWidth !== null && maxWidth >= 0.4;
  const statusLabel = scanState === "success" ? (statusIsWarning ? "Warning" : "Safe") : scanState === "error" ? "Needs attention" : "Awaiting scan";
  const statusTone = scanState === "success" ? (statusIsWarning ? "warning" : "safe") : scanState === "error" ? "warning" : "neutral";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenuOpen ? "sidebar-open" : ""}`}>
        <div className="brand-block">
          <div className="brand-mark"><ScanLine size={19} strokeWidth={2.4} /></div>
          <div>
            <p className="eyebrow">STRUCTURAL AI</p>
            <p className="brand-name">CRACK<span>/</span>SCAN</p>
          </div>
          <button className="mobile-close" onClick={() => setMobileMenuOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <div className="sidebar-rule" />
        <div className="nav-label">WORKSPACE</div>
        <nav className="nav-list">
          <button className="nav-item active"><Gauge size={17} /><span>Inspection console</span><ChevronRight size={15} className="nav-arrow" /></button>
          <button className="nav-item" onClick={() => toast.info("History is coming soon.")}><Activity size={17} /><span>Scan history</span><span className="nav-count">00</span></button>
          <button className="nav-item" onClick={() => toast.info("Calibration tools are coming soon.")}><Ruler size={17} /><span>Calibration</span></button>
        </nav>

        <div className="nav-label nav-label-spaced">SYSTEM</div>
        <nav className="nav-list">
          <button className="nav-item" onClick={() => toast.info("API is configured for localhost:8000.")}><Zap size={17} /><span>API connection</span><span className="connection-dot" /></button>
          <button className="nav-item" onClick={() => toast.info("Help center is coming soon.")}><CircleHelp size={17} /><span>Documentation</span></button>
        </nav>

        <div className="sidebar-spacer" />
        <div className="system-card">
          <div className="system-card-top"><span className="live-dot" /><span>MODEL STATUS</span><span className="system-version">v1.0.4</span></div>
          <p>Detection model online</p>
          <div className="system-bar"><span /></div>
          <div className="system-meta"><span>LATENCY</span><strong>— ms</strong></div>
        </div>
        <div className="sidebar-footer"><span>BRIDGE INSPECTION LAB</span><span>© 2026</span></div>
      </aside>

      {mobileMenuOpen && <button className="sidebar-backdrop" onClick={() => setMobileMenuOpen(false)} aria-label="Close menu" />}

      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileMenuOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>CONSOLE</span><ChevronRight size={14} /><strong>NEW INSPECTION</strong></div>
          <div className="topbar-actions">
            <div className="topbar-status"><span className="live-dot" /> <span>API READY</span></div>
            <div className="avatar">JD</div>
          </div>
        </header>

        <div className="content-wrap">
          <section className="page-intro fade-up">
            <div>
              <div className="section-kicker"><span className="kicker-line" /> LIVE WORKSPACE</div>
              <h1>Bridge surface<br /><em>inspection.</em></h1>
              <p className="intro-copy">Upload a concrete surface image to detect and measure visible cracks with the structural vision model.</p>
            </div>
            <div className="intro-meta"><span className="meta-label">INSPECTION ID</span><strong>BRG-{new Date().getFullYear()}-0042</strong><span className="meta-sub">LOCAL SESSION / UNSAVED</span></div>
          </section>

          <div className="workspace-grid">
            <div className="workspace-main">
              {!file ? (
                <div
                  className={`upload-zone ${isDragging ? "is-dragging" : ""} fade-up delay-1`}
                  onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => inputRef.current?.click()}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
                >
                  <input ref={inputRef} type="file" accept="image/*" className="sr-only" onChange={(event) => handleFile(event.target.files?.[0])} />
                  <div className="upload-grid-glow" />
                  <div className="upload-icon"><UploadCloud size={28} strokeWidth={1.5} /></div>
                  <div className="upload-title">Drop bridge image here</div>
                  <p className="upload-subtitle">or <span>browse files</span> from your device</p>
                  <div className="upload-specs"><span><FileImage size={14} /> JPG / PNG / WEBP</span><span className="spec-divider" /><span>MAX 25 MB</span></div>
                  <div className="upload-corner corner-tl" /><div className="upload-corner corner-tr" /><div className="upload-corner corner-bl" /><div className="upload-corner corner-br" />
                </div>
              ) : (
                <div className="analysis-card fade-up delay-1">
                  <div className="analysis-head">
                    <div><div className="section-kicker"><span className="kicker-line" /> IMAGE ANALYSIS</div><h2>Surface comparison</h2></div>
                    <div className="analysis-actions"><span className="file-name"><FileImage size={14} />{file.name}</span><button className="icon-button" onClick={() => { setFile(null); setOriginalPreview(null); setProcessedPreview(null); setScanState("idle"); }} aria-label="Remove image"><X size={16} /></button></div>
                  </div>
                  <div className="comparison-grid">
                    <ImagePane label="ORIGINAL IMAGE" image={originalPreview} />
                    <ImagePane label="PROCESSED IMAGE" image={processedPreview} loading={scanState === "loading"} emptyText={scanState === "error" ? "PROCESSING UNAVAILABLE" : "AWAITING MODEL OUTPUT"} />
                  </div>
                  {scanState === "error" && <div className="inline-error"><AlertTriangle size={16} /><span>{errorMessage}</span><button onClick={() => file && void analyzeImage(file)}><RefreshCw size={14} /> Retry</button></div>}
                  <div className="analysis-foot"><span><Info size={13} /> Images are processed locally via your configured API.</span><span className="frame-id">FRAME / 001</span></div>
                </div>
              )}

              <section className="results-section fade-up delay-2">
                <div className="section-heading"><div><div className="section-kicker"><span className="kicker-line" /> DETECTION OUTPUT</div><h2>Results panel</h2></div><div className={`scan-status ${statusTone}`}><span className="status-pip" />{statusLabel.toUpperCase()}</div></div>
                <div className="metric-grid">
                  <MetricCard label="STATUS" value={statusLabel} suffix={scanState === "success" ? (statusIsWarning ? "Review recommended" : "Within threshold") : "Upload an image to begin"} icon={statusIsWarning ? <AlertTriangle size={18} /> : <ShieldCheck size={18} />} tone={statusTone} />
                  <MetricCard label="MAX CRACK WIDTH" value={formatMetric(maxWidth)} unit="mm" suffix="Peak detected width" icon={<Activity size={18} />} tone="lime" />
                  <MetricCard label="AVERAGE WIDTH" value={formatMetric(avgWidth)} unit="mm" suffix="Across detected cracks" icon={<Minus size={18} />} tone="blue" />
                </div>
              </section>
            </div>

            <aside className="settings-panel fade-up delay-2">
              <div className="settings-head"><div className="settings-icon"><Settings2 size={18} /></div><div><div className="section-kicker"><span className="kicker-line" /> CONFIGURATION</div><h2>Scan settings</h2></div></div>
              <div className="settings-rule" />
              <div className="setting-block">
                <div className="setting-title-row"><label htmlFor="scale">Measurement scale</label><span className="help-icon" title="Millimeters represented by one image pixel"><CircleHelp size={14} /></span></div>
                <p className="setting-description">Pixel to millimeter ratio used by the detection model.</p>
                <div className="slider-value"><strong>{scale.toFixed(2)}</strong><span>mm / px</span></div>
                <input id="scale" type="range" min="0.01" max="0.20" step="0.01" value={scale} style={{ "--range-progress": `${((scale - 0.01) / 0.19) * 100}%` } as CSSProperties} onChange={(event) => setScale(Number(event.target.value))} />
                <div className="slider-labels"><span>0.01</span><span>0.20</span></div>
              </div>
              <div className="settings-rule" />
              <div className="setting-block compact"><div className="setting-title-row"><span>Detection threshold</span><span className="threshold-value">0.40 mm</span></div><p className="setting-description">Cracks above this width are flagged as a warning.</p><div className="threshold-track"><span /></div></div>
              <div className="settings-rule" />
              <div className="model-info"><div className="model-icon"><HardHat size={17} /></div><div><span className="model-label">ACTIVE MODEL</span><strong>Concrete-Vision / v1.0</strong><span className="model-sub">Crack segmentation + measurement</span></div></div>
              <button className="settings-action" onClick={() => toast.info("Advanced settings are coming soon.")}><SlidersHorizontal size={15} /> Advanced settings <ArrowUpRight size={14} /></button>
            </aside>
          </div>

          <footer className="content-footer"><span><span className="footer-pip" /> SYSTEM NOMINAL</span><span>MEASUREMENT ENGINEERING / STRUCTURAL VISION</span></footer>
        </div>
      </main>
    </div>
  );
}

function ImagePane({ label, image, loading, emptyText }: { label: string; image: string | null; loading?: boolean; emptyText?: string }) {
  return (
    <div className="image-pane">
      <div className="pane-label"><span>{label}</span><span className="pane-index">{label === "ORIGINAL IMAGE" ? "A" : "B"}</span></div>
      <div className={`image-frame ${loading ? "is-loading" : ""} ${!image ? "is-empty" : ""}`}>
        {image ? <img src={image} alt={label.toLowerCase()} /> : loading ? <div className="processing-state"><div className="processing-ring"><Loader2 size={25} /></div><strong>Analyzing surface</strong><span>Segmenting crack geometry...</span></div> : <div className="empty-processed"><div className="empty-diamond"><ImageIcon size={21} /></div><strong>{emptyText || "AWAITING IMAGE"}</strong><span>Model output will appear here</span></div>}
        <span className="frame-corner frame-tl" /><span className="frame-corner frame-tr" /><span className="frame-corner frame-bl" /><span className="frame-corner frame-br" />
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit, suffix, icon, tone }: { label: string; value: string; unit?: string; suffix: string; icon: ReactNode; tone: string }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <div className="metric-top"><span className="metric-label">{label}</span><span className="metric-icon">{icon}</span></div>
      <div className="metric-value">{value}{unit && <small>{unit}</small>}</div>
      <div className="metric-suffix">{suffix}</div>
    </div>
  );
}

export default App;
