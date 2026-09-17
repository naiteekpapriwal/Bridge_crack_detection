import { useCallback, useRef, useState, type CSSProperties, type DragEvent, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Calculator,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  CloudUpload,
  Download,
  Droplets,
  FileImage,
  Files,
  FileText,
  Gauge,
  HardHat,
  Image as ImageIcon,
  Info,
  Loader2,
  TrendingUp,
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
import { Link, Route, Switch, useLocation } from "wouter";

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
        <Route path="/console/:tab" component={BridgeDashboard} />
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
          <div className="hero-visual" style={{ background: 'transparent', border: 'none', boxShadow: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {/* Premium Dashboard UI Mockup */}
            <div style={{ width: '100%', height: '380px', background: '#0a0b10', borderRadius: 12, border: '1px solid #232533', boxShadow: '0 30px 60px rgba(0,0,0,0.15), 0 0 0 6px rgba(255,255,255,0.4)', overflow: 'hidden', display: 'flex', flexDirection: 'column', position: 'relative' }}>
              
              {/* Window chrome / header */}
              <div style={{ height: 44, borderBottom: '1px solid #1a1d27', display: 'flex', alignItems: 'center', padding: '0 16px', gap: 8, background: '#11121a' }}>
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#ff5f56' }} />
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#ffbd2e' }} />
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#27c93f' }} />
                <div style={{ margin: '0 auto', fontSize: 11, color: '#687282', fontFamily: 'monospace', letterSpacing: '0.05em' }}>/workspace/bridge_inspection_492.raw</div>
              </div>
              
              {/* Mockup body */}
              <div style={{ flex: 1, display: 'flex' }}>
                
                {/* Mockup sidebar */}
                <div style={{ width: 64, borderRight: '1px solid #1a1d27', display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 20, gap: 16, background: '#0a0b10' }}>
                  <div style={{ width: 28, height: 28, borderRadius: 6, background: 'rgba(157, 114, 255, 0.15)', border: '1px solid rgba(157,114,255,0.4)', display: 'grid', placeItems: 'center' }}>
                    <div style={{ width: 14, height: 14, border: '1.5px solid #9d72ff', borderRadius: 3 }} />
                  </div>
                  <div style={{ width: 28, height: 28, borderRadius: 6, background: '#151722' }} />
                  <div style={{ width: 28, height: 28, borderRadius: 6, background: '#151722' }} />
                </div>
                
                {/* Mockup main area */}
                <div style={{ flex: 1, padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
                      <div style={{ width: 42, height: 42, borderRadius: 8, background: 'linear-gradient(135deg, #9d72ff, #5c3bba)' }} />
                      <div>
                        <div style={{ width: 140, height: 8, background: '#232533', borderRadius: 4, marginBottom: 10 }} />
                        <div style={{ width: 90, height: 6, background: '#1a1d27', borderRadius: 3 }} />
                      </div>
                    </div>
                    <div style={{ width: 90, height: 30, borderRadius: 15, background: 'rgba(39, 201, 63, 0.1)', border: '1px solid rgba(39, 201, 63, 0.3)', color: '#27c93f', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, letterSpacing: '0.05em' }}>SAFE STATUS</div>
                  </div>
                  
                  {/* Analysis preview block */}
                  <div style={{ flex: 1, borderRadius: 8, border: '1px solid #232533', background: '#050508', position: 'relative', overflow: 'hidden' }}>
                    
                    {/* Grid */}
                    <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(#151722 1px, transparent 1px), linear-gradient(90deg, #151722 1px, transparent 1px)', backgroundSize: '30px 30px' }} />
                    
                    {/* Fake structural cracks on the dark canvas */}
                    <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
                      <path d="M 120 30 L 140 80 L 110 130 L 150 180" fill="none" stroke="#27c93f" strokeWidth="2.5" filter="drop-shadow(0 0 6px #27c93f)" />
                      <circle cx="125" cy="105" r="16" fill="none" stroke="rgba(39, 201, 63, 0.5)" strokeWidth="1" strokeDasharray="3,3" />
                      <text x="150" y="108" fill="#27c93f" fontSize="11" fontFamily="monospace">0.12mm</text>
                      
                      <path d="M 280 50 L 260 110 L 300 160 L 280 200" fill="none" stroke="#27c93f" strokeWidth="2.5" filter="drop-shadow(0 0 6px #27c93f)" />
                      <circle cx="280" cy="135" r="16" fill="none" stroke="rgba(39, 201, 63, 0.5)" strokeWidth="1" strokeDasharray="3,3" />
                      <text x="305" y="138" fill="#27c93f" fontSize="11" fontFamily="monospace">0.18mm</text>
                    </svg>
                    
                    {/* Animated scanning overlay */}
                    <div className="hero-scanner" style={{ position: 'absolute', top: 0, bottom: 0, width: 120, background: 'linear-gradient(90deg, transparent, rgba(157, 114, 255, 0.15))', borderRight: '2px solid #9d72ff', filter: 'drop-shadow(0 0 8px #9d72ff)' }} />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Inline CSS animation for the scanner */}
            <style dangerouslySetInnerHTML={{__html: `
              .hero-scanner {
                animation: scan-anim 3s infinite linear;
              }
              @keyframes scan-anim {
                0% { left: -120px; opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { left: 100%; opacity: 0; }
              }
            `}} />
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
  const [pipelineInfo, setPipelineInfo] = useState<{ preprocessing?: string; filtering?: string; raw_detections?: number; filtered_detections?: number; accepted_detections?: number } | null>(null);
  const [maxWidth, setMaxWidth] = useState<number | null>(null);
  const [avgWidth, setAvgWidth] = useState<number | null>(null);
  const [numCracks, setNumCracks] = useState(0);
  const [scale, setScale] = useState(0.05);
  const [threshold, setThreshold] = useState(0.40);
  const [isDragging, setIsDragging] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [location, setLocation] = useLocation();

  const currentTab = location.split("/").pop() || "console";

  const analyzeImage = useCallback(
    async (selectedFile: File) => {
      setFile(selectedFile);
      setOriginalPreview(URL.createObjectURL(selectedFile));
      setProcessedPreview(null);
      setMaxWidth(null);
      setAvgWidth(null);
      setNumCracks(0);
      setErrorMessage("");
      setPipelineInfo(null);
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
        const allDetections = result.detections || [];
        setNumCracks(allDetections.length);

        // Aggregate measurements across ALL detections
        if (allDetections.length > 0) {
          const maxW = Math.max(...allDetections.map((d: any) => d.measurements?.max_width_mm ?? 0));
          const avgW = allDetections.reduce((sum: number, d: any) => sum + (d.measurements?.mean_width_mm ?? 0), 0) / allDetections.length;
          setMaxWidth(maxW);
          setAvgWidth(avgW);
        } else {
          setMaxWidth(null);
          setAvgWidth(null);
        }
        
        // The backend directly returns a data URI now, so we can use it directly
        setProcessedPreview(result.annotated_image || null);
        setPipelineInfo(result.pipeline_info || null);
        setScanState("success");
        toast.success("Analysis complete", {
          description: result.pipeline_info
            ? `${result.pipeline_info.raw_detections ?? 0} raw → ${result.pipeline_info.filtered_detections ?? 0} filtered → ${result.pipeline_info.accepted_detections ?? 0} accepted`
            : "Crack measurements are ready to review.",
        });
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

  const statusIsWarning = maxWidth !== null && maxWidth >= threshold;
  const statusLabel = scanState === "success" ? (statusIsWarning ? `${numCracks} Crack${numCracks !== 1 ? "s" : ""} found` : "Safe") : scanState === "error" ? "Needs attention" : "Awaiting scan";
  const statusTone = scanState === "success" ? (statusIsWarning ? "warning" : "safe") : scanState === "error" ? "warning" : "neutral";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenuOpen ? "sidebar-open" : ""}`}>
        <div className="brand-block" onClick={() => setLocation("/")} style={{ cursor: "pointer" }} title="Return to home screen">
          <div className="brand-mark"><ScanLine size={19} strokeWidth={2.4} /></div>
          <div>
            <p className="eyebrow">STRUCTURAL AI</p>
            <p className="brand-name">CRACK<span>/</span>SCAN</p>
          </div>
          <button className="mobile-close" onClick={(e) => { e.stopPropagation(); setMobileMenuOpen(false); }} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <div className="sidebar-rule" />
        <div className="nav-label">WORKSPACE</div>
        <nav className="nav-list">
          <button className={`nav-item ${currentTab === "console" ? "active" : ""}`} onClick={() => setLocation("/console")}><Gauge size={17} /><span>Inspection console</span>{currentTab === "console" && <ChevronRight size={15} className="nav-arrow" />}</button>
          <button className={`nav-item ${currentTab === "batch" ? "active" : ""}`} onClick={() => setLocation("/console/batch")}><Files size={17} /><span>Batch processing</span></button>
          <button className={`nav-item ${currentTab === "estimator" ? "active" : ""}`} onClick={() => setLocation("/console/estimator")}><Calculator size={17} /><span>Repair estimator</span></button>
        </nav>

        <div className="nav-label nav-label-spaced">SYSTEM</div>
        <nav className="nav-list">
          <button className={`nav-item ${currentTab === "propagation" ? "active" : ""}`} onClick={() => setLocation("/console/propagation")}><TrendingUp size={17} /><span>Crack propagation</span></button>
          <button className={`nav-item ${currentTab === "report" ? "active" : ""}`} onClick={() => setLocation("/console/report")}><FileText size={17} /><span>Generate report</span></button>
        </nav>

        <div className="sidebar-spacer" />
        <div className="system-card">
          <div className="system-card-top"><span>ENVIRONMENT</span><span className="system-version">LIVE DATA</span></div>
          <p>Surface Temp: 24°C</p>
          <div className="system-meta" style={{ marginTop: 8 }}><span>THERMAL EXPANSION</span><strong>Nominal</strong></div>
        </div>
        <div className="sidebar-footer"><span>BRIDGE INSPECTION LAB</span><span>© 2026</span></div>
      </aside>

      {mobileMenuOpen && <button className="sidebar-backdrop" onClick={() => setMobileMenuOpen(false)} aria-label="Close menu" />}

      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileMenuOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>CONSOLE</span><ChevronRight size={14} /><strong>{currentTab === "console" ? "NEW INSPECTION" : currentTab.toUpperCase()}</strong></div>
          <div className="topbar-actions">
            <div className="topbar-status"><span className="live-dot" /> <span>API READY</span></div>
            <div className="avatar">JD</div>
          </div>
        </header>

        <div className="content-wrap">
          {currentTab === "console" && (
            <>
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
                {pipelineInfo && scanState === "success" && (
                  <div className="pipeline-info">
                    <div className="pipeline-info-row">
                      <span className="pipeline-stage"><ScanLine size={13} /> PREPROCESSING</span>
                      <span>{pipelineInfo.preprocessing ?? "—"}</span>
                    </div>
                    <div className="pipeline-info-row">
                      <span className="pipeline-stage"><Activity size={13} /> FILTERING</span>
                      <span>{pipelineInfo.filtering ?? "—"}</span>
                    </div>
                    <div className="pipeline-info-row">
                      <span className="pipeline-stage"><Gauge size={13} /> DETECTIONS</span>
                      <span>{pipelineInfo.raw_detections ?? 0} raw → {pipelineInfo.filtered_detections ?? 0} filtered → <strong>{pipelineInfo.accepted_detections ?? 0} accepted</strong></span>
                    </div>
                  </div>
                )}
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
              <div className="setting-block compact">
                <div className="setting-title-row"><label htmlFor="threshold">Detection threshold</label><span className="threshold-value">{threshold.toFixed(2)} mm</span></div>
                <p className="setting-description">Cracks above this width are flagged as a warning.</p>
                <input id="threshold" type="range" min="0.10" max="1.50" step="0.05" value={threshold} style={{ "--range-progress": `${((threshold - 0.10) / 1.40) * 100}%` } as CSSProperties} onChange={(event) => setThreshold(Number(event.target.value))} />
                <div className="slider-labels" style={{ marginTop: 8 }}><span>0.10</span><span>1.50</span></div>
              </div>
              <div className="settings-rule" />
              <div className="model-info" style={{ paddingBottom: 12 }}><div className="model-icon"><HardHat size={17} /></div><div><span className="model-label">ACTIVE MODEL</span><strong>Concrete-Vision / v1.0</strong><span className="model-sub">Crack segmentation + measurement</span></div></div>
              </aside>
            </div>
          </>)}

          {currentTab === "batch" && (
            <div className="placeholder-page fade-in">
              <Files size={48} className="placeholder-icon" />
              <h2>Drone & Batch Processing</h2>
              <p>Upload a folder or ZIP file containing multiple drone images to process an entire structure automatically.</p>
              <div className="upload-zone is-dimmed" style={{ marginTop: 24, maxWidth: 500, marginInline: "auto" }}>
                <div className="upload-icon"><CloudUpload size={28} /></div>
                <p>Drop ZIP file here</p>
                <button className="secondary-cta" style={{ marginTop: 12 }}>Select folder</button>
              </div>
            </div>
          )}

          {currentTab === "estimator" && (
            <div className="placeholder-page fade-in">
              <Calculator size={48} className="placeholder-icon" />
              <h2>Epoxy Repair Estimator</h2>
              <p>Calculate required sealant volume based on detected crack dimensions (width × length × depth).</p>
              <div className="estimator-mockup" style={{ marginTop: 24, textAlign: 'left', background: 'rgba(255,255,255,0.02)', padding: 24, borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)', maxWidth: 400, marginInline: "auto" }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}><span>Total Detected Length:</span> <strong>{file ? '3.2 meters' : '—'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}><span>Average Width:</span> <strong>{file && avgWidth ? `${formatMetric(avgWidth)} mm` : '—'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24, paddingBottom: 24, borderBottom: '1px solid rgba(255,255,255,0.1)' }}><span>Assumed Depth:</span> <strong>25.0 mm</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#9d72ff' }}><span>Required Injection Epoxy:</span> <strong>{file ? '285 mL' : '—'}</strong></div>
              </div>
            </div>
          )}

          {currentTab === "propagation" && (
            <div className="placeholder-page fade-in">
              <TrendingUp size={48} className="placeholder-icon" />
              <h2>Crack Propagation Analysis</h2>
              <p>Compare current crack dimensions against historical scans to measure structural deterioration over time.</p>
              
              <div style={{ display: 'flex', gap: 24, marginTop: 32, justifyContent: 'center' }}>
                <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, padding: 16, width: 220, textAlign: 'left', boxShadow: '0 2px 10px rgba(0,0,0,0.02)' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>PREVIOUS SCAN (2024)</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#0f172a' }}>0.35 <small style={{ fontSize: 13, color: '#64748b', fontWeight: 400 }}>mm</small></div>
                  <div style={{ fontSize: 13, marginTop: 4, color: '#64748b' }}>Max Width</div>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <TrendingUp size={24} style={{ color: '#e11d48', opacity: 0.8 }} />
                </div>

                <div style={{ background: '#fff1f2', border: '1px solid #fecdd3', borderRadius: 8, padding: 16, width: 220, textAlign: 'left' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#e11d48', marginBottom: 8 }}>CURRENT SCAN (2026)</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#9f1239' }}>0.42 <small style={{ fontSize: 13, color: '#e11d48', fontWeight: 400 }}>mm</small></div>
                  <div style={{ fontSize: 13, marginTop: 4, color: '#e11d48', fontWeight: 500 }}>+20% Growth Detected</div>
                </div>
              </div>
            </div>
          )}

          {currentTab === "report" && (
            <div className="placeholder-page fade-in" style={file ? { padding: 0, justifyContent: 'flex-start', background: 'transparent' } : {}}>
              {!file ? (
                <>
                  <FileText size={48} className="placeholder-icon" />
                  <h2>Compliance Report</h2>
                  <p>Generate a standardized structural inspection report (PDF/CSV) for maintenance records.</p>
                  <button className="primary-cta" style={{ marginTop: 24 }} disabled={true}><Download size={16}/> Run Inspection First</button>
                </>
              ) : (
                <div style={{ width: '100%', maxWidth: 800, margin: '0 auto', textAlign: 'left', paddingBottom: 60 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
                    <h2 style={{ margin: 0, fontSize: 20 }}>Generated Report</h2>
                    <button className="primary-cta" onClick={() => window.print()}><Download size={16}/> Download PDF</button>
                  </div>
                  
                  <div className="print-report" style={{ background: '#ffffff', color: '#000000', padding: 40, borderRadius: 8, boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
                    <div style={{ borderBottom: '2px solid #000', paddingBottom: 16, marginBottom: 24 }}>
                      <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, textTransform: 'uppercase' }}>Structural Inspection Report</h1>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, fontSize: 13, color: '#555' }}>
                        <span><strong>DATE:</strong> {new Date().toLocaleDateString()}</span>
                        <span><strong>INSPECTOR:</strong> CRACK/SCAN AI v1.0.4</span>
                      </div>
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 32 }}>
                      <div>
                        <h3 style={{ margin: '0 0 8px 0', fontSize: 14, textTransform: 'uppercase', color: '#666' }}>Scan Metrics</h3>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                          <tbody>
                            <tr style={{ borderBottom: '1px solid #ddd' }}><td style={{ padding: '8px 0' }}>Max Crack Width</td><td style={{ textAlign: 'right', fontWeight: 'bold' }}>{formatMetric(maxWidth)} mm</td></tr>
                            <tr style={{ borderBottom: '1px solid #ddd' }}><td style={{ padding: '8px 0' }}>Avg Crack Width</td><td style={{ textAlign: 'right', fontWeight: 'bold' }}>{formatMetric(avgWidth)} mm</td></tr>
                            <tr style={{ borderBottom: '1px solid #ddd' }}><td style={{ padding: '8px 0' }}>Total Cracks Detected</td><td style={{ textAlign: 'right', fontWeight: 'bold' }}>{numCracks}</td></tr>
                            <tr style={{ borderBottom: '1px solid #ddd' }}><td style={{ padding: '8px 0' }}>Measurement Scale</td><td style={{ textAlign: 'right', fontWeight: 'bold' }}>{scale.toFixed(2)} mm/px</td></tr>
                          </tbody>
                        </table>
                      </div>
                      <div>
                        <h3 style={{ margin: '0 0 8px 0', fontSize: 14, textTransform: 'uppercase', color: '#666' }}>Assessment</h3>
                        <div style={{ background: statusIsWarning ? '#fff3f3' : '#f0fdf4', padding: 16, borderRadius: 4, border: `1px solid ${statusIsWarning ? '#fca5a5' : '#bbf7d0'}` }}>
                          <strong style={{ color: statusIsWarning ? '#dc2626' : '#166534', fontSize: 16, display: 'block', marginBottom: 4 }}>{statusIsWarning ? 'WARNING: REVIEW REQUIRED' : 'SAFE: WITHIN TOLERANCE'}</strong>
                          <p style={{ margin: 0, fontSize: 13, color: statusIsWarning ? '#991b1b' : '#14532d' }}>
                            {statusIsWarning 
                              ? `Detected crack width (${formatMetric(maxWidth)}mm) exceeds the safety threshold of ${threshold.toFixed(2)}mm.` 
                              : `All detected cracks are below the safety threshold of ${threshold.toFixed(2)}mm.`}
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    <h3 style={{ margin: '0 0 12px 0', fontSize: 14, textTransform: 'uppercase', color: '#666' }}>Analyzed Surface</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                      {originalPreview && <div><img src={originalPreview} style={{ width: '100%', borderRadius: 4, border: '1px solid #ddd' }} alt="Original" /><div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>Source Image</div></div>}
                      {processedPreview && <div><img src={processedPreview} style={{ width: '100%', borderRadius: 4, border: '1px solid #ddd' }} alt="Processed" /><div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>Detection Mask</div></div>}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

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
