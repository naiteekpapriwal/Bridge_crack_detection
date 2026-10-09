# Bridge Crack Detection — Complete Project Report
### Updated: 9 October 2026

Team, here is the full breakdown of our project, including our **three major original contributions** that set this project apart from a standard ML project.

---

## 🏗️ 1. Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + TypeScript)                  │
│  Upload → Configure (scale, env, calibration) → View Results     │
└───────────────────────────┬──────────────────────────────────────┘
                            │ POST /detect
┌───────────────────────────▼──────────────────────────────────────┐
│                    BACKEND (FastAPI + Python)                     │
│                                                                  │
│  ┌─────────────────┐  ┌────────────────────┐  ┌───────────────┐ │
│  │  YOLOv8-Seg     │  │  Classical CV      │  │  ArUco Auto   │ │
│  │  Detection      │→ │  Fallback Pipeline │  │  Calibration  │ │
│  └────────┬────────┘  └────────┬───────────┘  └───────┬───────┘ │
│           └────────────────────┘                      │         │
│                        │                              │         │
│  ┌─────────────────────▼──────────────────────────────▼───────┐ │
│  │            SEVERITY CLASSIFIER (IS 456 / ACI 318)          │ │
│  │  width × 0.40 + length × 0.25 + branching × 0.20          │ │
│  │  + density × 0.15 → Score 0-100 → Hairline/Minor/Mod/Sev  │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                        │
│  ┌─────────────────────▼──────────────────────────────────────┐ │
│  │         THERMAL EXPANSION ENGINE (IS 456 Clause 6.2.6)     │ │
│  │  ε = α × ΔT  →  Width correction to 20°C reference        │ │
│  │  Freeze-thaw risk (ACI 201.2R) + Corrosion risk (Tuutti)   │ │
│  │  → Environmental severity multiplier (up to ×1.4)          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Tech Stack
- **Frontend:** React, TypeScript, Vite, Custom CSS
- **Backend:** FastAPI, OpenCV, NumPy, Ultralytics YOLOv8-Seg
- **New Modules:** `severity_classifier.py`, `calibration.py`, `thermal_adjuster.py`

---

## 🚀 2. Our Three Original Contributions

### Contribution A: Crack Severity Classification System
**File:** `severity_classifier.py` | **Standard:** IS 456:2000, ACI 318-19

YOLO only says "crack." Our custom algorithm answers **"how dangerous is it?"**

**Our scoring formula (designed by us):**
```
severity = 0.40 × width_score + 0.25 × length_score
         + 0.20 × branching_score + 0.15 × density_score
```

**The key original algorithm — Branching Detection:**
We skeletonize the crack mask to 1px width, then run a custom 3×3 convolution kernel to count **junction points** (pixels with ≥3 neighbours). Each junction means the crack splits — a sign of dangerous stress redistribution. **No existing library provides this analysis.**

| Level | Score | Engineering Action |
|-------|-------|--------------------|
| 🟢 Hairline | 0–25 | Monitor in 6-12 months |
| 🟡 Minor | 25–50 | Schedule epoxy sealing |
| 🟠 Moderate | 50–75 | Urgent assessment in 2 weeks |
| 🔴 Severe | 75–100 | Immediate closure/evaluation |

---

### Contribution B: Auto Pixel-to-MM Calibration
**File:** `calibration.py` | **Method:** ArUco Marker Detection

Solves a real field problem — no more guessing the `pixel_to_mm` ratio.

**How it works:**
1. Inspector places a printed ArUco marker (known size: e.g. 50mm) next to the crack
2. Our code detects the marker corners → measures pixel side length
3. Computes: `pixel_to_mm = marker_size_mm / avg_side_px`
4. Reports confidence based on marker squareness and size

**Test result:** 50mm marker at 200px → computed 0.2513 mm/px (expected 0.2500) = **99.5% accurate**

---

### Contribution C: Environmental Stress & Thermal Expansion Engine 🆕
**File:** `thermal_adjuster.py` | **Standards:** IS 456 Clause 6.2.6, ACI 201.2R-16, Tuutti's Model

**This is the showstopper.** We integrate real-world physics and thermodynamics into our severity assessment.

**The physics we implement:**

1. **Thermal Expansion (IS 456):**
   - Concrete CTE: α = 10 × 10⁻⁶ /°C
   - Thermal strain: `ε = α × (T - 20°C)`
   - Crack width is corrected to what it would be at 20°C reference temperature
   - Hot weather → concrete expands → cracks appear smaller than they structurally are
   - Cold weather → concrete contracts → cracks open wider

2. **Freeze-Thaw Risk (ACI 201.2R):**
   - Water in cracks expands **9% when it freezes**
   - Each freeze-thaw cycle progressively widens cracks
   - Our model computes: `risk = proximity_to_freezing × moisture_availability`
   - At -2°C with 92% humidity → freeze-thaw risk = **88.6/100**

3. **Humidity & Corrosion Risk (Tuutti/Papadakis):**
   - Above 60% RH → carbonation rate peaks → rebar corrosion accelerates
   - Above 85% RH → active corrosion current
   - We model this with a piecewise function based on published corrosion data

4. **Combined Environmental Multiplier:**
   - `env_risk = 0.45 × freeze_thaw + 0.30 × humidity + 0.25 × thermal`
   - Severity can increase up to **×1.4** in bad conditions
   - Effectively: a 45-score crack in mild weather becomes a 59.3-score crack in freezing rain

**Test Results:**
| Condition | Temp | RH | Severity | Multiplier | Risk Level |
|-----------|------|----|----------|------------|------------|
| Hot & Dry (Rajasthan summer) | 42°C | 25% | 45 → 47.3 | ×1.052 | Minimal |
| Freezing Rain (worst case) | -2°C | 92% | 45 → 59.3 | ×1.319 | ⚠️ Critical |
| Mild & Dry (ideal) | 22°C | 45% | 45 → 45.6 | ×1.014 | Minimal |

The same crack gets a **32% higher severity score** in freezing rain vs mild weather. This is real structural engineering, not just ML.

---

## 💻 3. Frontend Features

The dashboard displays all three systems:
- **Severity Gauge:** Circular progress ring with 0-100 score + factor breakdown bars
- **Calibration Panel:** Shows ArUco detection confidence and computed scale
- **Thermal Analysis Card:** Weather data, 3 risk bars (thermal/humidity/freeze-thaw), width correction, advisory text
- **Environment Input:** Temperature and humidity fields in the settings panel

---

## 🗣️ 4. Presentation Talking Points

**Opening statement:**
> "While we use YOLOv8 for initial segmentation, the real engineering is our custom pipeline built around it. We implemented three systems from scratch."

**On Severity:**
> "We designed our own scoring formula based on IS 456 crack width limits. Our branching detection algorithm — which counts skeleton junction points using 8-connected neighbour convolution — is entirely our own. No library does this."

**On Calibration:**
> "We solved the real-world problem of inaccurate pixel scaling by building automatic ArUco marker detection. It achieves 99.5% accuracy."

**On Thermal (the knockout punch):**
> "We integrated real thermodynamics into our system. Using the coefficient of thermal expansion from IS 456 Clause 6.2.6, we correct crack widths to a 20°C reference temperature. We model freeze-thaw risk from ACI 201.2R and corrosion risk from Tuutti's model. The same crack gets a 32% higher danger score in freezing rain than in mild weather. This shows we're thinking like structural engineers, not just software developers."

---

## 📂 5. File Map

| File | Purpose | Original? |
|------|---------|-----------|
| `main.py` | FastAPI endpoint with all integrations | Modified |
| `vision_pipeline.py` | YOLO + Classical CV detection pipeline | Base |
| `train_model.py` | YOLOv8 training script | Base |
| `severity_classifier.py` | **Custom severity scoring algorithm** | ✅ 100% ours |
| `calibration.py` | **ArUco auto-calibration system** | ✅ 100% ours |
| `thermal_adjuster.py` | **Thermal expansion + environmental risk** | ✅ 100% ours |
| `frontend/...App.tsx` | Dashboard with severity + thermal UI | Modified |
| `frontend/...index.css` | Styles for all new components | Modified |

---

*Run backend: `uvicorn main:app --host 0.0.0.0 --port 8000`*
*Run frontend: `cd frontend/bridge-crack-dashboard/client && npm run dev`*
