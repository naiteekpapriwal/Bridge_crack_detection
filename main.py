"""
main.py — Bridge Crack Detection API

FastAPI application exposing a POST /detect endpoint for image-based
crack detection and measurement.

Features:
  - Multi-stage crack detection (YOLO + Classical CV fallback)
  - Automatic pixel-to-mm calibration via ArUco markers
  - Crack severity classification based on IS 456 / ACI 318 standards
  - Environmental stress & thermal expansion adjustment
"""

import base64
import io
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from calibration import AutoCalibrator, auto_calibrate
from severity_classifier import CrackSeverityClassifier, classify_all_detections
from thermal_adjuster import ThermalExpansionEngine, compute_thermal_adjustment
from vision_pipeline import CrackDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan — initialise detector once at startup
# ---------------------------------------------------------------------------
detector: CrackDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CrackDetector model (or mock) at startup."""
    global detector

    # Point to trained weights — falls back to mock if file doesn't exist yet.
    model_path = "models/crack_seg_best.pt"
    detector = CrackDetector(model_path=model_path)
    logger.info(
        "CrackDetector ready  ·  device=%s  ·  mock=%s",
        detector.device,
        detector.model is None,
    )
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bridge Crack Detection API",
    description=(
        "Upload a bridge image to detect cracks using YOLOv8-Seg "
        "and receive physical width measurements."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """Return service status and active compute device."""
    return {
        "status": "ok",
        "device": detector.device if detector else "not_initialised",
        "mock_mode": detector.model is None if detector else True,
    }


# ---------------------------------------------------------------------------
# POST /detect — main detection endpoint
# ---------------------------------------------------------------------------
@app.post("/detect", tags=["Detection"])
async def detect_cracks(
    file: UploadFile = File(..., description="Image file (JPEG / PNG)"),
    pixel_to_mm_ratio: float = Form(
        0.05, description="Calibration: millimetres per pixel (used if auto_calibrate=false)"
    ),
    use_mock: bool = Form(
        True, description="Force mock inference (ignore loaded model)"
    ),
    auto_calibrate_enabled: bool = Form(
        False, description="Auto-detect ArUco marker for calibration"
    ),
    marker_size_mm: float = Form(
        50.0, description="Physical side length of ArUco marker in mm"
    ),
    # --- Environmental / Thermal parameters ---
    temperature_c: float | None = Form(
        None, description="Ambient temperature in °C (for thermal adjustment)"
    ),
    humidity_pct: float | None = Form(
        None, description="Relative humidity in % (for corrosion risk)"
    ),
    latitude: float | None = Form(
        None, description="Site latitude for live weather (requires OWM key)"
    ),
    longitude: float | None = Form(
        None, description="Site longitude for live weather (requires OWM key)"
    ),
    owm_api_key: str | None = Form(
        None, description="OpenWeatherMap API key (free tier)"
    ),
):
    """
    Accept an image, run crack detection, and return:
      • JSON measurements for every detected crack
      • A base64-encoded annotated image with crack overlays
    """
    # --- 1. Validate & decode the uploaded image -------------------------
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    np_arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded file as an image.",
        )

    logger.info(
        "Received image: %s  ·  shape=%s  ·  use_mock=%s",
        file.filename,
        image.shape,
        use_mock,
    )

    # --- 2. Auto-calibration (if enabled) ---------------------------------
    calibration_info = None
    effective_ratio = pixel_to_mm_ratio

    if auto_calibrate_enabled:
        cal_result = auto_calibrate(
            image,
            marker_size_mm=marker_size_mm,
            fallback_ratio=pixel_to_mm_ratio,
        )
        effective_ratio = cal_result.pixel_to_mm
        calibration_info = cal_result.to_dict()
        logger.info(
            "Auto-calibration: method=%s, ratio=%.6f mm/px, confidence=%.2f",
            cal_result.method, cal_result.pixel_to_mm, cal_result.confidence,
        )

    # --- 3. Run detection ------------------------------------------------
    if use_mock or detector.model is None:
        detections = detector.predict_mock(
            image, pixel_to_mm_ratio=effective_ratio
        )
        pipeline_info = {
            "preprocessing": "none (mock mode)",
            "filtering": "none (mock mode)",
            "raw_detections": len(detections),
            "filtered_detections": 0,
            "accepted_detections": len(detections),
        }
    else:
        detections, pipeline_info = detector.predict(
            image, pixel_to_mm_ratio=effective_ratio
        )

    logger.info(
        "Pipeline result: raw=%d  filtered=%d  accepted=%d",
        pipeline_info.get("raw_detections", 0),
        pipeline_info.get("filtered_detections", 0),
        pipeline_info.get("accepted_detections", 0),
    )

    # --- 4. Severity classification --------------------------------------
    severity_classifier = CrackSeverityClassifier()
    classify_all_detections(detections, image.shape, severity_classifier)

    # --- 4b. Environmental / Thermal adjustment ---------------------------
    thermal_info = None
    has_env_data = (
        (temperature_c is not None and humidity_pct is not None)
        or (latitude is not None and longitude is not None)
    )
    if has_env_data and len(detections) > 0:
        # Use the worst (highest severity) detection for thermal analysis
        worst_det = max(
            detections,
            key=lambda d: d.get("severity", {}).get("severity_score", 0),
        )
        worst_severity = worst_det.get("severity", {})
        worst_measurements = worst_det.get("measurements", {})

        thermal_result = compute_thermal_adjustment(
            crack_width_mm=worst_measurements.get("mean_width_mm", 0.0),
            crack_length_px=worst_measurements.get("crack_length_px", 0),
            severity_score=worst_severity.get("severity_score", 0.0),
            latitude=latitude,
            longitude=longitude,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            api_key=owm_api_key,
        )
        thermal_info = thermal_result.to_dict()

        # Update each detection's severity with the environmental multiplier
        for det in detections:
            sev = det.get("severity", {})
            original_score = sev.get("severity_score", 0)
            adjusted = min(100.0, original_score * thermal_result.severity_multiplier)
            sev["severity_score_original"] = original_score
            sev["severity_score"] = round(adjusted, 1)
            sev["environmental_multiplier"] = round(thermal_result.severity_multiplier, 3)

    # --- 5. Build annotated image ----------------------------------------
    annotated = image.copy()
    for det in detections:
        mask = det.get("mask")
        severity = det.get("severity", {})
        # Use severity-specific colour if available
        overlay_color = severity.get("severity_color", (0, 0, 255))
        if mask is not None:
            annotated = detector.draw_overlay(
                annotated, mask, color=overlay_color
            )

    # Encode annotated image to base64 JPEG with data URI prefix
    # so frontend <img src="..."> tags can render it directly.
    success, encoded = cv2.imencode(
        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to encode annotated image."
        )
    b64_image = base64.b64encode(encoded.tobytes()).decode("utf-8")
    annotated_data_uri = f"data:image/jpeg;base64,{b64_image}"

    # --- 6. Strip binary masks from the JSON response --------------------
    response_detections = []
    for det in detections:
        severity_data = det.get("severity", {})
        # Remove non-serialisable fields (BGR colour tuple)
        serialisable_severity = {
            k: v for k, v in severity_data.items()
            if k != "severity_color"
        }

        response_detections.append(
            {
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "measurements": det["measurements"],
                "severity": serialisable_severity,
            }
        )

    response_content = {
        "success": True,
        "num_detections": len(response_detections),
        "detections": response_detections,
        "image_shape": list(image.shape),
        "annotated_image": annotated_data_uri,
        "pipeline_info": pipeline_info,
    }

    # Include calibration info if auto-calibration was used
    if calibration_info is not None:
        response_content["calibration"] = calibration_info

    # Include thermal/environmental analysis if performed
    if thermal_info is not None:
        response_content["thermal_analysis"] = thermal_info

    return JSONResponse(content=response_content)
