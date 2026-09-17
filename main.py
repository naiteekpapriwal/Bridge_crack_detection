"""
main.py — Bridge Crack Detection API

FastAPI application exposing a POST /detect endpoint for image-based
crack detection and measurement.
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
        0.05, description="Calibration: millimetres per pixel"
    ),
    use_mock: bool = Form(
        True, description="Force mock inference (ignore loaded model)"
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

    # --- 2. Run detection ------------------------------------------------
    if use_mock or detector.model is None:
        detections = detector.predict_mock(
            image, pixel_to_mm_ratio=pixel_to_mm_ratio
        )
    else:
        detections = detector.predict(
            image, pixel_to_mm_ratio=pixel_to_mm_ratio
        )

    # --- 3. Build annotated image ----------------------------------------
    annotated = image.copy()
    for det in detections:
        mask = det.get("mask")
        if mask is not None:
            annotated = detector.draw_overlay(annotated, mask)

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

    # --- 4. Strip binary masks from the JSON response --------------------
    response_detections = []
    for det in detections:
        response_detections.append(
            {
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "measurements": det["measurements"],
            }
        )

    return JSONResponse(
        content={
            "success": True,
            "num_detections": len(response_detections),
            "detections": response_detections,
            "image_shape": list(image.shape),
            "annotated_image": annotated_data_uri,
        }
    )
