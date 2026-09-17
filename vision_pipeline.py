"""
vision_pipeline.py — Bridge Crack Detection Pipeline

YOLOv8-Seg inference (real + mock) and OpenCV crack-width measurement.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CrackDetector:
    """Runs YOLOv8-Seg inference and computes physical crack measurements."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise the detector.

        Args:
            model_path: Path to a YOLOv8-Seg .pt weights file.
                        If None, all calls are routed to predict_mock().
        """
        self.model = None
        self.device = "cpu"
        self._model_path = model_path

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            if model_path:
                logger.warning(
                    "Model file '%s' not found — falling back to mock inference.",
                    model_path,
                )
            else:
                logger.info("No model path provided — using mock inference.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Load YOLOv8-Seg model with MPS → CPU fallback."""
        try:
            from ultralytics import YOLO  # noqa: delayed import
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for real inference. "
                "Install with: pip install ultralytics"
            ) from exc

        # Device selection: prefer Apple Metal (MPS), fall back to CPU.
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        logger.info("Loading YOLOv8-Seg model on device: %s", self.device)
        self.model = YOLO(model_path)
        # Force model onto the selected device by running a tiny warm-up.
        self.model.to(self.device)
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Prediction (real)
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.50,
        pixel_to_mm_ratio: float = 0.05,
    ) -> list[dict]:
        """
        Run real YOLOv8-Seg inference on *image* (BGR, uint8).

        Returns a list of detection dicts (one per detected crack).
        """
        if self.model is None:
            return self.predict_mock(image, pixel_to_mm_ratio=pixel_to_mm_ratio)

        results = self.model.predict(
            source=image,
            conf=confidence,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        result = results[0]  # single image → single Results object

        if result.masks is None:
            return detections

        for i, mask_tensor in enumerate(result.masks.data):
            # Convert mask tensor → uint8 binary mask at original image size.
            mask = mask_tensor.cpu().numpy().squeeze()
            mask_resized = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

            # Bounding box & metadata.
            box = result.boxes[i]
            bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            measurements = self.calculate_measurements(
                binary_mask, pixel_to_mm_ratio=pixel_to_mm_ratio
            )

            detections.append(
                {
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in bbox],
                    "mask": binary_mask,
                    "measurements": measurements,
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Prediction (mock) — for prototyping without trained weights
    # ------------------------------------------------------------------

    def predict_mock(
        self,
        image: np.ndarray,
        pixel_to_mm_ratio: float = 0.05,
    ) -> list[dict]:
        """
        Generate a synthetic crack detection result.

        Draws a diagonal crack-like stripe on a blank mask so the rest of
        the pipeline (measurement + overlay) can be developed and tested
        without a trained model.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # --- Synthetic crack: a slightly curved diagonal line --------
        # Generate a polyline from top-left area to bottom-right area
        # with random perturbations to look organic.
        rng = np.random.default_rng(seed=42)
        num_points = 20
        xs = np.linspace(int(w * 0.15), int(w * 0.85), num_points).astype(int)
        ys = np.linspace(int(h * 0.20), int(h * 0.80), num_points).astype(int)
        # Add random jitter to simulate natural crack path
        xs = xs + rng.integers(-int(w * 0.03), int(w * 0.03), size=num_points)
        ys = ys + rng.integers(-int(h * 0.02), int(h * 0.02), size=num_points)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        points = np.column_stack((xs, ys)).reshape((-1, 1, 2)).astype(np.int32)

        # Draw with varying thickness to simulate tapering crack
        thickness = max(3, min(15, int(min(h, w) * 0.02)))
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness)

        # Slight Gaussian blur to mimic soft segmentation edges
        ksize = thickness | 1  # ensure odd
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Bounding box around the non-zero mask region
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            bbox = [float(x), float(y), float(x + bw), float(y + bh)]
        else:
            bbox = [0.0, 0.0, float(w), float(h)]

        measurements = self.calculate_measurements(
            mask, pixel_to_mm_ratio=pixel_to_mm_ratio
        )

        return [
            {
                "class_name": "crack",
                "confidence": 0.92,
                "bbox": bbox,
                "mask": mask,
                "measurements": measurements,
            }
        ]

    # ------------------------------------------------------------------
    # Crack-width measurement via distanceTransform + skeletonisation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_measurements(
        mask: np.ndarray,
        pixel_to_mm_ratio: float = 0.05,
    ) -> dict:
        """
        Compute crack width along its skeleton.

        Algorithm:
          1. distanceTransform  → distance of each crack pixel to nearest edge
          2. thinning (Zhang-Suen) → 1-pixel skeleton (centerline)
          3. width = 2 × distance_map values sampled along the skeleton

        Args:
            mask: Binary mask (uint8, 0 or 255).
            pixel_to_mm_ratio: mm-per-pixel calibration constant.

        Returns:
            Dict of measurement values.
        """
        # Ensure binary uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255

        # Crack pixel area
        crack_area_px = int(np.count_nonzero(mask))

        if crack_area_px == 0:
            return {
                "max_width_px": 0.0,
                "mean_width_px": 0.0,
                "max_width_mm": 0.0,
                "mean_width_mm": 0.0,
                "crack_area_px": 0,
                "crack_length_px": 0,
            }

        # 1. Distance transform — gives radius at each foreground pixel
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # 2. Skeletonisation (centerline extraction)
        skeleton = CrackDetector._skeletonize(mask)

        # Crack length ≈ number of skeleton pixels
        skeleton_pixels = skeleton > 0
        crack_length_px = int(np.count_nonzero(skeleton_pixels))

        if crack_length_px == 0:
            # Fallback: if skeleton is empty (very small mask), estimate width
            # from the distance transform maximum.
            max_radius = float(dist_map.max())
            return {
                "max_width_px": round(2 * max_radius, 2),
                "mean_width_px": round(2 * max_radius, 2),
                "max_width_mm": round(2 * max_radius * pixel_to_mm_ratio, 4),
                "mean_width_mm": round(2 * max_radius * pixel_to_mm_ratio, 4),
                "crack_area_px": crack_area_px,
                "crack_length_px": 1,
            }

        # 3. Sample the distance map along the skeleton → radii
        radii = dist_map[skeleton_pixels]
        widths = 2 * radii  # diameter = 2 × radius

        max_width_px = float(np.max(widths))
        mean_width_px = float(np.mean(widths))

        return {
            "max_width_px": round(max_width_px, 2),
            "mean_width_px": round(mean_width_px, 2),
            "max_width_mm": round(max_width_px * pixel_to_mm_ratio, 4),
            "mean_width_mm": round(mean_width_px * pixel_to_mm_ratio, 4),
            "crack_area_px": crack_area_px,
            "crack_length_px": crack_length_px,
        }

    # ------------------------------------------------------------------
    # Skeletonisation helper (ximgproc → morphological fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        """
        Reduce a binary mask to a 1-pixel-wide skeleton.

        Tries cv2.ximgproc.thinning first (Zhang-Suen); if unavailable
        (e.g. OpenCV 5.x), falls back to iterative morphological
        erosion — produces equivalent results.

        Args:
            mask: Binary mask (uint8, 0 or 255).

        Returns:
            Skeleton image (uint8, 0 or 255).
        """
        # Strategy 1: cv2.ximgproc.thinning (fastest, cleanest)
        try:
            return cv2.ximgproc.thinning(
                mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except (AttributeError, cv2.error):
            pass

        # Strategy 2: Morphological skeleton (works on all OpenCV builds)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(mask)
        temp = mask.copy()

        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            diff = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()

            if cv2.countNonZero(temp) == 0:
                break

        return skeleton

    # ------------------------------------------------------------------
    # Visualisation overlay
    # ------------------------------------------------------------------

    @staticmethod
    def draw_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Alpha-blend the crack mask onto the original image.

        Args:
            image: Original BGR image (uint8).
            mask:  Binary mask (uint8, 0 or 255), same H×W as image.
            color: BGR overlay colour (default: red).
            alpha: Overlay opacity.

        Returns:
            Annotated image (BGR, uint8).
        """
        overlay = image.copy()
        mask_bool = mask > 0

        # Fill the crack region with the overlay colour
        overlay[mask_bool] = color

        # Blend
        annotated = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw contour outlines for crisp edges
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, thickness=2)

        return annotated
