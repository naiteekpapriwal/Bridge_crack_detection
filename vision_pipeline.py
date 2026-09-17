"""
vision_pipeline.py — Bridge Crack Detection Pipeline

Multi-stage pipeline:
  Stage 1: Image preprocessing  (CLAHE + Black-Hat enhancement)
  Stage 2: YOLOv8-Seg inference (real + mock)
  Stage 3: Joint/edge filtering (straightness, irregularity, width variance)
  Stage 4: Crack-width measurement via distanceTransform + skeletonisation
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CrackDetector:
    """Multi-stage crack detection: preprocess → detect → filter → measure."""

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

    # ==================================================================
    # STAGE 1 — Image Preprocessing
    # ==================================================================

    @staticmethod
    def preprocess(
        image: np.ndarray,
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        blackhat_ksize: int = 15,
        blend_alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Enhance crack visibility using CLAHE and morphological Black-Hat.

        Pipeline:
          1. Convert to grayscale
          2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
             to boost local contrast — makes subtle cracks more visible.
          3. Apply morphological Black-Hat transform to isolate thin dark
             features (cracks) against a brighter background.
          4. Blend the enhanced channel back into the original image
             so YOLO receives a crack-enhanced colour image.

        Args:
            image:          BGR uint8 input image.
            clahe_clip:     CLAHE clip limit (higher = more contrast).
            clahe_grid:     CLAHE tile grid size.
            blackhat_ksize: Kernel size for Black-Hat (should be larger than
                            crack width in pixels to capture them).
            blend_alpha:    How much of the enhanced channel to blend in.

        Returns:
            Enhanced BGR image (same shape as input).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- CLAHE: boost local contrast ---
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        enhanced_gray = clahe.apply(gray)

        # --- Black-Hat: highlight thin dark structures (cracks) ---
        # Black-Hat = closing(image) - image
        # It isolates dark features smaller than the kernel.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (blackhat_ksize, blackhat_ksize)
        )
        blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)

        # Amplify the black-hat response
        blackhat_amplified = cv2.normalize(
            blackhat, None, 0, 255, cv2.NORM_MINMAX
        )

        # --- Blend back into the original image ---
        # Subtract the crack-highlight from the value channel to darken cracks
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Replace V channel with CLAHE-enhanced version
        v_enhanced = cv2.addWeighted(
            enhanced_gray, blend_alpha,
            v, (1 - blend_alpha),
            0,
        )

        # Darken pixels where black-hat detected crack-like features
        crack_boost = cv2.subtract(
            v_enhanced,
            (blackhat_amplified * 0.5).astype(np.uint8),
        )

        hsv_out = cv2.merge([h, s, crack_boost])
        result = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

        logger.debug(
            "Preprocessing complete: CLAHE(clip=%.1f, grid=%d) + "
            "BlackHat(k=%d) + blend(α=%.2f)",
            clahe_clip, clahe_grid, blackhat_ksize, blend_alpha,
        )
        return result

    # ==================================================================
    # STAGE 2 — YOLO Detection
    # ==================================================================

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.20,
        pixel_to_mm_ratio: float = 0.05,
    ) -> tuple[list[dict], dict]:
        """
        Full multi-stage pipeline: preprocess → detect → filter → measure.

        Args:
            image:             BGR uint8 input image (original).
            confidence:        YOLO confidence threshold (low — we filter later).
            pixel_to_mm_ratio: mm-per-pixel calibration constant.

        Returns:
            Tuple of (detections_list, pipeline_info_dict).
        """
        pipeline_info = {
            "preprocessing": "CLAHE + Black-Hat enhancement",
            "model_confidence": confidence,
            "filtering": "Straightness + Irregularity + Width-Variance scoring",
            "raw_detections": 0,
            "filtered_detections": 0,
            "accepted_detections": 0,
        }

        if self.model is None:
            detections = self.predict_mock(image, pixel_to_mm_ratio=pixel_to_mm_ratio)
            pipeline_info["preprocessing"] = "none (mock mode)"
            pipeline_info["filtering"] = "none (mock mode)"
            pipeline_info["accepted_detections"] = len(detections)
            return detections, pipeline_info

        # --- Stage 1: Preprocess ---
        enhanced = self.preprocess(image)

        # --- Stage 2: YOLO inference on enhanced image ---
        results = self.model.predict(
            source=enhanced,
            conf=confidence,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        result = results[0]  # single image → single Results object

        if result.masks is not None:
            pipeline_info["raw_detections"] = len(result.masks.data)

            for i, mask_tensor in enumerate(result.masks.data):
                # Convert mask tensor → uint8 binary mask at original image size.
                mask = mask_tensor.cpu().numpy().squeeze()
                mask_resized = cv2.resize(
                    mask, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
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

                # --- Stage 3: Joint/edge filtering ---
                is_crack, reject_reason = self._is_plausible_crack(
                    binary_mask, image.shape, measurements
                )
                if not is_crack:
                    pipeline_info["filtered_detections"] += 1
                    logger.info(
                        "Filtered detection %d (class=%s, conf=%.2f): %s",
                        i, cls_name, conf, reject_reason,
                    )
                    continue

                detections.append(
                    {
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 1) for v in bbox],
                        "mask": binary_mask,
                        "measurements": measurements,
                    }
                )

        pipeline_info["accepted_detections"] = len(detections)

        # --- Fallback: Classical CV (full image + tiled zoom) ---
        if len(detections) == 0:
            logger.info("YOLO found 0 cracks — running classical CV fallback.")

            # Pass 1: Full-image classical CV
            full_detections = self.predict_classical(
                image, pixel_to_mm_ratio=pixel_to_mm_ratio
            )

            # Pass 2: Tiled detection (zoom into patches)
            tiled_detections = self.predict_tiled(
                image, pixel_to_mm_ratio=pixel_to_mm_ratio
            )

            # Merge both passes, deduplicate by IoU
            all_classical = self._merge_detections(
                full_detections, tiled_detections
            )

            if all_classical:
                pipeline_info["preprocessing"] += " + Classical CV + Tiled zoom"
                pipeline_info["accepted_detections"] = len(all_classical)
                logger.info(
                    "Classical CV: %d full + %d tiled → %d merged.",
                    len(full_detections), len(tiled_detections), len(all_classical),
                )
                return all_classical, pipeline_info

        return detections, pipeline_info

    # ==================================================================
    # Tiled (zoom) detection — catches far-away cracks
    # ==================================================================

    def predict_tiled(
        self,
        image: np.ndarray,
        pixel_to_mm_ratio: float = 0.05,
        grid: tuple[int, int] = (2, 2),
        overlap: float = 0.25,
    ) -> list[dict]:
        """
        Split the image into overlapping tiles, run classical CV on each,
        then map detections back to original image coordinates.

        This effectively "zooms in" on each region, making distant/small
        cracks more detectable.

        Args:
            image:             BGR input image.
            pixel_to_mm_ratio: Calibration constant.
            grid:              (rows, cols) tile grid.
            overlap:           Fraction of overlap between adjacent tiles.
        """
        h, w = image.shape[:2]
        rows, cols = grid

        # Calculate tile sizes with overlap
        tile_h = int(h / rows * (1 + overlap))
        tile_w = int(w / cols * (1 + overlap))
        step_y = int(h / rows)
        step_x = int(w / cols)

        all_detections: list[dict] = []

        for r in range(rows):
            for c in range(cols):
                # Tile boundaries (clipped to image)
                y1 = max(0, r * step_y - int(tile_h * overlap / 2))
                x1 = max(0, c * step_x - int(tile_w * overlap / 2))
                y2 = min(h, y1 + tile_h)
                x2 = min(w, x1 + tile_w)

                tile = image[y1:y2, x1:x2]

                if tile.shape[0] < 50 or tile.shape[1] < 50:
                    continue

                # Run classical CV on the tile
                tile_detections = self.predict_classical(
                    tile, pixel_to_mm_ratio=pixel_to_mm_ratio
                )

                # Map masks and bboxes back to original image coordinates
                for det in tile_detections:
                    # Remap mask to full image
                    tile_mask = det["mask"]
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = tile_mask

                    # Remap bbox
                    bx1, by1, bx2, by2 = det["bbox"]
                    det["mask"] = full_mask
                    det["bbox"] = [
                        round(bx1 + x1, 1),
                        round(by1 + y1, 1),
                        round(bx2 + x1, 1),
                        round(by2 + y1, 1),
                    ]

                    all_detections.append(det)

        logger.info(
            "Tiled detection (%dx%d, overlap=%.0f%%): %d cracks from %d tiles.",
            rows, cols, overlap * 100, len(all_detections), rows * cols,
        )

        return all_detections

    @staticmethod
    def _merge_detections(
        primary: list[dict],
        secondary: list[dict],
        iou_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Merge two detection lists, deduplicating by mask IoU.
        """
        merged = list(primary)

        for sec_det in secondary:
            sec_mask = sec_det.get("mask")
            if sec_mask is None:
                continue

            is_duplicate = False
            for existing in merged:
                pri_mask = existing.get("mask")
                if pri_mask is None:
                    continue

                intersection = cv2.bitwise_and(sec_mask, pri_mask)
                union = cv2.bitwise_or(sec_mask, pri_mask)
                i_area = np.count_nonzero(intersection)
                u_area = np.count_nonzero(union)

                if u_area > 0 and (i_area / u_area) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(sec_det)

        return merged

    # ==================================================================
    # Classical CV Crack Detection (Black-Hat based, balanced filters)
    # ==================================================================

    def predict_classical(
        self,
        image: np.ndarray,
        pixel_to_mm_ratio: float = 0.05,
    ) -> list[dict]:
        """
        Detect cracks using Black-Hat morphology + balanced contour filtering.

        Uses ONLY Black-Hat (the most crack-specific CV operation) to avoid
        false positives from edges, shadows, and sky that Canny/dark-channel
        pick up.
        """
        h, w = image.shape[:2]

        # --- Grayscale + CLAHE ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # --- Black-Hat at multiple scales ---
        blackhat_combined = np.zeros_like(enhanced)
        for ksize in [11, 23, 45, 75]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            bh = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            blackhat_combined = cv2.max(blackhat_combined, bh)

        bh_norm = cv2.normalize(blackhat_combined, None, 0, 255, cv2.NORM_MINMAX)

        # --- Threshold: Otsu + adaptive, combined ---
        _, otsu = cv2.threshold(
            bh_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            bh_norm, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=-2,
        )
        binary = cv2.bitwise_or(otsu, adaptive)

        # --- Morphological cleanup ---
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

        # --- Contour analysis with balanced filters ---
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Dynamic min area: at least 0.05% of image area, minimum 150px
        min_area = max(150, int(h * w * 0.0005))

        detections: list[dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, bw, bh_box = cv2.boundingRect(contour)
            long_side = max(bw, bh_box, 1)
            short_side = max(min(bw, bh_box), 1)
            aspect = long_side / short_side

            area_ratio = area / (h * w)
            is_massive_defect = area_ratio > 0.015  # Greater than 1.5% of the image

            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 1.0

            if not is_massive_defect:
                # Cracks are elongated — reject compact blobs
                if aspect < 1.5:
                    continue

                # Solidity: cracks are not solidly filled
                if solidity > 0.75:
                    continue
            else:
                # For massive structural damage (spalling/huge cracks), allow blobby shapes
                # but still reject perfectly square non-defects.
                if aspect < 1.1 and solidity > 0.90:
                    continue

            # Reject if it covers more than a third of the image (likely a wall/sky)
            if area_ratio > 0.35:
                continue

            # Build mask
            crack_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(crack_mask, [contour], -1, 255, thickness=cv2.FILLED)

            bbox = [float(x), float(y), float(x + bw), float(y + bh_box)]
            measurements = self.calculate_measurements(
                crack_mask, pixel_to_mm_ratio=pixel_to_mm_ratio
            )

            confidence = min(0.82, 0.45 + (aspect / 15.0) + (1.0 - solidity) * 0.25)

            detections.append({
                "class_name": "crack",
                "confidence": round(confidence, 4),
                "bbox": [round(v, 1) for v in bbox],
                "mask": crack_mask,
                "measurements": measurements,
            })

        detections.sort(
            key=lambda d: d["measurements"].get("crack_area_px", 0),
            reverse=True,
        )

        logger.info(
            "Classical CV: %d contours → %d cracks (min_area=%d).",
            len(contours), len(detections), min_area,
        )

        return detections

    # ==================================================================
    # STAGE 3 — Joint / Edge Filtering
    # ==================================================================

    @staticmethod
    def _is_plausible_crack(
        mask: np.ndarray,
        image_shape: tuple,
        measurements: dict,
    ) -> tuple[bool, str]:
        """
        Multi-factor scoring to distinguish real cracks from joints/edges.

        Checks:
          1. Basic geometry  (area ratio, aspect ratio)
          2. Straightness    (Hough line detection — joints are straight)
          3. Irregularity    (contour perimeter vs convex hull — cracks are jagged)
          4. Width variance  (cracks vary in width, joints are uniform)

        Returns:
            (is_plausible: bool, reason: str)
        """
        h, w = image_shape[:2]
        image_area = h * w

        # --- 1. Basic geometry ---
        crack_area = measurements.get("crack_area_px", 0)
        if crack_area == 0:
            return False, "empty mask"

        area_ratio = crack_area / image_area
        if area_ratio > 0.15:
            return False, f"area too large ({area_ratio:.1%} of image)"

        # Aspect ratio check
        coords = cv2.findNonZero(mask)
        if coords is None:
            return False, "no foreground pixels"

        _, _, bw, bh = cv2.boundingRect(coords)
        long_side = max(bw, bh, 1)
        short_side = max(min(bw, bh), 1)
        aspect = long_side / short_side
        if aspect < 1.5:
            return False, f"too compact (aspect={aspect:.1f})"

        # --- 2. Straightness score (Hough Lines) ---
        straightness = CrackDetector._compute_straightness(mask)
        if straightness > 0.70:
            return False, f"too straight ({straightness:.0%}) — likely a joint or edge"

        # --- 3. Contour irregularity ---
        irregularity = CrackDetector._compute_irregularity(mask)
        # Low irregularity = smooth contour = likely a joint
        if irregularity < 1.05 and straightness > 0.40:
            return False, (
                f"too smooth (irregularity={irregularity:.2f}) "
                f"and moderately straight ({straightness:.0%})"
            )

        # --- 4. Width variance ---
        width_cv = CrackDetector._compute_width_variance(mask)
        # Very low width variance + high straightness = joint
        if width_cv < 0.15 and straightness > 0.35:
            return False, (
                f"uniform width (CV={width_cv:.2f}) "
                f"and moderately straight ({straightness:.0%}) — likely a joint"
            )

        # Mean width check (very thick features are unlikely cracks)
        mean_width = measurements.get("mean_width_px", 0.0)
        if mean_width > 80.0:
            return False, f"too wide (mean_width={mean_width:.0f}px)"

        return True, "passed all checks"

    @staticmethod
    def _compute_straightness(mask: np.ndarray) -> float:
        """
        Score how straight a mask region is using Hough Line Transform.

        Computes what fraction of the mask's skeleton pixels lie on
        detected straight lines. High fraction = straight (joint-like).

        Returns:
            Float in [0, 1]. Higher = more straight.
        """
        # Skeletonize the mask first
        skeleton = CrackDetector._skeletonize(mask)
        skeleton_px = np.count_nonzero(skeleton)

        if skeleton_px < 10:
            return 0.0  # Too few pixels to judge

        # Detect straight lines in the skeleton
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, skeleton_px // 10),
            minLineLength=max(30, skeleton_px // 5),
            maxLineGap=10,
        )

        if lines is None:
            return 0.0

        # Count how many skeleton pixels fall near a detected line
        line_mask = np.zeros_like(skeleton)
        for line in lines:
            coords = line[0] if line.ndim > 1 else line
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)

        # Overlap between skeleton and detected lines
        overlap = cv2.bitwise_and(skeleton, line_mask)
        overlap_px = np.count_nonzero(overlap)

        return min(overlap_px / skeleton_px, 1.0)

    @staticmethod
    def _compute_irregularity(mask: np.ndarray) -> float:
        """
        Measure contour irregularity (jaggedness).

        Ratio = actual contour perimeter / convex hull perimeter.
        - Cracks: irregular, jagged edges → ratio > 1.2
        - Joints: smooth, regular edges  → ratio ≈ 1.0

        Returns:
            Float ≥ 1.0. Higher = more irregular / jagged.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 1.0

        # Use the largest contour
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, closed=True)

        hull = cv2.convexHull(largest)
        hull_perimeter = cv2.arcLength(hull, closed=True)

        if hull_perimeter < 1:
            return 1.0

        return perimeter / hull_perimeter

    @staticmethod
    def _compute_width_variance(mask: np.ndarray) -> float:
        """
        Coefficient of variation of crack width along the skeleton.

        Joints have very uniform width (low CV).
        Cracks taper and vary (high CV).

        Returns:
            Coefficient of variation (std / mean). Higher = more variable.
        """
        # Distance transform → radius at each pixel
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Skeleton → sample widths along centerline
        skeleton = CrackDetector._skeletonize(mask)
        skeleton_pixels = skeleton > 0

        if np.count_nonzero(skeleton_pixels) < 5:
            return 0.0

        radii = dist_map[skeleton_pixels]
        widths = 2 * radii

        mean_w = np.mean(widths)
        if mean_w < 1e-6:
            return 0.0

        return float(np.std(widths) / mean_w)

    # ==================================================================
    # STAGE 2 (mock) — for prototyping without trained weights
    # ==================================================================

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

    # ==================================================================
    # STAGE 4 — Crack-width measurement
    # ==================================================================

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
