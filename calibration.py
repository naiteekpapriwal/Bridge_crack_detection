"""
calibration.py — Automatic Pixel-to-MM Calibration System

Solves a real field engineering problem: instead of manually measuring
pixel_to_mm_ratio, this module automatically detects a known reference
object in the image and computes the calibration.

Supported calibration methods:
    1. ArUco Marker Detection  — Place a printed ArUco marker next to the
       crack. The system detects it, measures its pixel size, and
       computes the exact mm/pixel ratio from its known physical size.

    2. Reference Object (Circle) — Detect a circular reference object
       (e.g., a coin) of known diameter.

    3. Manual Fallback — If no reference is detected, use user-provided
       ratio with a clear warning.

The engineering value: field measurements become reproducible and
traceable, which is essential for structural monitoring over time.

Author: Naiteek Papriwal
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Calibration result
# =====================================================================

@dataclass
class CalibrationResult:
    """Result of an automatic calibration attempt."""
    pixel_to_mm: float              # The computed mm-per-pixel ratio
    method: str                     # "aruco", "circle", "manual"
    confidence: float               # 0.0 = fallback, 1.0 = high confidence
    reference_size_mm: float        # Known physical size of the reference
    reference_size_px: float        # Measured pixel size of the reference
    marker_location: Optional[tuple] = None   # (x, y) center of reference
    message: str = ""               # Human-readable description

    def to_dict(self) -> dict:
        """Serialise for JSON API response."""
        return {
            "pixel_to_mm": round(self.pixel_to_mm, 6),
            "method": self.method,
            "confidence": round(self.confidence, 2),
            "reference_size_mm": round(self.reference_size_mm, 2),
            "reference_size_px": round(self.reference_size_px, 2),
            "marker_location": self.marker_location,
            "message": self.message,
        }


# =====================================================================
# ArUco marker configuration
# =====================================================================

# Default ArUco dictionary — 4x4 markers with 50 IDs
# These are small, easy to print, and work well at typical bridge
# inspection distances.
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Default physical size of the ArUco marker (mm)
# Standard print size: 50mm × 50mm (easy to measure and verify)
DEFAULT_MARKER_SIZE_MM = 50.0

# Common reference coin diameters (mm) for circle-based calibration
KNOWN_COINS = {
    "indian_1_rupee": 25.0,
    "indian_2_rupee": 27.0,
    "indian_5_rupee": 23.0,
    "us_quarter": 24.26,
    "us_penny": 19.05,
    "euro_1": 23.25,
    "euro_2": 25.75,
}


# =====================================================================
# Main calibration engine
# =====================================================================

class AutoCalibrator:
    """
    Automatic pixel-to-mm calibration using reference objects.

    Usage:
        calibrator = AutoCalibrator(marker_size_mm=50.0)
        result = calibrator.calibrate(image)
        pixel_to_mm = result.pixel_to_mm
    """

    def __init__(
        self,
        marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
        coin_diameter_mm: float | None = None,
        fallback_ratio: float = 0.05,
    ):
        """
        Args:
            marker_size_mm:   Physical side length of the ArUco marker (mm).
            coin_diameter_mm: If using a coin as reference, its diameter (mm).
            fallback_ratio:   Default mm/pixel if no reference is detected.
        """
        self.marker_size_mm = marker_size_mm
        self.coin_diameter_mm = coin_diameter_mm
        self.fallback_ratio = fallback_ratio

        # Initialise ArUco detector
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self._aruco_params = cv2.aruco.DetectorParameters()

        # Tune detection parameters for robustness in field conditions
        # (variable lighting, distance, angle)
        self._aruco_params.adaptiveThreshWinSizeMin = 3
        self._aruco_params.adaptiveThreshWinSizeMax = 23
        self._aruco_params.adaptiveThreshWinSizeStep = 5
        self._aruco_params.minMarkerPerimeterRate = 0.02
        self._aruco_params.maxMarkerPerimeterRate = 4.0
        self._aruco_params.polygonalApproxAccuracyRate = 0.05

    def calibrate(self, image: np.ndarray) -> CalibrationResult:
        """
        Attempt automatic calibration on the given image.

        Tries methods in order of reliability:
            1. ArUco marker detection (most reliable)
            2. Circular reference object detection
            3. Manual fallback

        Args:
            image: BGR input image (uint8).

        Returns:
            CalibrationResult with the computed pixel_to_mm ratio.
        """
        # --- Method 1: ArUco marker ---
        aruco_result = self._detect_aruco(image)
        if aruco_result is not None:
            logger.info(
                "ArUco calibration successful: %.6f mm/px (marker %dpx)",
                aruco_result.pixel_to_mm,
                aruco_result.reference_size_px,
            )
            return aruco_result

        # --- Method 2: Circular reference (coin) ---
        if self.coin_diameter_mm is not None:
            circle_result = self._detect_circle_reference(image)
            if circle_result is not None:
                logger.info(
                    "Circle calibration successful: %.6f mm/px",
                    circle_result.pixel_to_mm,
                )
                return circle_result

        # --- Method 3: Manual fallback ---
        logger.warning(
            "No reference object detected — using fallback ratio %.4f mm/px. "
            "For accurate measurements, place an ArUco marker in the image.",
            self.fallback_ratio,
        )
        return CalibrationResult(
            pixel_to_mm=self.fallback_ratio,
            method="manual",
            confidence=0.0,
            reference_size_mm=0.0,
            reference_size_px=0.0,
            message=(
                "No reference object detected. Using default ratio "
                f"({self.fallback_ratio} mm/px). For accurate results, "
                "place a printed ArUco marker next to the crack."
            ),
        )

    # ==================================================================
    # Method 1: ArUco Marker Detection
    # ==================================================================

    def _detect_aruco(self, image: np.ndarray) -> CalibrationResult | None:
        """
        Detect ArUco markers and compute pixel_to_mm from marker size.

        Algorithm:
            1. Convert to grayscale
            2. Detect ArUco markers using OpenCV's detector
            3. Compute the perimeter of the detected marker (in pixels)
            4. Side length (px) = perimeter / 4
            5. pixel_to_mm = marker_size_mm / side_length_px

        Returns:
            CalibrationResult or None if no marker found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            logger.debug("No ArUco markers detected.")
            return None

        logger.info("Detected %d ArUco marker(s): IDs=%s", len(ids), ids.ravel())

        # Use the first detected marker
        marker_corners = corners[0][0]  # Shape: (4, 2) — four corner points

        # Compute side lengths (all 4 sides)
        side_lengths = []
        for i in range(4):
            p1 = marker_corners[i]
            p2 = marker_corners[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p2 - p1))

        # Average side length in pixels
        avg_side_px = np.mean(side_lengths)

        if avg_side_px < 5:
            logger.warning("Detected marker too small (%0.1f px) — ignoring.", avg_side_px)
            return None

        # Compute pixel_to_mm ratio
        pixel_to_mm = self.marker_size_mm / avg_side_px

        # Compute marker center
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))

        # Confidence based on marker quality:
        #   - How square is it? (std of side lengths should be low)
        #   - How large is it? (bigger = more reliable)
        side_std = np.std(side_lengths)
        squareness = 1.0 - min(side_std / avg_side_px, 0.5) * 2  # 1.0 = perfect square
        size_conf = min(avg_side_px / 100.0, 1.0)  # Bigger marker = more confident
        confidence = 0.5 * squareness + 0.5 * size_conf

        # Get marker ID safely (handle both 1D and 2D arrays)
        marker_id = int(ids.ravel()[0])

        return CalibrationResult(
            pixel_to_mm=pixel_to_mm,
            method="aruco",
            confidence=round(min(confidence, 1.0), 2),
            reference_size_mm=self.marker_size_mm,
            reference_size_px=round(avg_side_px, 1),
            marker_location=(center_x, center_y),
            message=(
                f"ArUco marker (ID {marker_id}) detected. "
                f"Marker side: {avg_side_px:.1f}px = {self.marker_size_mm}mm → "
                f"{pixel_to_mm:.6f} mm/px"
            ),
        )

    # ==================================================================
    # Method 2: Circular Reference Detection
    # ==================================================================

    def _detect_circle_reference(
        self, image: np.ndarray
    ) -> CalibrationResult | None:
        """
        Detect a circular reference object (e.g., coin) and calibrate.

        Uses Hough Circle Transform to find circles, then computes
        pixel_to_mm from the known physical diameter.

        Returns:
            CalibrationResult or None if no suitable circle found.
        """
        if self.coin_diameter_mm is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        h, w = gray.shape

        # Hough Circle detection
        # minRadius/maxRadius tuned for typical coin sizes at
        # bridge inspection distances
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=40,
            minRadius=max(15, int(min(h, w) * 0.02)),
            maxRadius=int(min(h, w) * 0.15),
        )

        if circles is None:
            logger.debug("No circular reference objects detected.")
            return None

        circles = np.round(circles[0]).astype(int)
        logger.info("Detected %d circle(s).", len(circles))

        # Use the most prominent circle (first one from HoughCircles)
        cx, cy, radius = circles[0]
        diameter_px = 2 * radius

        if diameter_px < 20:
            logger.warning("Circle too small (%dpx diameter) — ignoring.", diameter_px)
            return None

        pixel_to_mm = self.coin_diameter_mm / diameter_px

        # Confidence: based on how round and large the detection is
        confidence = min(diameter_px / 80.0, 1.0) * 0.8  # Circle detection is less reliable

        return CalibrationResult(
            pixel_to_mm=pixel_to_mm,
            method="circle",
            confidence=round(confidence, 2),
            reference_size_mm=self.coin_diameter_mm,
            reference_size_px=float(diameter_px),
            marker_location=(int(cx), int(cy)),
            message=(
                f"Circular reference detected (diameter: {diameter_px}px = "
                f"{self.coin_diameter_mm}mm) → {pixel_to_mm:.6f} mm/px"
            ),
        )

    # ==================================================================
    # Visualisation: Draw calibration reference on image
    # ==================================================================

    def draw_calibration_overlay(
        self,
        image: np.ndarray,
        result: CalibrationResult,
    ) -> np.ndarray:
        """
        Draw the detected reference object and calibration info on the image.

        Args:
            image:  BGR image to annotate.
            result: CalibrationResult from calibrate().

        Returns:
            Annotated image copy.
        """
        annotated = image.copy()

        if result.method == "manual":
            # No reference to draw — just add a warning text
            cv2.putText(
                annotated,
                "WARNING: No reference — using manual calibration",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return annotated

        if result.marker_location is None:
            return annotated

        cx, cy = result.marker_location

        if result.method == "aruco":
            # Draw a box around the marker area
            half_size = int(result.reference_size_px / 2)
            cv2.rectangle(
                annotated,
                (cx - half_size, cy - half_size),
                (cx + half_size, cy + half_size),
                (0, 255, 0),
                2,
            )
            label = f"ArUco: {result.reference_size_mm}mm"

        elif result.method == "circle":
            radius = int(result.reference_size_px / 2)
            cv2.circle(annotated, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)
            label = f"Ref: {result.reference_size_mm}mm"

        else:
            return annotated

        # Draw calibration info label
        cv2.putText(
            annotated,
            label,
            (cx - 60, cy - int(result.reference_size_px / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Scale: {result.pixel_to_mm:.4f} mm/px",
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return annotated


# =====================================================================
# Convenience function for quick calibration
# =====================================================================

def auto_calibrate(
    image: np.ndarray,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
    coin_diameter_mm: float | None = None,
    fallback_ratio: float = 0.05,
) -> CalibrationResult:
    """
    One-shot auto-calibration.

    Args:
        image:           BGR input image.
        marker_size_mm:  Physical size of ArUco marker if present.
        coin_diameter_mm: Physical diameter of coin if using coin reference.
        fallback_ratio:  Default mm/pixel if nothing is detected.

    Returns:
        CalibrationResult with the best available calibration.
    """
    calibrator = AutoCalibrator(
        marker_size_mm=marker_size_mm,
        coin_diameter_mm=coin_diameter_mm,
        fallback_ratio=fallback_ratio,
    )
    return calibrator.calibrate(image)
