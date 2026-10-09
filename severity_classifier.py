"""
severity_classifier.py — Crack Severity Classification Engine

A custom multi-factor scoring algorithm that classifies detected cracks
into structural severity levels based on real engineering standards
(IS 456:2000, ACI 318-19).

Unlike YOLO (which just says "crack"), this module answers the question
every structural engineer actually needs answered:
    "HOW DANGEROUS is this crack?"

Scoring formula (our own design):
    severity_score =  W1 * width_factor
                    + W2 * length_factor
                    + W3 * branching_factor
                    + W4 * density_factor

Each factor is normalised to [0, 100] and weighted by domain importance.
The final score maps to one of four severity levels with engineering
recommendations drawn from IS 456 Table 3 permissible crack widths.

Author: Naiteek Papriwal
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Severity levels — based on IS 456:2000 & ACI 318-19 crack limits
# =====================================================================

@dataclass
class SeverityLevel:
    """Represents a structural severity classification."""
    level: str              # "hairline", "minor", "moderate", "severe"
    label: str              # Human-readable label
    score_range: tuple      # (min_score, max_score)
    color: tuple            # BGR colour for visualisation
    action: str             # Engineering recommendation
    urgency: str            # Timeline for action


# Severity lookup table — our own engineering-informed classification
SEVERITY_LEVELS = [
    SeverityLevel(
        level="hairline",
        label="Hairline Crack",
        score_range=(0, 25),
        color=(0, 200, 0),          # Green
        action="Monitor periodically. No structural concern at this stage.",
        urgency="Routine inspection (6-12 months)",
    ),
    SeverityLevel(
        level="minor",
        label="Minor Crack",
        score_range=(25, 50),
        color=(0, 200, 255),        # Yellow/Orange
        action=(
            "Schedule surface sealing. Apply epoxy injection if crack "
            "is in a moisture-exposed zone to prevent rebar corrosion."
        ),
        urgency="Planned maintenance (1-3 months)",
    ),
    SeverityLevel(
        level="moderate",
        label="Moderate Crack",
        score_range=(50, 75),
        color=(0, 100, 255),        # Orange/Red
        action=(
            "Urgent structural assessment required. Perform core sampling "
            "to check depth. Apply structural repair (carbon fibre wrapping "
            "or stitching) if crack penetrates beyond cover concrete."
        ),
        urgency="Urgent repair (within 2 weeks)",
    ),
    SeverityLevel(
        level="severe",
        label="Severe Structural Crack",
        score_range=(75, 100),
        color=(0, 0, 255),          # Red
        action=(
            "IMMEDIATE structural evaluation by a licensed engineer. "
            "Consider load restriction or temporary closure. "
            "Full structural rehabilitation likely required."
        ),
        urgency="Immediate action (within 24-48 hours)",
    ),
]


# =====================================================================
# Factor weights — tuned based on structural engineering significance
# =====================================================================
# Width is the most critical factor in IS 456 (Table 3 limits are
# defined entirely by crack width). Branching and length are secondary
# indicators of structural distress.

WEIGHTS = {
    "width":      0.40,   # Most important — IS 456 limits are width-based
    "length":     0.25,   # Longer cracks indicate larger stress fields
    "branching":  0.20,   # Branching = stress redistribution = worse
    "density":    0.15,   # Dense cracking in a small area = deterioration
}


# =====================================================================
# Width thresholds (mm) — from IS 456:2000 Table 3
# =====================================================================
# IS 456 defines permissible crack widths:
#   - Mild exposure:    0.3 mm
#   - Moderate exposure: 0.2 mm
#   - Severe exposure:  0.1 mm
# We use these as anchor points for our scoring curve.

WIDTH_THRESHOLDS_MM = {
    "safe":     0.1,    # Below this = hairline, no concern
    "caution":  0.3,    # IS 456 mild exposure limit
    "danger":   1.0,    # Beyond this = severe structural issue
    "critical": 2.0,    # Upper bound for scoring saturation
}


# =====================================================================
# Main classification engine
# =====================================================================

class CrackSeverityClassifier:
    """
    Multi-factor crack severity scoring engine.

    Analyses a detected crack's geometry and topology to produce a
    severity score (0-100) and classification (hairline → severe).
    """

    def __init__(self, weights: dict | None = None):
        """
        Initialise with optional custom weights.

        Args:
            weights: Dict with keys 'width', 'length', 'branching',
                     'density'. Values should sum to 1.0.
        """
        self.weights = weights or WEIGHTS.copy()
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Weights sum to %.2f (expected 1.0) — normalising.", total
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

    def classify(
        self,
        mask: np.ndarray,
        measurements: dict,
        image_shape: tuple,
    ) -> dict:
        """
        Classify a single crack's severity.

        Args:
            mask:          Binary mask (uint8, 0 or 255) of the crack.
            measurements:  Dict from CrackDetector.calculate_measurements().
            image_shape:   (H, W, C) shape of the original image.

        Returns:
            Dict containing:
                - severity_score (float 0-100)
                - severity_level (str)
                - severity_label (str)
                - severity_color (tuple BGR)
                - action (str)
                - urgency (str)
                - factor_breakdown (dict of individual factor scores)
        """
        # --- Compute each factor (0-100) ---
        width_score = self._score_width(measurements)
        length_score = self._score_length(measurements, image_shape)
        branching_score = self._score_branching(mask)
        density_score = self._score_density(mask, measurements)

        # --- Weighted combination ---
        severity_score = (
            self.weights["width"]     * width_score
            + self.weights["length"]    * length_score
            + self.weights["branching"] * branching_score
            + self.weights["density"]   * density_score
        )

        # Clamp to [0, 100]
        severity_score = max(0.0, min(100.0, severity_score))

        # --- Map to severity level ---
        severity = self._lookup_severity(severity_score)

        factor_breakdown = {
            "width_score": round(width_score, 1),
            "length_score": round(length_score, 1),
            "branching_score": round(branching_score, 1),
            "density_score": round(density_score, 1),
            "weights": self.weights,
        }

        result = {
            "severity_score": round(severity_score, 1),
            "severity_level": severity.level,
            "severity_label": severity.label,
            "severity_color": severity.color,
            "action": severity.action,
            "urgency": severity.urgency,
            "factor_breakdown": factor_breakdown,
        }

        logger.info(
            "Severity: %.1f/100 [%s] — width=%.0f, length=%.0f, "
            "branch=%.0f, density=%.0f",
            severity_score, severity.level,
            width_score, length_score, branching_score, density_score,
        )

        return result

    # ==================================================================
    # Factor 1: Width Score
    # ==================================================================

    @staticmethod
    def _score_width(measurements: dict) -> float:
        """
        Score crack width against IS 456 thresholds.

        Uses a piecewise-linear mapping:
            0.0 mm → 0
            0.1 mm → 25   (IS 456 severe exposure limit)
            0.3 mm → 50   (IS 456 mild exposure limit)
            1.0 mm → 75   (structural concern threshold)
            2.0 mm → 100  (severe structural failure)

        We use mean_width_mm as primary, boosted by max_width_mm.
        """
        mean_w = measurements.get("mean_width_mm", 0.0)
        max_w = measurements.get("max_width_mm", 0.0)

        # Use a weighted combination: mean is the baseline, max boosts it
        # because a single very wide point indicates structural weakness
        effective_width = 0.7 * mean_w + 0.3 * max_w

        # Piecewise linear mapping
        thresholds = [
            (0.0,  0.0),
            (WIDTH_THRESHOLDS_MM["safe"],     25.0),
            (WIDTH_THRESHOLDS_MM["caution"],  50.0),
            (WIDTH_THRESHOLDS_MM["danger"],   75.0),
            (WIDTH_THRESHOLDS_MM["critical"], 100.0),
        ]

        return _piecewise_linear(effective_width, thresholds)

    # ==================================================================
    # Factor 2: Length Score
    # ==================================================================

    @staticmethod
    def _score_length(measurements: dict, image_shape: tuple) -> float:
        """
        Score crack length relative to the image/structure size.

        Rationale: A crack spanning 80% of a beam is far more concerning
        than a short 5% crack, even if both have the same width.

        Scoring:
            < 5% of image diagonal  → 0-20  (localised, minor)
            5-15%                   → 20-50 (moderate extent)
            15-40%                  → 50-80 (significant extent)
            > 40%                   → 80-100 (extensive damage)
        """
        crack_length_px = measurements.get("crack_length_px", 0)
        h, w = image_shape[:2]
        image_diagonal = np.sqrt(h**2 + w**2)

        if image_diagonal < 1:
            return 0.0

        length_ratio = crack_length_px / image_diagonal

        thresholds = [
            (0.00,  0.0),
            (0.05,  20.0),
            (0.15,  50.0),
            (0.40,  80.0),
            (0.70,  100.0),
        ]

        return _piecewise_linear(length_ratio, thresholds)

    # ==================================================================
    # Factor 3: Branching Score (Topology Analysis)
    # ==================================================================

    @staticmethod
    def _score_branching(mask: np.ndarray) -> float:
        """
        Score crack branching by counting skeleton junction points.

        Branching indicates stress redistribution — the structure is
        finding new paths to relieve stress, which means the original
        load path is compromised.

        Algorithm (our own):
            1. Skeletonise the crack mask → 1px centerline
            2. For each skeleton pixel, count its 8-connected neighbours
            3. Pixels with ≥ 3 neighbours are JUNCTION POINTS (Y/T/X shapes)
            4. Score based on junction density per unit crack length

        No library function does this — it's our own topological analysis.
        """
        # Skeletonise
        skeleton = _skeletonize(mask)
        skeleton_binary = (skeleton > 0).astype(np.uint8)

        skeleton_px = np.count_nonzero(skeleton_binary)
        if skeleton_px < 10:
            return 0.0

        # Count neighbours for each skeleton pixel using convolution
        # A 3×3 kernel of ones, minus the center pixel
        neighbour_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=np.uint8)

        neighbour_count = cv2.filter2D(
            skeleton_binary, ddepth=-1, kernel=neighbour_kernel
        )

        # Junction points: skeleton pixels with ≥ 3 neighbours
        junction_mask = (neighbour_count >= 3) & (skeleton_binary > 0)
        num_junctions = np.count_nonzero(junction_mask)

        # Endpoint points: skeleton pixels with exactly 1 neighbour
        endpoint_mask = (neighbour_count == 1) & (skeleton_binary > 0)
        num_endpoints = np.count_nonzero(endpoint_mask)

        # --- Scoring ---
        # Junction density = junctions per 100 skeleton pixels
        junction_density = (num_junctions / skeleton_px) * 100

        # A simple crack has 0 junctions (2 endpoints, 1 path).
        # Each junction adds a branch → exponentially worse.
        # Scoring curve:
        #   0 junctions/100px → 0
        #   1 junction/100px  → 30
        #   3 junctions/100px → 60
        #   5+ junctions/100px → 90-100

        thresholds = [
            (0.0,  0.0),
            (1.0,  30.0),
            (3.0,  60.0),
            (5.0,  90.0),
            (8.0,  100.0),
        ]

        score = _piecewise_linear(junction_density, thresholds)

        logger.debug(
            "Branching: %d junctions, %d endpoints, %d skeleton_px, "
            "density=%.2f/100px → score=%.0f",
            num_junctions, num_endpoints, skeleton_px,
            junction_density, score,
        )

        return score

    # ==================================================================
    # Factor 4: Area Density Score
    # ==================================================================

    @staticmethod
    def _score_density(mask: np.ndarray, measurements: dict) -> float:
        """
        Score how densely the crack fills its bounding box.

        Rationale: A crack that fills a large fraction of its bounding
        region indicates widespread deterioration (e.g., map cracking,
        alligator cracking) vs a single thin line.

        Also considers absolute area — a crack covering a large absolute
        area of the image is more concerning.

        Scoring combines:
            - Fill ratio: crack_area / bbox_area (how dense within bbox)
            - Image coverage: crack_area / image_area (absolute extent)
        """
        crack_area = measurements.get("crack_area_px", 0)
        if crack_area == 0:
            return 0.0

        # Compute bounding box area
        coords = cv2.findNonZero(mask)
        if coords is None:
            return 0.0

        _, _, bw, bh = cv2.boundingRect(coords)
        bbox_area = max(bw * bh, 1)

        # Fill ratio — how much of the bounding box is cracked
        fill_ratio = crack_area / bbox_area

        # Image coverage — what fraction of total image is cracked
        image_area = mask.shape[0] * mask.shape[1]
        coverage = crack_area / max(image_area, 1)

        # Scoring: combine fill ratio and coverage
        # High fill ratio = dense cracking (map/alligator pattern)
        # High coverage = large-scale damage

        fill_score_thresholds = [
            (0.00,  0.0),
            (0.10,  20.0),   # Thin crack in large bbox
            (0.30,  50.0),   # Moderate fill
            (0.60,  80.0),   # Dense cracking
            (0.80,  100.0),  # Almost solid fill = severe spalling
        ]

        coverage_score_thresholds = [
            (0.000,  0.0),
            (0.005,  15.0),
            (0.020,  40.0),
            (0.050,  70.0),
            (0.100,  100.0),
        ]

        fill_score = _piecewise_linear(fill_ratio, fill_score_thresholds)
        coverage_score = _piecewise_linear(coverage, coverage_score_thresholds)

        # Weighted combination: fill ratio matters more than raw coverage
        combined = 0.6 * fill_score + 0.4 * coverage_score

        logger.debug(
            "Density: fill_ratio=%.3f (score=%.0f), coverage=%.4f "
            "(score=%.0f) → combined=%.0f",
            fill_ratio, fill_score, coverage, coverage_score, combined,
        )

        return combined

    # ==================================================================
    # Severity level lookup
    # ==================================================================

    @staticmethod
    def _lookup_severity(score: float) -> SeverityLevel:
        """Map a score (0-100) to a SeverityLevel."""
        for level in SEVERITY_LEVELS:
            lo, hi = level.score_range
            if lo <= score < hi:
                return level
        # Score exactly 100 → severe
        return SEVERITY_LEVELS[-1]


# =====================================================================
# Batch classification — analyse multiple cracks together
# =====================================================================

def classify_all_detections(
    detections: list[dict],
    image_shape: tuple,
    classifier: CrackSeverityClassifier | None = None,
) -> list[dict]:
    """
    Add severity classification to every detection in a list.

    Modifies each detection dict in-place by adding a 'severity' key.

    Args:
        detections:  List of detection dicts (must have 'mask' and
                     'measurements' keys).
        image_shape: (H, W, C) of the original image.
        classifier:  Optional pre-configured classifier instance.

    Returns:
        The same list, with 'severity' added to each detection.
    """
    if classifier is None:
        classifier = CrackSeverityClassifier()

    for det in detections:
        mask = det.get("mask")
        measurements = det.get("measurements", {})

        if mask is None:
            det["severity"] = {
                "severity_score": 0.0,
                "severity_level": "unknown",
                "severity_label": "Unknown",
                "action": "No mask available for analysis.",
                "urgency": "N/A",
            }
            continue

        severity = classifier.classify(mask, measurements, image_shape)
        det["severity"] = severity

    return detections


# =====================================================================
# Helper utilities
# =====================================================================

def _piecewise_linear(x: float, thresholds: list[tuple]) -> float:
    """
    Piecewise-linear interpolation.

    Args:
        x:          Input value.
        thresholds: List of (input_val, output_val) breakpoints,
                    sorted by input_val ascending.

    Returns:
        Interpolated output value, clamped to the output range.
    """
    if x <= thresholds[0][0]:
        return thresholds[0][1]

    for i in range(1, len(thresholds)):
        x0, y0 = thresholds[i - 1]
        x1, y1 = thresholds[i]
        if x <= x1:
            # Linear interpolation between breakpoints
            t = (x - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.0
            return y0 + t * (y1 - y0)

    # Beyond last threshold → return last value
    return thresholds[-1][1]


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Reduce a binary mask to a 1-pixel-wide skeleton.

    Uses cv2.ximgproc.thinning if available, otherwise falls back
    to iterative morphological erosion.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255

    try:
        return cv2.ximgproc.thinning(
            mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except (AttributeError, cv2.error):
        pass

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
