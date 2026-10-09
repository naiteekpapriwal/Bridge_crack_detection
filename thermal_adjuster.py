"""
thermal_adjuster.py — Environmental Stress & Thermal Expansion Engine

Adjusts crack severity using real-world physics and live weather data.

Concrete is a thermally responsive material. Cracks behave differently
depending on ambient temperature, humidity, and freeze-thaw exposure:

    1. THERMAL EXPANSION: Concrete expands when heated, contracts when
       cooled. The coefficient of thermal expansion (CTE) for concrete
       is approximately 10–12 × 10⁻⁶ /°C (IS 456:2000, Clause 6.2.6).
       
       A crack measured at 40°C may appear narrower than its true
       structural width because the surrounding concrete has expanded
       and compressed the crack. At -5°C, the same crack opens wider
       as the concrete contracts.

    2. FREEZE-THAW DAMAGE: Water trapped in cracks expands ~9% when
       it freezes (0°C). Repeated freeze-thaw cycles progressively
       widen cracks — this is the #1 cause of concrete deterioration
       in cold climates (ACI 201.2R).

    3. HUMIDITY & CORROSION: When relative humidity exceeds 60%, the
       rate of carbonation-induced rebar corrosion increases sharply
       (Tuutti's model). Cracks exposed to high humidity are more
       dangerous because they allow moisture to reach the steel
       reinforcement.

This module fetches live weather from OpenWeatherMap and applies
thermodynamic corrections to both the crack width measurement and
the severity score.

Author: Naiteek Papriwal
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Physical constants — from IS 456:2000, ACI 318-19, and ACI 201.2R
# =====================================================================

# Coefficient of Thermal Expansion (CTE) for Portland cement concrete
# Range: 9.0 – 12.5 × 10⁻⁶ /°C depending on aggregate type.
# We use the IS 456 recommended value for general design.
CTE_CONCRETE = 10.0e-6  # per °C (IS 456, Clause 6.2.6)

# Reference temperature — standard lab conditions for material testing.
# Crack widths measured in literature are calibrated at ~20°C.
T_REFERENCE = 20.0  # °C

# Water expansion factor on freezing (by volume)
WATER_FREEZE_EXPANSION = 0.09  # 9% volume increase

# Carbonation-critical humidity threshold (Tuutti's corrosion model)
CARBONATION_RH_THRESHOLD = 60.0  # % relative humidity

# OpenWeatherMap API base URL (free tier)
OWM_API_URL = "https://api.openweathermap.org/data/2.5/weather"


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class WeatherData:
    """Live weather conditions at the inspection site."""
    temperature_c: float           # Current temperature in °C
    humidity_pct: float            # Relative humidity in %
    wind_speed_ms: float = 0.0    # Wind speed in m/s
    description: str = ""          # e.g. "clear sky", "light rain"
    city: str = ""                 # Location name
    country: str = ""              # Country code

    def to_dict(self) -> dict:
        return {
            "temperature_c": round(self.temperature_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "wind_speed_ms": round(self.wind_speed_ms, 1),
            "description": self.description,
            "city": self.city,
            "country": self.country,
        }


@dataclass
class ThermalAdjustment:
    """Result of thermal/environmental analysis."""
    # --- Weather ---
    weather: WeatherData

    # --- Width correction ---
    measured_width_mm: float        # Original crack width (at ambient T)
    normalized_width_mm: float      # Width corrected to reference T (20°C)
    thermal_strain: float           # ε = α × ΔT
    width_correction_mm: float      # How much the width was adjusted

    # --- Risk factors ---
    thermal_risk_score: float       # 0–100 temperature-based risk
    humidity_risk_score: float      # 0–100 humidity-based risk
    freeze_thaw_risk: float         # 0–100 freeze-thaw risk
    environmental_risk_score: float # Combined environmental risk (0–100)

    # --- Severity adjustment ---
    severity_multiplier: float      # Multiplier applied to severity score
    adjusted_severity_score: float  # Final severity after environmental adj.
    original_severity_score: float  # Severity before adjustment

    # --- Human-readable ---
    risk_level: str                 # "low", "moderate", "high", "critical"
    advisory: str                   # Engineering recommendation
    factors: list                   # List of active risk factors

    def to_dict(self) -> dict:
        return {
            "weather": self.weather.to_dict(),
            "measured_width_mm": round(self.measured_width_mm, 4),
            "normalized_width_mm": round(self.normalized_width_mm, 4),
            "thermal_strain": f"{self.thermal_strain:.2e}",
            "width_correction_mm": round(self.width_correction_mm, 4),
            "thermal_risk_score": round(self.thermal_risk_score, 1),
            "humidity_risk_score": round(self.humidity_risk_score, 1),
            "freeze_thaw_risk": round(self.freeze_thaw_risk, 1),
            "environmental_risk_score": round(self.environmental_risk_score, 1),
            "severity_multiplier": round(self.severity_multiplier, 3),
            "adjusted_severity_score": round(self.adjusted_severity_score, 1),
            "original_severity_score": round(self.original_severity_score, 1),
            "risk_level": self.risk_level,
            "advisory": self.advisory,
            "factors": self.factors,
        }


# =====================================================================
# Weather API Client
# =====================================================================

class WeatherClient:
    """Fetches live weather data from OpenWeatherMap."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def fetch(
        self,
        latitude: float,
        longitude: float,
    ) -> WeatherData | None:
        """
        Fetch current weather for given coordinates.

        Args:
            latitude:  Decimal degrees (e.g. 26.9124 for Jaipur).
            longitude: Decimal degrees (e.g. 75.7873 for Jaipur).

        Returns:
            WeatherData or None if the API call fails.
        """
        if not self.api_key:
            logger.warning("No OpenWeatherMap API key — using mock weather.")
            return None

        try:
            url = (
                f"{OWM_API_URL}"
                f"?lat={latitude}&lon={longitude}"
                f"&appid={self.api_key}"
                f"&units=metric"
            )

            req = urllib.request.Request(url, headers={"User-Agent": "BridgeCrackDetector/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

            weather = WeatherData(
                temperature_c=float(data["main"]["temp"]),
                humidity_pct=float(data["main"]["humidity"]),
                wind_speed_ms=float(data.get("wind", {}).get("speed", 0)),
                description=data.get("weather", [{}])[0].get("description", ""),
                city=data.get("name", ""),
                country=data.get("sys", {}).get("country", ""),
            )

            logger.info(
                "Weather fetched: %s, %s — %.1f°C, %d%% RH, %s",
                weather.city, weather.country,
                weather.temperature_c, weather.humidity_pct,
                weather.description,
            )
            return weather

        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            logger.error("Weather API error: %s", e)
            return None

    @staticmethod
    def mock_weather(temperature_c: float = 35.0, humidity_pct: float = 65.0) -> WeatherData:
        """
        Generate mock weather data for testing without an API key.

        Args:
            temperature_c: Simulated temperature.
            humidity_pct:  Simulated humidity.

        Returns:
            WeatherData with the provided values.
        """
        return WeatherData(
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            wind_speed_ms=3.2,
            description="simulated conditions",
            city="Test Location",
            country="--",
        )


# =====================================================================
# Thermal Expansion Engine — The Core Physics
# =====================================================================

class ThermalExpansionEngine:
    """
    Applies thermodynamic corrections to crack measurements.

    This is our own engineering algorithm — no library does this.

    Physics implemented:
        1. Thermal strain calculation:  ε = α × (T - T_ref)
        2. Width normalization to reference temperature
        3. Freeze-thaw risk assessment
        4. Humidity-driven corrosion risk (Tuutti's model)
        5. Combined environmental severity adjustment
    """

    def __init__(
        self,
        cte: float = CTE_CONCRETE,
        t_reference: float = T_REFERENCE,
        api_key: str | None = None,
    ):
        """
        Args:
            cte:         Coefficient of thermal expansion (/°C).
            t_reference: Reference temperature for normalization (°C).
            api_key:     OpenWeatherMap API key (optional).
        """
        self.cte = cte
        self.t_reference = t_reference
        self.weather_client = WeatherClient(api_key)

    def analyze(
        self,
        crack_width_mm: float,
        crack_length_px: float,
        severity_score: float,
        weather: WeatherData | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> ThermalAdjustment:
        """
        Full environmental analysis of a detected crack.

        Args:
            crack_width_mm:  Measured mean crack width (mm).
            crack_length_px: Crack length in pixels (from skeleton).
            severity_score:  Original severity score (0-100).
            weather:         Pre-fetched weather (skips API call).
            latitude:        Site latitude (used if weather is None).
            longitude:       Site longitude (used if weather is None).

        Returns:
            ThermalAdjustment with corrected values and risk assessment.
        """
        # --- 1. Get weather data ---
        if weather is None:
            if latitude is not None and longitude is not None:
                weather = self.weather_client.fetch(latitude, longitude)
            if weather is None:
                weather = self.weather_client.mock_weather()

        T = weather.temperature_c
        RH = weather.humidity_pct

        # --- 2. Thermal strain calculation ---
        # ε = α × ΔT
        # This is the fractional change in length per unit length
        delta_T = T - self.t_reference
        thermal_strain = self.cte * delta_T

        # --- 3. Normalize crack width to reference temperature ---
        # When concrete heats up (T > T_ref), it expands, compressing
        # cracks shut slightly. The measured width underestimates the
        # true structural width.
        #
        # When concrete cools (T < T_ref), it contracts, pulling cracks
        # open. The measured width overestimates the resting width.
        #
        # We use the crack's own length as the "gauge length" for the
        # thermal displacement. The influence zone of a crack is
        # approximately proportional to its length.
        #
        # Correction formula (our own derivation):
        #   w_structural = w_measured + α × ΔT × L_influence
        #
        # where L_influence is a fraction of crack length that
        # contributes to thermal displacement at the crack opening.
        # We use an empirical influence factor of 0.001 (calibrated
        # to typical concrete crack spacings per Eurocode 2, §7.3.4).

        influence_factor = 0.001  # Fraction of crack length contributing
        width_correction = thermal_strain * crack_length_px * influence_factor
        normalized_width = crack_width_mm + width_correction

        # Ensure normalized width is physically reasonable
        normalized_width = max(0.01, normalized_width)

        # --- 4. Thermal risk scoring ---
        thermal_risk = self._compute_thermal_risk(T)

        # --- 5. Humidity / corrosion risk scoring ---
        humidity_risk = self._compute_humidity_risk(RH)

        # --- 6. Freeze-thaw risk ---
        freeze_thaw = self._compute_freeze_thaw_risk(T, RH)

        # --- 7. Combined environmental risk ---
        # Weighted combination reflecting structural engineering priority:
        #   Freeze-thaw is the most destructive mechanism (ACI 201.2R)
        #   followed by corrosion (humidity), then thermal stress.
        environmental_risk = (
            0.45 * freeze_thaw
            + 0.30 * humidity_risk
            + 0.25 * thermal_risk
        )

        # --- 8. Compute severity multiplier ---
        # The environmental risk can increase severity by up to 40%
        # (a crack in freezing rain is 1.4× worse than the same crack
        # in mild dry conditions).
        severity_multiplier = 1.0 + (environmental_risk / 100.0) * 0.40
        adjusted_severity = min(100.0, severity_score * severity_multiplier)

        # --- 9. Risk level classification ---
        risk_level, advisory = self._classify_risk(
            environmental_risk, T, RH, freeze_thaw
        )

        # --- 10. List active risk factors ---
        factors = self._list_active_factors(T, RH, thermal_strain)

        result = ThermalAdjustment(
            weather=weather,
            measured_width_mm=crack_width_mm,
            normalized_width_mm=normalized_width,
            thermal_strain=thermal_strain,
            width_correction_mm=width_correction,
            thermal_risk_score=thermal_risk,
            humidity_risk_score=humidity_risk,
            freeze_thaw_risk=freeze_thaw,
            environmental_risk_score=environmental_risk,
            severity_multiplier=severity_multiplier,
            adjusted_severity_score=adjusted_severity,
            original_severity_score=severity_score,
            risk_level=risk_level,
            advisory=advisory,
            factors=factors,
        )

        logger.info(
            "Thermal analysis: T=%.1f°C, RH=%.0f%%, strain=%.2e, "
            "width %.4f→%.4f mm, severity %.1f→%.1f (×%.3f), risk=%s",
            T, RH, thermal_strain,
            crack_width_mm, normalized_width,
            severity_score, adjusted_severity, severity_multiplier,
            risk_level,
        )

        return result

    # ==================================================================
    # Risk scoring functions — our own engineering models
    # ==================================================================

    @staticmethod
    def _compute_thermal_risk(temperature_c: float) -> float:
        """
        Score temperature-related structural risk (0-100).

        Based on IS 456:2000 and ACI 209R concrete behavior curves:
        - Extreme heat (>45°C): Risk of thermal cracking and concrete
          strength reduction. Compressive strength drops ~10-20% above
          50°C (ACI 216.1).
        - Extreme cold (<-10°C): Risk of ice lens formation in pores,
          causing internal microcracking.
        - Optimal range (10-30°C): Minimal thermal stress.

        The risk curve is V-shaped around the reference temperature,
        with steeper rise below freezing due to ice damage.
        """
        T = temperature_c

        if T < -15:
            return 100.0  # Extreme cold — maximum thermal risk
        elif T < 0:
            # Below freezing: steep rise (ice formation)
            return 50.0 + (0 - T) / 15.0 * 50.0
        elif T < 5:
            # Near-freezing: significant risk
            return 30.0 + (5 - T) / 5.0 * 20.0
        elif T < 10:
            # Cool: mild thermal stress from contraction
            return 15.0 + (10 - T) / 5.0 * 15.0
        elif T < 30:
            # Optimal range: minimal risk
            return max(0.0, abs(T - 20) / 10.0 * 15.0)
        elif T < 40:
            # Hot: moderate expansion stress
            return 15.0 + (T - 30) / 10.0 * 25.0
        elif T < 50:
            # Very hot: significant thermal stress
            return 40.0 + (T - 40) / 10.0 * 30.0
        else:
            # Extreme heat: concrete strength loss
            return min(100.0, 70.0 + (T - 50) / 10.0 * 30.0)

    @staticmethod
    def _compute_humidity_risk(humidity_pct: float) -> float:
        """
        Score humidity-driven corrosion risk (0-100).

        Based on Tuutti's corrosion initiation model and
        Papadakis' carbonation model:

        - RH < 40%: Too dry for significant corrosion. Carbonation
          rate is also low (insufficient moisture for CO₂ reaction).
        - RH 40-60%: Carbonation progresses but corrosion rate is low.
        - RH 60-80%: Critical zone — carbonation rate peaks around
          65% RH (Papadakis et al., 1991). Corrosion accelerates.
        - RH > 80%: Maximum corrosion current density. Pore water
          facilitates ionic transport to the steel surface.
        - RH > 95%: Standing water — risk of chloride ingress and
          accelerated pitting corrosion.
        """
        RH = humidity_pct

        if RH < 40:
            return 5.0
        elif RH < 60:
            # Below carbonation threshold
            return 5.0 + (RH - 40) / 20.0 * 15.0
        elif RH < 75:
            # Carbonation zone
            return 20.0 + (RH - 60) / 15.0 * 30.0
        elif RH < 85:
            # High corrosion zone
            return 50.0 + (RH - 75) / 10.0 * 25.0
        elif RH < 95:
            # Very high — active corrosion
            return 75.0 + (RH - 85) / 10.0 * 15.0
        else:
            # Standing water / monsoon conditions
            return 90.0 + min(10.0, (RH - 95) / 5.0 * 10.0)

    @staticmethod
    def _compute_freeze_thaw_risk(temperature_c: float, humidity_pct: float) -> float:
        """
        Score freeze-thaw damage risk (0-100).

        Freeze-thaw is the most destructive environmental mechanism
        for cracked concrete (ACI 201.2R-16):

        - Water in cracks expands 9% upon freezing
        - Each freeze-thaw cycle widens cracks progressively
        - Risk is highest when temperature hovers around 0°C with
          high humidity (water is present AND freezing)

        Our scoring model (original):
            risk = proximity_to_freezing × moisture_availability
        """
        T = temperature_c
        RH = humidity_pct

        # Moisture availability factor (0-1)
        # Water must be present in the crack for freeze-thaw damage
        moisture = min(1.0, max(0.0, (RH - 30) / 70.0))

        # Proximity to freezing factor (0-1)
        # Maximum risk at 0°C, drops off as you move away
        if T > 10:
            proximity = 0.0  # Too warm for freezing
        elif T > 5:
            proximity = (10 - T) / 10.0 * 0.3  # Getting cool
        elif T > 0:
            proximity = 0.3 + (5 - T) / 5.0 * 0.7  # Near freezing — peak risk
        elif T > -5:
            proximity = 1.0  # At/below freezing — ice is forming
        elif T > -15:
            proximity = 0.9  # Deep freeze — less cycling but still damaging
        else:
            proximity = 0.7  # Extreme cold — ice is stable (less cycling damage)

        # Combined risk
        risk = proximity * moisture * 100.0

        return min(100.0, risk)

    @staticmethod
    def _classify_risk(
        environmental_risk: float,
        temperature_c: float,
        humidity_pct: float,
        freeze_thaw_risk: float,
    ) -> tuple[str, str]:
        """
        Map environmental risk score to a human-readable classification
        with engineering advisory text.
        """
        # Special case: active freeze-thaw
        if freeze_thaw_risk > 60:
            return "critical", (
                "⚠️ FREEZE-THAW ALERT: Current conditions indicate active "
                "freeze-thaw cycling risk. Water in cracks will expand ~9% "
                "upon freezing, progressively widening cracks. Recommend "
                "immediate crack sealing with flexible epoxy and drainage "
                "improvement to prevent water ingress. (Ref: ACI 201.2R-16)"
            )

        if environmental_risk >= 65:
            return "high", (
                f"High environmental stress detected (T={temperature_c:.0f}°C, "
                f"RH={humidity_pct:.0f}%). Thermal expansion and moisture "
                f"are actively accelerating crack deterioration. Schedule "
                f"priority repair and apply protective coating to prevent "
                f"carbonation-induced corrosion. (Ref: IS 456, Clause 8.2.2)"
            )
        elif environmental_risk >= 35:
            return "moderate", (
                f"Moderate environmental stress (T={temperature_c:.0f}°C, "
                f"RH={humidity_pct:.0f}%). Conditions may accelerate "
                f"crack propagation over time. Consider surface sealing "
                f"and schedule follow-up inspection in 3-6 months."
            )
        elif environmental_risk >= 15:
            return "low", (
                f"Low environmental risk (T={temperature_c:.0f}°C, "
                f"RH={humidity_pct:.0f}%). Conditions are within normal "
                f"range. Standard monitoring schedule is adequate."
            )
        else:
            return "minimal", (
                f"Minimal environmental stress (T={temperature_c:.0f}°C, "
                f"RH={humidity_pct:.0f}%). Conditions are near-optimal "
                f"for concrete. No environmental adjustment needed."
            )

    @staticmethod
    def _list_active_factors(
        temperature_c: float,
        humidity_pct: float,
        thermal_strain: float,
    ) -> list[str]:
        """
        List all active environmental risk factors for display.
        """
        factors = []

        if temperature_c < 0:
            factors.append(f"🧊 Below freezing ({temperature_c:.1f}°C) — ice expansion risk")
        elif temperature_c < 5:
            factors.append(f"❄️ Near-freezing ({temperature_c:.1f}°C) — freeze-thaw cycling risk")
        elif temperature_c > 40:
            factors.append(f"🔥 Extreme heat ({temperature_c:.1f}°C) — thermal stress + strength reduction")
        elif temperature_c > 30:
            factors.append(f"☀️ High temperature ({temperature_c:.1f}°C) — thermal expansion active")

        if humidity_pct > 85:
            factors.append(f"💧 Very high humidity ({humidity_pct:.0f}%) — active corrosion risk")
        elif humidity_pct > 60:
            factors.append(f"💦 Elevated humidity ({humidity_pct:.0f}%) — carbonation zone")

        if abs(thermal_strain) > 200e-6:
            factors.append(f"📐 Significant thermal strain ({thermal_strain:.2e}) — width correction applied")
        elif abs(thermal_strain) > 50e-6:
            factors.append(f"📏 Moderate thermal strain ({thermal_strain:.2e}) — minor width correction")

        if not factors:
            factors.append("✅ No significant environmental risk factors detected")

        return factors


# =====================================================================
# Convenience function
# =====================================================================

def compute_thermal_adjustment(
    crack_width_mm: float,
    crack_length_px: float,
    severity_score: float,
    latitude: float | None = None,
    longitude: float | None = None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    api_key: str | None = None,
) -> ThermalAdjustment:
    """
    One-shot environmental analysis.

    Provide either (latitude, longitude) for live weather, or
    (temperature_c, humidity_pct) for manual/mock input.

    Args:
        crack_width_mm:  Measured crack width.
        crack_length_px: Crack length in pixels.
        severity_score:  Original severity score (0-100).
        latitude:        Site latitude (for API weather).
        longitude:       Site longitude (for API weather).
        temperature_c:   Manual temperature override.
        humidity_pct:    Manual humidity override.
        api_key:         OpenWeatherMap API key.

    Returns:
        ThermalAdjustment with full environmental analysis.
    """
    engine = ThermalExpansionEngine(api_key=api_key)

    # Use manual weather if coordinates aren't provided
    weather = None
    if temperature_c is not None and humidity_pct is not None:
        weather = WeatherData(
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            description="manual input",
            city="User-specified",
        )

    return engine.analyze(
        crack_width_mm=crack_width_mm,
        crack_length_px=crack_length_px,
        severity_score=severity_score,
        weather=weather,
        latitude=latitude,
        longitude=longitude,
    )
