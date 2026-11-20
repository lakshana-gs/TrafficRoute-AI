"""Congestion & Delay Intelligence Engine for TrafficRoute AI (Milestone P5).

Provides data-source-independent calculations for segment-level traffic state:
- Configurable ratio-based congestion level classification
- Deterministic segment travel time and signed delay calculations
- Defensive validation for zero speeds, non-finite values, and negative dimensions
- Strict canonical output contracts preserving genuine provenance without fabrication
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class CongestionLevel(str, Enum):
    """Categorical congestion classification tiers."""

    LOW = "low"  # Free-flow or near free-flow conditions (speed ratio >= 0.80)
    MODERATE = "moderate"  # Moderate traffic impedance (0.60 <= speed ratio < 0.80)
    HIGH = "high"  # High congestion / significant queuing (0.40 <= speed ratio < 0.60)
    SEVERE = "severe"  # Severe breakdown / heavy gridlock (speed ratio < 0.40)
    UNKNOWN = "unknown"  # Missing, invalid, or uncomputable traffic speed


class CongestionStatus(str, Enum):
    """Detailed operational and diagnostic status for segment calculations."""

    NORMAL = "NORMAL"  # Valid calculation under expected operating bounds
    STOPPED = "STOPPED"  # Legitimate zero speed (stopped traffic); travel time undefined
    FASTER_THAN_FREE_FLOW = "FASTER_THAN_FREE_FLOW"  # Speed exceeds reference free-flow speed
    MISSING_SPEED = "MISSING_SPEED"  # Speed input is None or NaN
    INVALID_SPEED = "INVALID_SPEED"  # Speed input is negative or infinite
    INVALID_FREE_FLOW_SPEED = "INVALID_FREE_FLOW_SPEED"  # Free-flow speed <= 0 or non-finite
    INVALID_SEGMENT_LENGTH = "INVALID_SEGMENT_LENGTH"  # Segment length <= 0 or non-finite


@dataclass(frozen=True)
class CongestionThresholdConfig:
    """Configurable threshold boundaries for congestion classification.
    
    Default heuristic boundaries:
      ratio >= 0.80: low
      0.60 <= ratio < 0.80: moderate
      0.40 <= ratio < 0.60: high
      ratio < 0.40: severe
    """

    low_min_ratio: float = 0.80
    moderate_min_ratio: float = 0.60
    high_min_ratio: float = 0.40

    def __post_init__(self) -> None:
        """Validate threshold ordering and sanity."""
        for name, val in [
            ("low_min_ratio", self.low_min_ratio),
            ("moderate_min_ratio", self.moderate_min_ratio),
            ("high_min_ratio", self.high_min_ratio),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Threshold '{name}' must be a finite float, got {val}")
            if val <= 0.0 or val > 1.0:
                raise ValueError(f"Threshold '{name}' must be in range (0.0, 1.0], got {val}")

        if not (self.high_min_ratio < self.moderate_min_ratio < self.low_min_ratio):
            raise ValueError(
                f"Thresholds must satisfy 0 < high_min_ratio ({self.high_min_ratio}) < "
                f"moderate_min_ratio ({self.moderate_min_ratio}) < "
                f"low_min_ratio ({self.low_min_ratio}) <= 1.0"
            )


class CongestionClassifier:
    """Pure, deterministic congestion level classification engine."""

    @staticmethod
    def classify(
        predicted_speed_kmh: Optional[float],
        free_flow_speed_kmh: Optional[float],
        config: Optional[CongestionThresholdConfig] = None,
    ) -> Tuple[CongestionLevel, CongestionStatus, Optional[float], str]:
        """Classify congestion level from predicted speed and reference free-flow speed.
        
        Args:
            predicted_speed_kmh: Predicted or observed speed in km/h.
            free_flow_speed_kmh: Road segment free-flow speed in km/h.
            config: Optional threshold configuration (uses defaults if None).
            
        Returns:
            Tuple of (CongestionLevel, CongestionStatus, Optional[speed_ratio], status_message).
        """
        cfg = config or CongestionThresholdConfig()

        # 1. Validate free-flow speed
        if (
            free_flow_speed_kmh is None
            or not isinstance(free_flow_speed_kmh, (int, float))
            or not math.isfinite(free_flow_speed_kmh)
            or free_flow_speed_kmh <= 0.0
        ):
            return (
                CongestionLevel.UNKNOWN,
                CongestionStatus.INVALID_FREE_FLOW_SPEED,
                None,
                f"Free-flow speed must be strictly positive and finite, got {free_flow_speed_kmh}",
            )

        # 2. Validate predicted speed missingness
        if predicted_speed_kmh is None or (
            isinstance(predicted_speed_kmh, float) and math.isnan(predicted_speed_kmh)
        ):
            return (
                CongestionLevel.UNKNOWN,
                CongestionStatus.MISSING_SPEED,
                None,
                "Predicted speed is missing or NaN",
            )

        # 3. Validate predicted speed finiteness and sign
        if (
            not isinstance(predicted_speed_kmh, (int, float))
            or not math.isfinite(predicted_speed_kmh)
            or predicted_speed_kmh < 0.0
        ):
            return (
                CongestionLevel.UNKNOWN,
                CongestionStatus.INVALID_SPEED,
                None,
                f"Predicted speed must be non-negative and finite, got {predicted_speed_kmh}",
            )

        speed = float(predicted_speed_kmh)
        ff_speed = float(free_flow_speed_kmh)
        ratio = speed / ff_speed

        # 4. Handle exact zero speed (stopped traffic)
        if speed == 0.0:
            return (
                CongestionLevel.SEVERE,
                CongestionStatus.STOPPED,
                0.0,
                "Traffic is stopped (0.0 km/h); severe congestion stop",
            )

        # 5. Handle speeds exceeding free-flow speed
        if speed > ff_speed:
            return (
                CongestionLevel.LOW,
                CongestionStatus.FASTER_THAN_FREE_FLOW,
                round(ratio, 6),
                f"Predicted speed ({speed:.1f} km/h) exceeds free-flow speed ({ff_speed:.1f} km/h)",
            )

        # 6. Apply threshold boundaries
        if ratio >= cfg.low_min_ratio:
            return (
                CongestionLevel.LOW,
                CongestionStatus.NORMAL,
                round(ratio, 6),
                f"Low congestion (ratio {ratio:.3f} >= {cfg.low_min_ratio})",
            )
        elif ratio >= cfg.moderate_min_ratio:
            return (
                CongestionLevel.MODERATE,
                CongestionStatus.NORMAL,
                round(ratio, 6),
                f"Moderate congestion ({cfg.moderate_min_ratio} <= ratio {ratio:.3f} < {cfg.low_min_ratio})",
            )
        elif ratio >= cfg.high_min_ratio:
            return (
                CongestionLevel.HIGH,
                CongestionStatus.NORMAL,
                round(ratio, 6),
                f"High congestion ({cfg.high_min_ratio} <= ratio {ratio:.3f} < {cfg.moderate_min_ratio})",
            )
        else:
            return (
                CongestionLevel.SEVERE,
                CongestionStatus.NORMAL,
                round(ratio, 6),
                f"Severe congestion (ratio {ratio:.3f} < {cfg.high_min_ratio})",
            )


class DelayCalculator:
    """Deterministic calculation of segment travel time and signed delay in minutes."""

    @staticmethod
    def calculate(
        segment_length_km: Optional[float],
        predicted_speed_kmh: Optional[float],
        free_flow_speed_kmh: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], CongestionStatus, str]:
        """Compute free-flow travel time, estimated travel time, and signed delay.
        
        Formula:
          free_flow_travel_time_minutes = 60.0 * L / v_free
          estimated_travel_time_minutes = 60.0 * L / v_pred   (for v_pred > 0)
          estimated_delay_minutes       = estimated_travel_time_minutes - free_flow_travel_time_minutes
          
        Args:
            segment_length_km: Segment length in kilometres (must be > 0).
            predicted_speed_kmh: Predicted or observed speed in km/h.
            free_flow_speed_kmh: Reference free-flow speed in km/h (must be > 0).
            
        Returns:
            Tuple of:
            - estimated_travel_time_minutes (Optional[float])
            - free_flow_travel_time_minutes (Optional[float])
            - estimated_delay_minutes (Optional[float], signed: positive=delay, negative=speedup)
            - CongestionStatus
            - status_message (str)
        """
        # 1. Validate segment length
        if (
            segment_length_km is None
            or not isinstance(segment_length_km, (int, float))
            or not math.isfinite(segment_length_km)
            or segment_length_km <= 0.0
        ):
            return (
                None,
                None,
                None,
                CongestionStatus.INVALID_SEGMENT_LENGTH,
                f"Segment length must be strictly positive and finite, got {segment_length_km}",
            )

        # 2. Validate free-flow speed
        if (
            free_flow_speed_kmh is None
            or not isinstance(free_flow_speed_kmh, (int, float))
            or not math.isfinite(free_flow_speed_kmh)
            or free_flow_speed_kmh <= 0.0
        ):
            return (
                None,
                None,
                None,
                CongestionStatus.INVALID_FREE_FLOW_SPEED,
                f"Free-flow speed must be strictly positive and finite, got {free_flow_speed_kmh}",
            )

        length = float(segment_length_km)
        ff_speed = float(free_flow_speed_kmh)

        # Baseline free-flow travel time is always computable and positive
        free_flow_time = 60.0 * length / ff_speed

        # 3. Validate predicted speed missingness
        if predicted_speed_kmh is None or (
            isinstance(predicted_speed_kmh, float) and math.isnan(predicted_speed_kmh)
        ):
            return (
                None,
                round(free_flow_time, 6),
                None,
                CongestionStatus.MISSING_SPEED,
                "Predicted speed is missing or NaN; travel time cannot be estimated",
            )

        # 4. Validate predicted speed finiteness and sign
        if (
            not isinstance(predicted_speed_kmh, (int, float))
            or not math.isfinite(predicted_speed_kmh)
            or predicted_speed_kmh < 0.0
        ):
            return (
                None,
                round(free_flow_time, 6),
                None,
                CongestionStatus.INVALID_SPEED,
                f"Predicted speed must be non-negative and finite, got {predicted_speed_kmh}",
            )

        speed = float(predicted_speed_kmh)

        # 5. Handle zero speed: travel time is undefined (division by zero avoided)
        if speed == 0.0:
            return (
                None,
                round(free_flow_time, 6),
                None,
                CongestionStatus.STOPPED,
                "Predicted speed is 0.0 km/h (traffic stopped); finite travel time is undefined",
            )

        # 6. Standard positive speed calculation
        travel_time = 60.0 * length / speed
        raw_delay = travel_time - free_flow_time

        status = CongestionStatus.FASTER_THAN_FREE_FLOW if speed > ff_speed else CongestionStatus.NORMAL
        msg = "Travel time and delay calculated successfully"
        if status == CongestionStatus.FASTER_THAN_FREE_FLOW:
            msg = f"Speed exceeds free-flow; signed delay is negative ({raw_delay:.3f} min savings)"

        return (
            round(travel_time, 6),
            round(free_flow_time, 6),
            round(raw_delay, 6),
            status,
            msg,
        )


class SegmentCongestionResult(BaseModel):
    """Canonical schema for segment-level congestion and delay intelligence results.
    
    Adheres strictly to the P5 output contract with verified provenance, explicit
    status flags, and zero fabricated metadata.
    """

    segment_id: str = Field(..., description="Unique road segment identifier (e.g., edge ID or sensor ID)")
    forecast_timestamp: Optional[datetime] = Field(
        default=None,
        description="Forecast timestamp when genuinely available (remains None if unavailable)",
    )
    forecast_horizon_minutes: Optional[int] = Field(
        default=None,
        description="Forecast horizon in minutes (e.g. 5, 15, 30, 60; remains None if not a horizon forecast)",
    )
    predicted_speed_kmh: Optional[float] = Field(
        default=None,
        description="Predicted or observed traffic speed in km/h (None if missing/invalid)",
    )
    free_flow_speed_kmh: Optional[float] = Field(
        default=None,
        description="Baseline free-flow reference speed in km/h (None if invalid or unavailable)",
    )
    speed_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of predicted speed to free-flow speed (v_pred / v_free)",
    )
    congestion_level: CongestionLevel = Field(
        ...,
        description="Categorical congestion level: low, moderate, high, severe, unknown",
    )
    congestion_status: CongestionStatus = Field(
        ...,
        description="Detailed operational classification status",
    )
    segment_length_km: Optional[float] = Field(
        default=None,
        description="Physical road segment length in kilometres (None if invalid or unavailable)",
    )
    estimated_travel_time_minutes: Optional[float] = Field(
        default=None,
        description="Estimated travel time in minutes (None if speed is 0, missing, or invalid)",
    )
    free_flow_travel_time_minutes: Optional[float] = Field(
        default=None,
        description="Theoretical baseline travel time under free-flow conditions in minutes (None if dimensions invalid)",
    )
    estimated_delay_minutes: Optional[float] = Field(
        default=None,
        description="Signed estimated delay in minutes (travel_time - free_flow_time; negative indicates speedup)",
    )
    model_source: str = Field(
        default="UNSPECIFIED",
        description="Originating model, benchmark, or telemetry source identifier",
    )
    source_geography: str = Field(
        default="UNSPECIFIED",
        description="Originating geographic scope (e.g. 'Los Angeles County, CA' vs 'Coimbatore, Tamil Nadu')",
    )
    is_valid: bool = Field(
        default=True,
        description="True if calculation succeeded with valid inputs; False on validation failure",
    )
    status_message: str = Field(
        default="",
        description="Diagnostic explanation or calculation detail",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional lineage or provenance metadata (no fabricated attributes)",
    )

    @field_validator("free_flow_speed_kmh")
    @classmethod
    def check_free_flow_speed(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"free_flow_speed_kmh must be positive and finite when provided, got {v}")
            return float(v)
        return None

    @field_validator("segment_length_km")
    @classmethod
    def check_segment_length(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"segment_length_km must be positive and finite when provided, got {v}")
            return float(v)
        return None

    @field_validator("free_flow_travel_time_minutes")
    @classmethod
    def check_free_flow_travel_time(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"free_flow_travel_time_minutes must be positive and finite when provided, got {v}")
            return float(v)
        return None


class CongestionIntelligenceEngine:
    """Unified engine for orchestrating segment congestion classification and delay calculation."""

    def __init__(self, config: Optional[CongestionThresholdConfig] = None) -> None:
        """Initialize engine with optional custom threshold configuration."""
        self.config = config or CongestionThresholdConfig()

    def evaluate_segment(
        self,
        segment_id: str,
        segment_length_km: Optional[float],
        free_flow_speed_kmh: Optional[float],
        predicted_speed_kmh: Optional[float],
        forecast_timestamp: Optional[datetime] = None,
        forecast_horizon_minutes: Optional[int] = None,
        model_source: str = "UNSPECIFIED",
        source_geography: str = "UNSPECIFIED",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SegmentCongestionResult:
        """Evaluate a single road segment's congestion level, travel time, and delay.
        
        Args:
            segment_id: Segment identifier (e.g., road edge ID or sensor ID).
            segment_length_km: Length in kilometres (must be > 0).
            free_flow_speed_kmh: Free-flow speed in km/h (must be > 0).
            predicted_speed_kmh: Predicted or observed speed in km/h.
            forecast_timestamp: Genuine timestamp of prediction if available.
            forecast_horizon_minutes: Horizon in minutes (e.g. 5, 15, 30, 60).
            model_source: Model or telemetry provider ID.
            source_geography: Geographic jurisdiction/corridor.
            metadata: Optional dictionary of provenance tags.
            
        Returns:
            Validated SegmentCongestionResult instance.
        """
        extra_meta = dict(metadata or {})

        # Step 1: Classify congestion
        level, cong_status, ratio, cong_msg = CongestionClassifier.classify(
            predicted_speed_kmh=predicted_speed_kmh,
            free_flow_speed_kmh=free_flow_speed_kmh,
            config=self.config,
        )

        # Step 2: Calculate travel time and delay
        tt_pred, tt_ff, delay, delay_status, delay_msg = DelayCalculator.calculate(
            segment_length_km=segment_length_km,
            predicted_speed_kmh=predicted_speed_kmh,
            free_flow_speed_kmh=free_flow_speed_kmh,
        )

        # Synthesize primary status and validity
        # Prioritize failure statuses
        is_valid = True
        primary_status = CongestionStatus.NORMAL
        status_msg = cong_msg

        if cong_status in (
            CongestionStatus.INVALID_FREE_FLOW_SPEED,
            CongestionStatus.INVALID_SPEED,
            CongestionStatus.MISSING_SPEED,
        ):
            is_valid = False
            primary_status = cong_status
            status_msg = cong_msg
        elif delay_status in (
            CongestionStatus.INVALID_SEGMENT_LENGTH,
            CongestionStatus.INVALID_FREE_FLOW_SPEED,
            CongestionStatus.INVALID_SPEED,
            CongestionStatus.MISSING_SPEED,
        ):
            is_valid = False
            primary_status = delay_status
            status_msg = delay_msg
        elif cong_status == CongestionStatus.STOPPED or delay_status == CongestionStatus.STOPPED:
            primary_status = CongestionStatus.STOPPED
            status_msg = f"{cong_msg}. {delay_msg}"
        elif cong_status == CongestionStatus.FASTER_THAN_FREE_FLOW:
            primary_status = CongestionStatus.FASTER_THAN_FREE_FLOW
            status_msg = delay_msg

        # Extract verified dimensions without substituting hardcoded fallbacks
        safe_ff_speed = (
            float(free_flow_speed_kmh)
            if (
                free_flow_speed_kmh is not None
                and isinstance(free_flow_speed_kmh, (int, float))
                and math.isfinite(free_flow_speed_kmh)
                and free_flow_speed_kmh > 0
            )
            else None
        )
        safe_length = (
            float(segment_length_km)
            if (
                segment_length_km is not None
                and isinstance(segment_length_km, (int, float))
                and math.isfinite(segment_length_km)
                and segment_length_km > 0
            )
            else None
        )
        safe_pred_speed = (
            float(predicted_speed_kmh)
            if (
                predicted_speed_kmh is not None
                and isinstance(predicted_speed_kmh, (int, float))
                and math.isfinite(predicted_speed_kmh)
                and predicted_speed_kmh >= 0
            )
            else None
        )

        return SegmentCongestionResult(
            segment_id=str(segment_id),
            forecast_timestamp=forecast_timestamp,
            forecast_horizon_minutes=forecast_horizon_minutes,
            predicted_speed_kmh=safe_pred_speed,
            free_flow_speed_kmh=safe_ff_speed,
            speed_ratio=ratio,
            congestion_level=level,
            congestion_status=primary_status,
            segment_length_km=safe_length,
            estimated_travel_time_minutes=tt_pred,
            free_flow_travel_time_minutes=tt_ff,
            estimated_delay_minutes=delay,
            model_source=str(model_source),
            source_geography=str(source_geography),
            is_valid=is_valid,
            status_message=status_msg,
            metadata=extra_meta,
        )

    def evaluate_batch(
        self,
        records: List[Dict[str, Any]],
        default_model_source: str = "UNSPECIFIED",
        default_geography: str = "UNSPECIFIED",
    ) -> List[SegmentCongestionResult]:
        """Evaluate a batch of segment traffic records.
        
        Args:
            records: List of dictionaries with keys:
              - segment_id (required)
              - segment_length_km (optional)
              - free_flow_speed_kmh (optional)
              - predicted_speed_kmh (optional)
              - forecast_timestamp (optional)
              - forecast_horizon_minutes (optional)
              - model_source (optional)
              - source_geography (optional)
              - metadata (optional)
              
        Returns:
            List of SegmentCongestionResult instances.
        """
        results = []
        for r in records:
            res = self.evaluate_segment(
                segment_id=r["segment_id"],
                segment_length_km=r.get("segment_length_km"),
                free_flow_speed_kmh=r.get("free_flow_speed_kmh"),
                predicted_speed_kmh=r.get("predicted_speed_kmh"),
                forecast_timestamp=r.get("forecast_timestamp"),
                forecast_horizon_minutes=r.get("forecast_horizon_minutes"),
                model_source=r.get("model_source", default_model_source),
                source_geography=r.get("source_geography", default_geography),
                metadata=r.get("metadata"),
            )
            results.append(res)
        return results

    def evaluate_road_network_edge(
        self,
        edge_data: Dict[str, Any],
        predicted_speed_kmh: Optional[float],
        forecast_timestamp: Optional[datetime] = None,
        forecast_horizon_minutes: Optional[int] = None,
        model_source: str = "ROAD_NETWORK_ESTIMATE",
        source_geography: str = "Tamil Nadu (Coimbatore-Avinashi Road)",
    ) -> SegmentCongestionResult:
        """Convenience method adapting a P1 RoadNetworkGraph edge dictionary directly.
        
        Extracts:
        - edge_id from edge_data['edge_id']
        - length_km from edge_data['length_m'] / 1000.0 or edge_data['length_km']
        - free_flow_speed_kmh from edge_data['free_flow_speed_kmh']
        """
        seg_id = str(edge_data.get("edge_id", "unknown_edge"))
        length_km = (
            float(edge_data["length_km"])
            if "length_km" in edge_data and edge_data["length_km"] is not None
            else (float(edge_data["length_m"]) / 1000.0 if "length_m" in edge_data and edge_data["length_m"] is not None else None)
        )
        ff_speed = (
            float(edge_data["free_flow_speed_kmh"])
            if "free_flow_speed_kmh" in edge_data and edge_data["free_flow_speed_kmh"] is not None
            else None
        )

        meta = {
            "highway_type": edge_data.get("highway_type", "unclassified"),
            "road_name": edge_data.get("name"),
            "road_reference": edge_data.get("road_reference"),
            "speed_source": edge_data.get("speed_source"),
        }

        return self.evaluate_segment(
            segment_id=seg_id,
            segment_length_km=length_km,
            free_flow_speed_kmh=ff_speed,
            predicted_speed_kmh=predicted_speed_kmh,
            forecast_timestamp=forecast_timestamp,
            forecast_horizon_minutes=forecast_horizon_minutes,
            model_source=model_source,
            source_geography=source_geography,
            metadata=meta,
        )
