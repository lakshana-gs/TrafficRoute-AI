"""Predictive Route Evaluation Engine for TrafficRoute AI (Milestone P7.1).

Provides forward spatio-temporal route simulation and pluggable speed-provider contracts:
- Forward time-propagation loop integrating dynamic segment velocities
- BaseSpeedProvider abstract interface and concrete implementations:
  - StaticFreeFlowSpeedProvider (baseline fallback)
  - HeuristicDiurnalSpeedProvider (assumption-based IST peak/off-peak profile)
  - MockSpeedProvider (deterministic synthetic testing fixture)
- Data contracts for speed query results, evaluated segments, and evaluated routes
- Numerical and physical safeguards: crawl floor for stopped traffic, non-negative dimensions,
  strict timestamp monotonicity, and explicit provenance tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RoutePlanResult, RouteSegment
from src.traffic.congestion import (
    CongestionClassifier,
    CongestionLevel,
    CongestionStatus,
    DelayCalculator,
    SegmentCongestionResult,
)

# Indian Standard Time (IST) offset: UTC + 05:30
IST_OFFSET = timezone(timedelta(hours=5, minutes=30), name="IST")


class SpeedProvenanceType(str, Enum):
    """Originating methodology for segment speed estimation."""

    ASSUMPTION_BASED_DIURNAL = "assumption_based_diurnal"  # Heuristic diurnal curve based on peak/off-peak assumptions
    MODEL_PREDICTED = "model_predicted"                    # Spatio-temporal ML/DCRNN multi-horizon forecast (METR-LA only)
    HISTORICAL_PROFILE = "historical_profile"              # Historical time-of-day tabular sensor mean (P3)
    FREE_FLOW_FALLBACK = "free_flow_fallback"              # Static OSM speed limit / road hierarchy default
    SYNTHETIC_TEST = "synthetic_test"                      # Deterministic test fixture only (explicitly marked)


class ForecastHorizonStatus(str, Enum):
    """Status of forecast horizon relative to model capabilities."""

    WITHIN_HORIZON = "within_horizon"                      # Arrival time is within valid forecast horizon (e.g. <= 60 min)
    BEYOND_MODEL_HORIZON = "beyond_model_horizon"          # Arrival time exceeds model horizon (> 60 min); fallback invoked
    NOT_APPLICABLE = "not_applicable"                      # Speed source does not use a horizon (diurnal, free-flow, synthetic)


class SpeedQueryResult(BaseModel):
    """Result of querying a speed provider for an edge at a given target arrival time."""

    speed_kmh: float = Field(..., ge=0.0, description="Estimated vehicle speed in km/h (0.0 represents stopped traffic)")
    provenance_type: SpeedProvenanceType = Field(..., description="Classification of speed derivation")
    model_source: str = Field(..., description="Identifier of model, provider, or fallback utilized")
    forecast_horizon_minutes: Optional[int] = Field(default=None, description="Forecast horizon step in minutes if model-derived")
    forecast_horizon_status: ForecastHorizonStatus = Field(
        default=ForecastHorizonStatus.NOT_APPLICABLE,
        description="Horizon availability classification",
    )
    confidence_tier: str = Field(..., description="Documented confidence tier: HEURISTIC_ESTIMATE, BENCHMARK_SENSOR, or BASELINE_UNMONITORED")
    is_fallback: bool = Field(default=False, description="True if primary model was unavailable or horizon was exceeded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic notes, limitations, or lineage")

    @field_validator("speed_kmh", mode="before")
    @classmethod
    def validate_speed_kmh(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
            raise ValueError(f"speed_kmh must be a finite, non-negative number, got {v}")
        return float(v)


class BaseSpeedProvider(ABC):
    """Abstract interface defining speed evaluation for network edges."""

    @abstractmethod
    def get_edge_speed(
        self,
        u: int,
        v: int,
        key: int,
        edge_data: Dict[str, Any],
        target_arrival_time: datetime,
        departure_time: datetime,
    ) -> SpeedQueryResult:
        """
        Evaluate speed for edge (u, v, key) at vehicle arrival timestamp.

        Args:
            u: Source node ID.
            v: Target node ID.
            key: Edge key in multigraph.
            edge_data: NetworkX edge attribute dictionary.
            target_arrival_time: Simulated arrival time of vehicle at edge start (timezone-aware).
            departure_time: Initial trip departure timestamp (timezone-aware).

        Returns:
            SpeedQueryResult containing non-negative speed, horizon status, and explicit provenance.
        """
        pass


class StaticFreeFlowSpeedProvider(BaseSpeedProvider):
    """Static provider returning edge free-flow speed without dynamic impedance."""

    def __init__(self, default_speed_kmh: float = 30.0):
        if not isinstance(default_speed_kmh, (int, float)) or isinstance(default_speed_kmh, bool) or not math.isfinite(default_speed_kmh) or default_speed_kmh <= 0.0:
            raise ValueError(f"default_speed_kmh must be a finite positive float, got {default_speed_kmh}")
        self.default_speed_kmh = float(default_speed_kmh)

    def get_edge_speed(
        self,
        u: int,
        v: int,
        key: int,
        edge_data: Dict[str, Any],
        target_arrival_time: datetime,
        departure_time: datetime,
    ) -> SpeedQueryResult:
        raw_speed = edge_data.get("free_flow_speed_kmh")
        if (
            isinstance(raw_speed, (int, float))
            and not isinstance(raw_speed, bool)
            and math.isfinite(raw_speed)
            and raw_speed > 0.0
        ):
            speed = float(raw_speed)
        else:
            speed = self.default_speed_kmh

        return SpeedQueryResult(
            speed_kmh=speed,
            provenance_type=SpeedProvenanceType.FREE_FLOW_FALLBACK,
            model_source="STATIC_FREE_FLOW_PROVIDER",
            forecast_horizon_minutes=None,
            forecast_horizon_status=ForecastHorizonStatus.NOT_APPLICABLE,
            confidence_tier="BASELINE_UNMONITORED",
            is_fallback=True,
            metadata={"highway_type": edge_data.get("highway_type", "unclassified")},
        )


class HeuristicDiurnalSpeedProvider(BaseSpeedProvider):
    """
    Assumption-based diurnal speed provider for the Coimbatore Pilot Corridor.

    LIMITATION AND PROVENANCE DISCLOSURE:
    The repository contains no measured Tamil Nadu speed telemetry, no IRC/MoRTH raw survey
    datasets, and no fitted calibration pipeline. This provider implements an uncalibrated
    engineering heuristic modeled after the synthetic diurnal generator in src/traffic/ingestion.py.
    It must NOT be described as 'calibrated' or 'empirically measured'.
    """

    def __init__(self, minimum_speed_kmh: float = 10.0):
        if not isinstance(minimum_speed_kmh, (int, float)) or isinstance(minimum_speed_kmh, bool) or not math.isfinite(minimum_speed_kmh) or minimum_speed_kmh <= 0.0:
            raise ValueError(f"minimum_speed_kmh must be a finite positive float, got {minimum_speed_kmh}")
        self.minimum_speed_kmh = float(minimum_speed_kmh)

    @staticmethod
    def _compute_diurnal_multiplier(hour_ist: float, is_weekend: bool, highway_type: str) -> float:
        """Calculate diurnal speed multiplier in [0.0, 1.0] for IST hour of day."""
        is_highway = highway_type in ("motorway", "trunk", "motorway_link", "trunk_link")

        # 1. Morning Peak (08:00 - 10:30 IST)
        if 8.0 <= hour_ist <= 10.5:
            intensity = math.sin((hour_ist - 8.0) / 2.5 * math.pi)
            if is_highway:
                multiplier = 0.85 - (0.10 * intensity)  # 0.75 - 0.85
            else:
                multiplier = 0.70 - (0.15 * intensity)  # 0.55 - 0.70

        # 2. Evening Peak (17:30 - 20:30 IST)
        elif 17.5 <= hour_ist <= 20.5:
            intensity = math.sin((hour_ist - 17.5) / 3.0 * math.pi)
            if is_highway:
                multiplier = 0.80 - (0.10 * intensity)  # 0.70 - 0.80
            else:
                multiplier = 0.65 - (0.15 * intensity)  # 0.50 - 0.65

        # 3. Midday Inter-Peak (11:00 - 16:30 IST)
        elif 11.0 <= hour_ist <= 16.5:
            intensity = math.sin((hour_ist - 11.0) / 5.5 * math.pi)
            multiplier = 0.85 - (0.05 * intensity)      # 0.80 - 0.85

        # 4. Night Off-Peak (23:00 - 05:00 IST)
        elif hour_ist >= 23.0 or hour_ist <= 5.0:
            multiplier = 0.98                           # 0.95 - 1.00

        # 5. Early Morning Shoulder (05:00 - 08:00 IST)
        elif 5.0 < hour_ist < 8.0:
            frac = (hour_ist - 5.0) / 3.0
            base_target = 0.85 if is_highway else 0.70
            multiplier = 0.98 - frac * (0.98 - base_target)

        # 6. Post-Morning Shoulder (10:30 - 11:00 IST)
        elif 10.5 < hour_ist < 11.0:
            frac = (hour_ist - 10.5) / 0.5
            start_val = 0.85 if is_highway else 0.70
            multiplier = start_val + frac * (0.85 - start_val)

        # 7. Pre-Evening Shoulder (16:30 - 17:30 IST)
        elif 16.5 < hour_ist < 17.5:
            frac = (hour_ist - 16.5) / 1.0
            end_val = 0.80 if is_highway else 0.65
            multiplier = 0.85 - frac * (0.85 - end_val)

        # 8. Post-Evening Shoulder (20:30 - 23:00 IST)
        else:
            frac = (hour_ist - 20.5) / 2.5
            start_val = 0.80 if is_highway else 0.65
            multiplier = start_val + frac * (0.98 - start_val)

        # Weekend adjustment: lighter congestion
        if is_weekend:
            multiplier = min(1.0, multiplier + (1.0 - multiplier) * 0.35)

        return max(0.20, min(1.0, multiplier))

    def get_edge_speed(
        self,
        u: int,
        v: int,
        key: int,
        edge_data: Dict[str, Any],
        target_arrival_time: datetime,
        departure_time: datetime,
    ) -> SpeedQueryResult:
        # Convert arrival time to Indian Standard Time (IST, UTC+05:30)
        if target_arrival_time.tzinfo is None:
            raise ValueError(f"target_arrival_time must be timezone-aware, got naive {target_arrival_time}")
        
        ist_time = target_arrival_time.astimezone(IST_OFFSET)
        hour_ist = ist_time.hour + (ist_time.minute / 60.0) + (ist_time.second / 3600.0)
        is_weekend = ist_time.weekday() in (5, 6)
        highway_type = str(edge_data.get("highway_type") or edge_data.get("highway") or "unclassified").lower()

        raw_ff = edge_data.get("free_flow_speed_kmh")
        if (
            isinstance(raw_ff, (int, float))
            and not isinstance(raw_ff, bool)
            and math.isfinite(raw_ff)
            and raw_ff > 0.0
        ):
            free_flow_kmh = float(raw_ff)
        else:
            free_flow_kmh = 30.0

        multiplier = self._compute_diurnal_multiplier(hour_ist, is_weekend, highway_type)
        estimated_speed = max(self.minimum_speed_kmh, free_flow_kmh * multiplier)

        return SpeedQueryResult(
            speed_kmh=round(estimated_speed, 2),
            provenance_type=SpeedProvenanceType.ASSUMPTION_BASED_DIURNAL,
            model_source="HEURISTIC_DIURNAL",
            forecast_horizon_minutes=None,
            forecast_horizon_status=ForecastHorizonStatus.NOT_APPLICABLE,
            confidence_tier="HEURISTIC_ESTIMATE",
            is_fallback=False,
            metadata={
                "hour_ist": round(hour_ist, 2),
                "is_weekend": is_weekend,
                "highway_type": highway_type,
                "diurnal_multiplier": round(multiplier, 4),
                "free_flow_speed_kmh": round(free_flow_kmh, 1),
            },
        )


class MockSpeedProvider(BaseSpeedProvider):
    """Deterministic mock speed provider for unit tests and synthetic simulation."""

    def __init__(
        self,
        default_speed_kmh: float = 60.0,
        edge_speeds: Optional[Dict[Tuple[int, int, int], float]] = None,
        speed_fn: Optional[Callable[[int, int, int, datetime], float]] = None,
        forecast_horizon_minutes: Optional[int] = None,
        forecast_horizon_status: ForecastHorizonStatus = ForecastHorizonStatus.NOT_APPLICABLE,
        is_fallback: bool = False,
        model_source: str = "MOCK_SPEED_PROVIDER",
    ):
        self.default_speed_kmh = float(default_speed_kmh)
        self.edge_speeds = edge_speeds or {}
        self.speed_fn = speed_fn
        self.forecast_horizon_minutes = forecast_horizon_minutes
        self.forecast_horizon_status = forecast_horizon_status
        self.is_fallback = is_fallback
        self.model_source = model_source

    def get_edge_speed(
        self,
        u: int,
        v: int,
        key: int,
        edge_data: Dict[str, Any],
        target_arrival_time: datetime,
        departure_time: datetime,
    ) -> SpeedQueryResult:
        if self.speed_fn is not None:
            speed = float(self.speed_fn(u, v, key, target_arrival_time))
        elif (u, v, key) in self.edge_speeds:
            speed = float(self.edge_speeds[(u, v, key)])
        else:
            speed = self.default_speed_kmh

        return SpeedQueryResult(
            speed_kmh=max(0.0, speed),
            provenance_type=SpeedProvenanceType.SYNTHETIC_TEST,
            model_source=self.model_source,
            forecast_horizon_minutes=self.forecast_horizon_minutes,
            forecast_horizon_status=self.forecast_horizon_status,
            confidence_tier="SYNTHETIC_TEST_ONLY",
            is_fallback=self.is_fallback,
            metadata={"mock_query": True},
        )


class RouteBottleneck(BaseModel):
    """Identified and attributed congestion hotspot along a candidate route."""

    rank: int = Field(..., ge=1, description="Rank ordered descending by delay contribution (1 = highest delay)")
    segment_id: str = Field(..., description="Graph edge identifier or clustered corridor identifier")
    u: int = Field(..., description="Origin node ID of edge")
    v: int = Field(..., description="Target node ID of edge")
    key: int = Field(default=0, description="Multigraph edge key")
    road_name: Optional[str] = Field(default=None, description="Human-readable road name")
    road_reference: Optional[str] = Field(default=None, description="Official road reference (e.g. NH-544)")
    highway_type: str = Field(..., description="Functional road class (trunk, primary, secondary, etc.)")
    delay_seconds: float = Field(..., gt=0.0, description="Absolute delay contribution in seconds")
    delay_minutes: float = Field(..., gt=0.0, description="Delay contribution in minutes")
    observed_speed_kmh: float = Field(..., gt=0.0, description="Estimated traversal speed in km/h")
    free_flow_speed_kmh: float = Field(..., gt=0.0, description="Reference baseline speed in km/h")
    speed_ratio: float = Field(..., ge=0.0, description="Ratio of speed to free-flow (v_pred / v_free)")
    congestion_level: CongestionLevel = Field(..., description="P5 congestion severity (HIGH, SEVERE)")
    arrival_time: datetime = Field(..., description="Timezone-aware vehicle arrival timestamp at bottleneck")
    display_title: str = Field(..., description="Concise bottleneck label for UI card")


class EvaluatedRouteSegment(BaseModel):
    """Traversed edge segment enriched with forward arrival time and dynamic congestion metrics."""

    segment: RouteSegment = Field(..., description="Underlying P6 structural route segment")
    arrival_time: datetime = Field(..., description="Timezone-aware arrival timestamp at segment start")
    traversal_time_seconds: float = Field(..., gt=0.0, description="Dynamic traversal duration in seconds")
    free_flow_time_seconds: float = Field(..., gt=0.0, description="Static free-flow reference time in seconds")
    signed_delay_seconds: float = Field(..., description="Signed delay in seconds: traversal_time - free_flow_time (positive=delay, negative=speedup)")
    speed_result: SpeedQueryResult = Field(..., description="Speed evaluation details, horizon status, and provenance")
    congestion_result: SegmentCongestionResult = Field(..., description="P5 canonical segment congestion result")
    is_bottleneck: bool = Field(default=False, description="True if delay contribution qualifies as an actionable bottleneck")


class EvaluatedRoute(BaseModel):
    """Comprehensive predictive evaluation of a single route candidate."""

    route_id: str = Field(..., description="Unique route identifier from P6")
    name: str = Field(default="Evaluated Route", description="Descriptive corridor name (e.g. 'Via NH-544')")
    departure_time: datetime = Field(..., description="Planned departure timestamp (timezone-aware)")
    estimated_arrival_time: datetime = Field(..., description="Simulated trip completion timestamp (timezone-aware)")
    total_distance_m: float = Field(..., ge=0.0, description="Total route distance in meters")
    total_distance_km: float = Field(..., ge=0.0, description="Total distance in kilometres")
    free_flow_time_seconds: float = Field(..., ge=0.0, description="Theoretical zero-traffic journey duration in seconds")
    predicted_travel_time_seconds: float = Field(..., ge=0.0, description="Forward time-integrated journey duration in seconds")
    predicted_travel_time_minutes: float = Field(..., ge=0.0, description="Journey duration in minutes")
    total_delay_seconds: float = Field(..., description="Signed cumulative delay in seconds: predicted_time - free_flow_time")
    total_delay_minutes: float = Field(..., description="Signed cumulative delay in minutes")
    total_positive_delay_seconds: float = Field(default=0.0, ge=0.0, description="Gross positive delay across route segments in seconds")
    total_positive_delay_minutes: float = Field(default=0.0, ge=0.0, description="Gross positive delay in minutes")
    average_speed_kmh: float = Field(..., ge=0.0, description="Average journey velocity under predicted traffic")
    route_congestion_level: CongestionLevel = Field(..., description="Overall Level of Service (LoS)")
    bottlenecks: List[RouteBottleneck] = Field(default_factory=list, description="Top delay-inducing bottlenecks (P7.2)")
    evaluated_segments: List[EvaluatedRouteSegment] = Field(..., description="Segment-by-segment simulation details")
    data_provenance: Dict[str, Any] = Field(default_factory=dict, description="Breakdown of models, assumptions, and fallback usage")


class PredictiveEvaluatorConfig(BaseModel):
    """Configurable evaluation and recommendation thresholds."""

    min_time_savings_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Minimum seconds an alternative must save over the primary route to trigger recommendation",
    )
    tie_threshold_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Threshold within which two route travel times are considered tied",
    )
    max_bottlenecks_per_route: int = Field(
        default=5,
        ge=1,
        description="Maximum number of top bottlenecks to surface per route",
    )
    zero_speed_crawl_kmh: float = Field(
        default=3.0,
        gt=0.0,
        description="Crawl speed floor for stopped traffic to avoid zero division",
    )
    min_bottleneck_delay_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Minimum positive delay in seconds to qualify as bottleneck",
    )


class RouteComparisonResult(BaseModel):
    """Side-by-side comparative evaluation of primary and alternative routes."""

    departure_time: datetime = Field(..., description="Evaluation departure timestamp (timezone-aware)")
    recommended_route_id: str = Field(..., description="Route ID recommended for lowest predicted travel time")
    recommendation_reason: str = Field(..., description="Human-readable decision rationale based strictly on travel time")
    time_savings_seconds: float = Field(..., ge=0.0, description="Time saved in seconds by choosing recommendation over primary route")
    time_savings_minutes: float = Field(..., ge=0.0, description="Time saved in minutes")
    primary_route: EvaluatedRoute = Field(..., description="Evaluated primary corridor")
    alternative_routes: List[EvaluatedRoute] = Field(default_factory=list, description="Evaluated alternative corridors")
    ranked_routes: List[EvaluatedRoute] = Field(..., description="All routes sorted by predicted travel time ascending")
    evaluation_disclaimer: str = Field(
        default=(
            "DISCLAIMER: This recommendation optimizes estimated travel time only based on simulated traffic impedance. "
            "It does NOT evaluate road safety, accident risk, pavement condition, lighting, or weather hazards, and must not "
            "be interpreted as a safety recommendation."
        ),
        description="Explicit non-safety advisory",
    )
    data_provenance: Dict[str, Any] = Field(default_factory=dict, description="Consolidated provenance across evaluated routes")

    def summary(self) -> Dict[str, Any]:
        """JSON-serializable summary representation for reporting and UI cards."""
        return {
            "departure_time": self.departure_time.isoformat(),
            "recommended_route_id": self.recommended_route_id,
            "recommendation_reason": self.recommendation_reason,
            "time_savings_seconds": self.time_savings_seconds,
            "time_savings_minutes": self.time_savings_minutes,
            "primary_route": {
                "route_id": self.primary_route.route_id,
                "name": self.primary_route.name,
                "distance_km": self.primary_route.total_distance_km,
                "predicted_time_min": self.primary_route.predicted_travel_time_minutes,
                "delay_min": self.primary_route.total_delay_minutes,
                "congestion_level": self.primary_route.route_congestion_level.value,
                "bottleneck_count": len(self.primary_route.bottlenecks),
            },
            "ranked_routes": [
                {
                    "rank": idx + 1,
                    "route_id": r.route_id,
                    "name": r.name,
                    "distance_km": r.total_distance_km,
                    "predicted_time_min": r.predicted_travel_time_minutes,
                    "delay_min": r.total_delay_minutes,
                    "congestion_level": r.route_congestion_level.value,
                    "bottleneck_count": len(r.bottlenecks),
                    "is_recommended": r.route_id == self.recommended_route_id,
                }
                for idx, r in enumerate(self.ranked_routes)
            ],
            "evaluation_disclaimer": self.evaluation_disclaimer,
        }



def generate_bottleneck_display_title(
    road_name: Optional[str],
    road_reference: Optional[str],
    highway_type: str,
    u: int,
    delay_minutes: float,
) -> str:
    """Format human-readable bottleneck card title adhering to specification."""
    name_clean = road_name.strip() if road_name and road_name.strip() else None
    ref_clean = road_reference.strip() if road_reference and road_reference.strip() else None
    hw_clean = highway_type.replace("_", " ").strip().title() if highway_type else "Road"

    if name_clean and ref_clean:
        return f"{name_clean} ({ref_clean}): +{delay_minutes:.1f}m delay"
    elif name_clean:
        return f"{name_clean}: +{delay_minutes:.1f}m delay"
    elif ref_clean:
        return f"{ref_clean}: +{delay_minutes:.1f}m delay"
    else:
        return f"{hw_clean} corridor near Node {u}: +{delay_minutes:.1f}m delay"


def extract_route_bottlenecks(
    evaluated_segments: List[EvaluatedRouteSegment],
    min_delay_seconds: float = 30.0,
    max_bottlenecks: Optional[int] = None,
) -> List[RouteBottleneck]:
    """
    Extract, cluster consecutive corridor links, rank, and attribute route bottlenecks.

    Invariant Rules:
    1. Qualification Criteria:
       - Traffic is stopped (CongestionStatus.STOPPED or predicted speed <= 0.0), OR
       - CongestionLevel is HIGH or SEVERE AND signed_delay_seconds >= min_delay_seconds.
       - Delay contribution must be strictly positive (delay_seconds > 0.0).
    2. Continuous Corridor Deduplication:
       - Consecutive qualifying segments sharing the same road name (or same road reference
         if road name is missing) are aggregated into a single continuous corridor bottleneck.
       - Unnamed, unreferenced segments remain separate to provide exact node localization.
    3. Disjoint Attribution and Mathematical Bound:
       - Subsets of segments in bottlenecks are mutually disjoint (each segment attributed at most once).
       - Sum of bottleneck delays is bounded by gross positive route delay:
         sum(b.delay_seconds for b in bottlenecks) <= gross_positive_delay_seconds.
    """
    if not evaluated_segments:
        return []

    # 1. Identify qualifying segment indices
    qualifying_indices: List[int] = []
    for idx, seg in enumerate(evaluated_segments):
        is_stopped = (
            seg.congestion_result.congestion_status == CongestionStatus.STOPPED
            or seg.speed_result.speed_kmh <= 0.0
        )
        is_high_severe = seg.congestion_result.congestion_level in (
            CongestionLevel.HIGH,
            CongestionLevel.SEVERE,
        )
        has_min_delay = seg.signed_delay_seconds >= min_delay_seconds

        if (is_stopped or (is_high_severe and has_min_delay)) and seg.signed_delay_seconds > 0.0:
            qualifying_indices.append(idx)

    if not qualifying_indices:
        return []

    # 2. Cluster consecutive qualifying segments sharing road name or reference
    def _get_group_key(s: EvaluatedRouteSegment) -> Tuple[str, Optional[str]]:
        name = s.segment.name.strip().lower() if s.segment.name and s.segment.name.strip() else None
        if name:
            return ("name", name)
        ref = s.segment.road_reference.strip().lower() if s.segment.road_reference and s.segment.road_reference.strip() else None
        if ref:
            return ("ref", ref)
        return ("unnamed", f"{s.segment.edge_id}_{s.segment.u}_{s.segment.v}")

    clusters: List[List[EvaluatedRouteSegment]] = []
    current_cluster: List[EvaluatedRouteSegment] = []
    prev_idx: Optional[int] = None

    for idx in qualifying_indices:
        seg = evaluated_segments[idx]
        if not current_cluster:
            current_cluster.append(seg)
            prev_idx = idx
        else:
            is_consecutive = (prev_idx is not None and prev_idx + 1 == idx)
            prev_key = _get_group_key(current_cluster[-1])
            curr_key = _get_group_key(seg)

            if is_consecutive and prev_key[0] in ("name", "ref") and prev_key == curr_key:
                current_cluster.append(seg)
                prev_idx = idx
            else:
                clusters.append(current_cluster)
                current_cluster = [seg]
                prev_idx = idx

    if current_cluster:
        clusters.append(current_cluster)

    # 3. Aggregate cluster properties
    raw_bottlenecks: List[Dict[str, Any]] = []
    for cluster in clusters:
        del_sec = round(sum(s.signed_delay_seconds for s in cluster), 3)
        if del_sec <= 0.0:
            continue
        del_min = round(del_sec / 60.0, 3)

        tot_dist_m = sum(s.segment.length_m for s in cluster)
        tot_trav_s = sum(s.traversal_time_seconds for s in cluster)
        tot_ff_s = sum(s.free_flow_time_seconds for s in cluster)

        obs_speed = (
            round((tot_dist_m / 1000.0) / (tot_trav_s / 3600.0), 2)
            if tot_trav_s > 0
            else cluster[0].congestion_result.predicted_speed_kmh
        )
        ff_speed = (
            round((tot_dist_m / 1000.0) / (tot_ff_s / 3600.0), 2)
            if tot_ff_s > 0
            else 30.0
        )
        ratio = round(obs_speed / ff_speed, 4) if ff_speed > 0 else 0.0

        is_severe = any(
            s.congestion_result.congestion_level == CongestionLevel.SEVERE
            or s.congestion_result.congestion_status == CongestionStatus.STOPPED
            for s in cluster
        )
        cg_level = CongestionLevel.SEVERE if is_severe else CongestionLevel.HIGH

        u_node = cluster[0].segment.u
        v_node = cluster[-1].segment.v
        key_val = cluster[0].segment.key

        r_name = cluster[0].segment.name
        r_ref = cluster[0].segment.road_reference
        hw_type = cluster[0].segment.highway_type

        seg_id = (
            cluster[0].segment.edge_id
            if len(cluster) == 1
            else f"{cluster[0].segment.edge_id}...{cluster[-1].segment.edge_id}"
        )

        title = generate_bottleneck_display_title(
            road_name=r_name,
            road_reference=r_ref,
            highway_type=hw_type,
            u=u_node,
            delay_minutes=del_min,
        )

        raw_bottlenecks.append({
            "cluster": cluster,
            "segment_id": seg_id,
            "u": u_node,
            "v": v_node,
            "key": key_val,
            "road_name": r_name,
            "road_reference": r_ref,
            "highway_type": hw_type,
            "delay_seconds": del_sec,
            "delay_minutes": del_min,
            "observed_speed_kmh": obs_speed,
            "free_flow_speed_kmh": ff_speed,
            "speed_ratio": ratio,
            "congestion_level": cg_level,
            "arrival_time": cluster[0].arrival_time,
            "display_title": title,
        })

    # 4. Rank descending by delay_seconds; break ties by arrival_time ascending, u ascending
    raw_bottlenecks.sort(
        key=lambda b: (-b["delay_seconds"], b["arrival_time"], b["u"])
    )

    if max_bottlenecks is not None and max_bottlenecks > 0:
        raw_bottlenecks = raw_bottlenecks[:max_bottlenecks]

    result_bottlenecks: List[RouteBottleneck] = []
    for rank_idx, b in enumerate(raw_bottlenecks, start=1):
        for s in b["cluster"]:
            s.is_bottleneck = True

        bottleneck = RouteBottleneck(
            rank=rank_idx,
            segment_id=b["segment_id"],
            u=b["u"],
            v=b["v"],
            key=b["key"],
            road_name=b["road_name"],
            road_reference=b["road_reference"],
            highway_type=b["highway_type"],
            delay_seconds=b["delay_seconds"],
            delay_minutes=b["delay_minutes"],
            observed_speed_kmh=b["observed_speed_kmh"],
            free_flow_speed_kmh=b["free_flow_speed_kmh"],
            speed_ratio=b["speed_ratio"],
            congestion_level=b["congestion_level"],
            arrival_time=b["arrival_time"],
            display_title=b["display_title"],
        )
        result_bottlenecks.append(bottleneck)

    return result_bottlenecks


def forward_propagate_route(
    route: Route,
    departure_time: datetime,
    speed_provider: BaseSpeedProvider,
    road_graph: Optional[RoadNetworkGraph] = None,
    crawl_speed_kmh: float = 3.0,
    min_bottleneck_delay_seconds: float = 30.0,
    max_bottlenecks: Optional[int] = None,
) -> EvaluatedRoute:
    """
    Execute forward spatio-temporal route simulation sequentially along route segments.

    Args:
        route: Validated P6 Route object.
        departure_time: Target planned departure timestamp (must be timezone-aware).
        speed_provider: Concrete BaseSpeedProvider implementation.
        road_graph: Optional RoadNetworkGraph to access edge attributes.
        crawl_speed_kmh: Positive minimum velocity floor applied when traffic is stopped (0.0 km/h)
                         to prevent division-by-zero errors.
        min_bottleneck_delay_seconds: Minimum positive delay contribution in seconds to qualify as bottleneck.
        max_bottlenecks: Optional maximum number of top bottlenecks to retain.

    Returns:
        EvaluatedRoute containing simulated arrival timestamps, dynamic durations,
        signed delays, P5 segment congestion results, and extracted bottlenecks.
    """
    # 1. Validate inputs
    if not isinstance(departure_time, datetime) or isinstance(departure_time, bool):
        raise ValueError(f"departure_time must be a datetime instance, got {departure_time}")

    if departure_time.tzinfo is None or departure_time.tzinfo.utcoffset(departure_time) is None:
        raise ValueError(f"departure_time must be timezone-aware (tzinfo is required), got naive {departure_time}")

    if not isinstance(route, Route):
        raise ValueError(f"route must be a Route instance, got {type(route).__name__}")

    if not isinstance(speed_provider, BaseSpeedProvider):
        raise ValueError(f"speed_provider must be a BaseSpeedProvider instance, got {type(speed_provider).__name__}")

    if not isinstance(crawl_speed_kmh, (int, float)) or isinstance(crawl_speed_kmh, bool) or not math.isfinite(crawl_speed_kmh) or crawl_speed_kmh <= 0.0:
        raise ValueError(f"crawl_speed_kmh must be a finite, strictly positive number, got {crawl_speed_kmh}")

    if not isinstance(min_bottleneck_delay_seconds, (int, float)) or isinstance(min_bottleneck_delay_seconds, bool) or not math.isfinite(min_bottleneck_delay_seconds) or min_bottleneck_delay_seconds < 0.0:
        raise ValueError(f"min_bottleneck_delay_seconds must be a non-negative finite float, got {min_bottleneck_delay_seconds}")

    if max_bottlenecks is not None:
        if not isinstance(max_bottlenecks, int) or isinstance(max_bottlenecks, bool) or max_bottlenecks <= 0:
            raise ValueError(f"max_bottlenecks must be a positive integer, got {max_bottlenecks}")

    # 2. Single-node / zero-edge route handling (e.g. SAME_LOCATION)
    if len(route.edge_sequence) == 0:
        return EvaluatedRoute(
            route_id=route.route_id,
            name=route.name,
            departure_time=departure_time,
            estimated_arrival_time=departure_time,
            total_distance_m=0.0,
            total_distance_km=0.0,
            free_flow_time_seconds=0.0,
            predicted_travel_time_seconds=0.0,
            predicted_travel_time_minutes=0.0,
            total_delay_seconds=0.0,
            total_delay_minutes=0.0,
            total_positive_delay_seconds=0.0,
            total_positive_delay_minutes=0.0,
            average_speed_kmh=0.0,
            route_congestion_level=CongestionLevel.LOW,
            bottlenecks=[],
            evaluated_segments=[],
            data_provenance={
                "single_node_route": True,
                "provider": type(speed_provider).__name__,
                "segment_count": 0,
                "gross_positive_delay_seconds": 0.0,
                "gross_negative_delay_seconds": 0.0,
                "bottleneck_count": 0,
                "total_bottleneck_delay_seconds": 0.0,
            },
        )

    # 3. Sequential dynamic simulation loop
    curr_arrival_time = departure_time
    evaluated_segments: List[EvaluatedRouteSegment] = []
    total_dist_m = 0.0
    total_pred_time_s = 0.0
    total_ff_time_s = 0.0
    ci_length_sum = 0.0
    provenance_counts: Dict[str, int] = {}
    graph_obj = road_graph.graph if road_graph is not None else None

    for seg in route.edge_sequence:
        # Retrieve edge dictionary if graph available, or reconstruct from segment attributes
        edge_data: Dict[str, Any] = {}
        if graph_obj is not None and graph_obj.has_edge(seg.u, seg.v):
            if graph_obj.is_multigraph():
                edge_data = dict(graph_obj[seg.u][seg.v].get(seg.key, {}))
            else:
                edge_data = dict(graph_obj[seg.u][seg.v])

        # Fill baseline attributes from RouteSegment if missing
        if "length_m" not in edge_data:
            edge_data["length_m"] = seg.length_m
        if "free_flow_speed_kmh" not in edge_data:
            edge_data["free_flow_speed_kmh"] = seg.speed_kmh
        if "name" not in edge_data:
            edge_data["name"] = seg.name
        if "road_reference" not in edge_data:
            edge_data["road_reference"] = seg.road_reference
        if "highway_type" not in edge_data:
            edge_data["highway_type"] = seg.highway_type
        if "edge_id" not in edge_data:
            edge_data["edge_id"] = seg.edge_id

        # Query dynamic speed at current simulated arrival time
        speed_result = speed_provider.get_edge_speed(
            u=seg.u,
            v=seg.v,
            key=seg.key,
            edge_data=edge_data,
            target_arrival_time=curr_arrival_time,
            departure_time=departure_time,
        )

        # Reference free-flow speed calculation
        raw_ff = edge_data.get("free_flow_speed_kmh", seg.speed_kmh)
        if (
            isinstance(raw_ff, (int, float))
            and not isinstance(raw_ff, bool)
            and math.isfinite(raw_ff)
            and raw_ff > 0.0
        ):
            v_free = float(raw_ff)
        else:
            v_free = 30.0

        ff_speed_mps = v_free * (1000.0 / 3600.0)
        seg_ff_time_s = seg.length_m / ff_speed_mps

        # Evaluate effective speed and stopped-traffic condition
        raw_speed = speed_result.speed_kmh
        if raw_speed <= 0.0:
            effective_speed_kmh = float(crawl_speed_kmh)
            is_stopped = True
        else:
            effective_speed_kmh = raw_speed
            is_stopped = False

        pred_speed_mps = effective_speed_kmh * (1000.0 / 3600.0)
        seg_pred_time_s = seg.length_m / pred_speed_mps
        seg_delay_s = seg_pred_time_s - seg_ff_time_s

        # Congestion classification via P5 primitives
        if is_stopped:
            cg_level = CongestionLevel.SEVERE
            cg_status = CongestionStatus.STOPPED
            speed_ratio = 0.0
            status_msg = f"Traffic is stopped (0.0 km/h); crawl floor of {crawl_speed_kmh:.1f} km/h applied for finite simulation"
        else:
            cg_level, cg_status, speed_ratio, status_msg = CongestionClassifier.classify(
                predicted_speed_kmh=effective_speed_kmh,
                free_flow_speed_kmh=v_free,
            )

        congestion_res = SegmentCongestionResult(
            segment_id=seg.edge_id,
            forecast_timestamp=curr_arrival_time,
            forecast_horizon_minutes=speed_result.forecast_horizon_minutes,
            predicted_speed_kmh=round(raw_speed, 2),
            free_flow_speed_kmh=round(v_free, 2),
            speed_ratio=round(speed_ratio, 4) if speed_ratio is not None else None,
            congestion_level=cg_level,
            congestion_status=cg_status,
            segment_length_km=round(seg.length_m / 1000.0, 4),
            estimated_travel_time_minutes=round(seg_pred_time_s / 60.0, 4),
            free_flow_travel_time_minutes=round(seg_ff_time_s / 60.0, 4),
            estimated_delay_minutes=round(seg_delay_s / 60.0, 4),
            model_source=speed_result.model_source,
            source_geography="Tamil Nadu (Coimbatore Pilot)" if "Diurnal" in type(speed_provider).__name__ else "UNSPECIFIED",
            is_valid=True,
            status_message=status_msg,
            metadata=dict(speed_result.metadata),
        )

        eval_seg = EvaluatedRouteSegment(
            segment=seg,
            arrival_time=curr_arrival_time,
            traversal_time_seconds=round(seg_pred_time_s, 3),
            free_flow_time_seconds=round(seg_ff_time_s, 3),
            signed_delay_seconds=round(seg_delay_s, 3),
            speed_result=speed_result,
            congestion_result=congestion_res,
            is_bottleneck=False,
        )
        evaluated_segments.append(eval_seg)

        # Accumulate metrics
        total_dist_m += seg.length_m
        total_pred_time_s += seg_pred_time_s
        total_ff_time_s += seg_ff_time_s

        # Length-weighted Congestion Index contribution: CI = 1.0 - min(1.0, v_pred / v_free)
        seg_ci = 1.0 - min(1.0, effective_speed_kmh / v_free) if v_free > 0 else 0.0
        ci_length_sum += seg.length_m * max(0.0, seg_ci)

        # Track provenance
        prov_key = speed_result.provenance_type.value
        provenance_counts[prov_key] = provenance_counts.get(prov_key, 0) + 1

        # Advance simulated arrival time for next segment
        curr_arrival_time = curr_arrival_time + timedelta(seconds=seg_pred_time_s)

    # 4. Final Aggregations and Bottleneck Extraction
    total_delay_s = total_pred_time_s - total_ff_time_s
    total_pos_delay_s = sum(max(0.0, s.signed_delay_seconds) for s in evaluated_segments)
    total_neg_delay_s = sum(max(0.0, -s.signed_delay_seconds) for s in evaluated_segments)

    bottlenecks = extract_route_bottlenecks(
        evaluated_segments=evaluated_segments,
        min_delay_seconds=min_bottleneck_delay_seconds,
        max_bottlenecks=max_bottlenecks,
    )

    avg_speed_kmh = (total_dist_m / 1000.0) / (total_pred_time_s / 3600.0) if total_pred_time_s > 0 else 0.0

    mean_ci = ci_length_sum / total_dist_m if total_dist_m > 0 else 0.0
    if mean_ci < 0.20:
        overall_los = CongestionLevel.LOW
    elif mean_ci < 0.40:
        overall_los = CongestionLevel.MODERATE
    elif mean_ci < 0.60:
        overall_los = CongestionLevel.HIGH
    else:
        overall_los = CongestionLevel.SEVERE

    return EvaluatedRoute(
        route_id=route.route_id,
        name=route.name,
        departure_time=departure_time,
        estimated_arrival_time=curr_arrival_time,
        total_distance_m=round(total_dist_m, 2),
        total_distance_km=round(total_dist_m / 1000.0, 3),
        free_flow_time_seconds=round(total_ff_time_s, 2),
        predicted_travel_time_seconds=round(total_pred_time_s, 2),
        predicted_travel_time_minutes=round(total_pred_time_s / 60.0, 2),
        total_delay_seconds=round(total_delay_s, 2),
        total_delay_minutes=round(total_delay_s / 60.0, 2),
        total_positive_delay_seconds=round(total_pos_delay_s, 2),
        total_positive_delay_minutes=round(total_pos_delay_s / 60.0, 2),
        average_speed_kmh=round(avg_speed_kmh, 1),
        route_congestion_level=overall_los,
        bottlenecks=bottlenecks,
        evaluated_segments=evaluated_segments,
        data_provenance={
            "provider": type(speed_provider).__name__,
            "segment_count": len(evaluated_segments),
            "provenance_distribution": provenance_counts,
            "mean_congestion_index": round(mean_ci, 4),
            "gross_positive_delay_seconds": round(total_pos_delay_s, 2),
            "gross_negative_delay_seconds": round(total_neg_delay_s, 2),
            "bottleneck_count": len(bottlenecks),
            "total_bottleneck_delay_seconds": round(sum(b.delay_seconds for b in bottlenecks), 2),
        },
    )


class PredictiveRouteEvaluator:
    """Unified facade for predictive multi-route evaluation, ranking, and recommendation."""

    def __init__(
        self,
        speed_provider: Optional[BaseSpeedProvider] = None,
        config: Optional[PredictiveEvaluatorConfig] = None,
        road_graph: Optional[RoadNetworkGraph] = None,
    ):
        if speed_provider is not None and not isinstance(speed_provider, BaseSpeedProvider):
            raise ValueError(f"speed_provider must be a BaseSpeedProvider instance, got {type(speed_provider).__name__}")
        if config is not None and not isinstance(config, PredictiveEvaluatorConfig):
            raise ValueError(f"config must be a PredictiveEvaluatorConfig instance, got {type(config).__name__}")
        if road_graph is not None and not isinstance(road_graph, RoadNetworkGraph):
            raise ValueError(f"road_graph must be a RoadNetworkGraph instance, got {type(road_graph).__name__}")

        self.speed_provider = speed_provider
        self.config = config or PredictiveEvaluatorConfig()
        self.road_graph = road_graph

    def evaluate_routes(
        self,
        primary_route: Route,
        alternative_routes: Optional[List[Route]] = None,
        departure_time: Optional[datetime] = None,
        speed_provider: Optional[BaseSpeedProvider] = None,
        config: Optional[PredictiveEvaluatorConfig] = None,
        road_graph: Optional[RoadNetworkGraph] = None,
    ) -> RouteComparisonResult:
        """
        Evaluate primary route and alternative routes at identical departure time.

        Args:
            primary_route: P6 primary Route object.
            alternative_routes: Optional list of alternative Route objects.
            departure_time: Target departure timestamp (must be timezone-aware).
            speed_provider: Speed provider override.
            config: Evaluator config override.
            road_graph: Road network graph override.

        Returns:
            RouteComparisonResult with deterministic ranking, recommendation, and provenance.
        """
        if not isinstance(primary_route, Route):
            raise ValueError(f"primary_route must be a Route instance, got {type(primary_route).__name__}")

        if departure_time is None:
            raise ValueError("departure_time must be provided as a timezone-aware datetime")

        if not isinstance(departure_time, datetime) or isinstance(departure_time, bool):
            raise ValueError(f"departure_time must be a datetime instance, got {departure_time}")

        if departure_time.tzinfo is None or departure_time.tzinfo.utcoffset(departure_time) is None:
            raise ValueError(f"departure_time must be timezone-aware (tzinfo is required), got naive {departure_time}")

        provider = speed_provider or self.speed_provider
        if provider is None:
            raise ValueError("speed_provider must be provided to evaluate routes")
        if not isinstance(provider, BaseSpeedProvider):
            raise ValueError(f"speed_provider must be a BaseSpeedProvider instance, got {type(provider).__name__}")

        cfg = config or self.config
        if not isinstance(cfg, PredictiveEvaluatorConfig):
            raise ValueError(f"config must be a PredictiveEvaluatorConfig instance, got {type(cfg).__name__}")

        graph = road_graph or self.road_graph
        if graph is not None and not isinstance(graph, RoadNetworkGraph):
            raise ValueError(f"road_graph must be a RoadNetworkGraph instance, got {type(graph).__name__}")

        alts = alternative_routes or []
        for alt in alts:
            if not isinstance(alt, Route):
                raise ValueError(f"Alternative route must be a Route instance, got {type(alt).__name__}")

        # 1. Forward propagate primary route
        primary_eval = forward_propagate_route(
            route=primary_route,
            departure_time=departure_time,
            speed_provider=provider,
            road_graph=graph,
            crawl_speed_kmh=cfg.zero_speed_crawl_kmh,
            min_bottleneck_delay_seconds=cfg.min_bottleneck_delay_seconds,
            max_bottlenecks=cfg.max_bottlenecks_per_route,
        )

        # 2. Forward propagate each alternative route
        alt_evals: List[EvaluatedRoute] = []
        for alt in alts:
            alt_eval = forward_propagate_route(
                route=alt,
                departure_time=departure_time,
                speed_provider=provider,
                road_graph=graph,
                crawl_speed_kmh=cfg.zero_speed_crawl_kmh,
                min_bottleneck_delay_seconds=cfg.min_bottleneck_delay_seconds,
                max_bottlenecks=cfg.max_bottlenecks_per_route,
            )
            alt_evals.append(alt_eval)

        # 3. Deterministic ranking across all routes
        all_evals = [primary_eval] + alt_evals
        ranked_routes = sorted(
            all_evals,
            key=lambda r: (
                round(r.predicted_travel_time_seconds, 2),
                round(r.total_distance_m, 1),
                0 if r.route_id == primary_eval.route_id else 1,
                r.route_id,
            ),
        )

        # 4. Recommendation and time savings calculation
        primary_time = primary_eval.predicted_travel_time_seconds

        if len(primary_route.edge_sequence) == 0:
            recommended_route_id = primary_eval.route_id
            savings_s = 0.0
            reason = "Origin and destination are at the same location; zero travel time expected."

        elif len(alt_evals) == 0:
            recommended_route_id = primary_eval.route_id
            savings_s = 0.0
            reason = "Primary route evaluated as the sole viable candidate (no alternative routes provided)."

        else:
            best_alt = min(
                alt_evals,
                key=lambda a: (
                    round(a.predicted_travel_time_seconds, 2),
                    round(a.total_distance_m, 1),
                    a.route_id,
                ),
            )
            time_diff = primary_time - best_alt.predicted_travel_time_seconds
            dist_diff = primary_eval.total_distance_m - best_alt.total_distance_m

            if time_diff <= 0.0:
                # Primary is faster than or equal to the best alternative
                if abs(time_diff) <= cfg.tie_threshold_seconds and dist_diff > 50.0 and cfg.min_time_savings_seconds <= 0.0:
                    recommended_route_id = best_alt.route_id
                    savings_s = 0.0
                    reason = (
                        f"Alternative route '{best_alt.name}' preferred; travel time is tied within "
                        f"{cfg.tie_threshold_seconds:.1f}s and distance is shorter by {dist_diff:.1f}m."
                    )
                else:
                    recommended_route_id = primary_eval.route_id
                    savings_s = 0.0
                    if abs(time_diff) <= cfg.tie_threshold_seconds:
                        reason = (
                            f"Primary route preferred; alternative route '{best_alt.name}' travel time is tied within "
                            f"{cfg.tie_threshold_seconds:.1f}s and distance difference is within 50.0m."
                        )
                    else:
                        reason = "Primary route offers the lowest estimated travel time under predicted traffic conditions."

            elif time_diff < cfg.min_time_savings_seconds:
                # Alternative is faster, but saves less than minimum required savings
                recommended_route_id = primary_eval.route_id
                savings_s = 0.0
                reason = (
                    f"Primary route retained; fastest alternative '{best_alt.name}' saves only {time_diff:.1f}s "
                    f"({time_diff / 60.0:.1f} mins), which is below the minimum recommendation threshold of "
                    f"{cfg.min_time_savings_seconds:.1f}s."
                )

            else:
                # Alternative saves >= min_time_savings_seconds
                if time_diff <= cfg.tie_threshold_seconds and dist_diff <= 50.0:
                    recommended_route_id = primary_eval.route_id
                    savings_s = 0.0
                    reason = (
                        f"Primary route preferred; alternative route '{best_alt.name}' travel time is within "
                        f"tie threshold ({cfg.tie_threshold_seconds:.1f}s) and distance difference is within 50.0m."
                    )
                else:
                    recommended_route_id = best_alt.route_id
                    savings_s = round(time_diff, 2)
                    savings_min = round(savings_s / 60.0, 1)
                    if time_diff <= cfg.tie_threshold_seconds and dist_diff > 50.0:
                        reason = (
                            f"Alternative route '{best_alt.name}' preferred; travel time is tied within "
                            f"{cfg.tie_threshold_seconds:.1f}s and distance is shorter by {dist_diff:.1f}m."
                        )
                    else:
                        reason = (
                            f"Alternative route '{best_alt.name}' has lower estimated travel time "
                            f"(saves {savings_min:.1f} mins) due to reduced congestion."
                        )

        # 5. Build consolidated provenance
        prov_summary = {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "speed_provider": type(provider).__name__,
            "route_count_evaluated": len(all_evals),
            "primary_route_id": primary_eval.route_id,
            "recommended_route_id": recommended_route_id,
            "min_time_savings_seconds_threshold": cfg.min_time_savings_seconds,
            "tie_threshold_seconds": cfg.tie_threshold_seconds,
            "routes_summary": [
                {
                    "route_id": r.route_id,
                    "name": r.name,
                    "distance_km": r.total_distance_km,
                    "predicted_time_seconds": r.predicted_travel_time_seconds,
                    "predicted_time_minutes": r.predicted_travel_time_minutes,
                    "delay_seconds": r.total_delay_seconds,
                    "average_speed_kmh": r.average_speed_kmh,
                    "route_congestion_level": r.route_congestion_level.value,
                    "bottleneck_count": len(r.bottlenecks),
                    "fallback_used": any(s.speed_result.is_fallback for s in r.evaluated_segments),
                }
                for r in ranked_routes
            ],
        }

        return RouteComparisonResult(
            departure_time=departure_time,
            recommended_route_id=recommended_route_id,
            recommendation_reason=reason,
            time_savings_seconds=savings_s,
            time_savings_minutes=round(savings_s / 60.0, 2),
            primary_route=primary_eval,
            alternative_routes=alt_evals,
            ranked_routes=ranked_routes,
            data_provenance=prov_summary,
        )

    def evaluate_route_plan(
        self,
        plan_result: Any = None,
        departure_time: Any = None,
        speed_provider: Optional[BaseSpeedProvider] = None,
        config: Optional[PredictiveEvaluatorConfig] = None,
        road_graph: Optional[RoadNetworkGraph] = None,
    ) -> RouteComparisonResult:
        """
        Evaluate and compare a P6 RoutePlanResult under dynamic predictive traffic.

        Supports both instance call: evaluator.evaluate_route_plan(plan, dep, ...)
        and class call: PredictiveRouteEvaluator.evaluate_route_plan(plan, dep, provider, ...)
        """
        if isinstance(self, RoutePlanResult):
            # Invoked as PredictiveRouteEvaluator.evaluate_route_plan(plan_result, departure_time, speed_provider, ...)
            actual_plan = self
            actual_dep = plan_result
            actual_provider = departure_time
            evaluator = PredictiveRouteEvaluator(
                speed_provider=actual_provider,
                config=config,
                road_graph=road_graph,
            )
            return evaluator.evaluate_route_plan(
                plan_result=actual_plan,
                departure_time=actual_dep,
                speed_provider=actual_provider,
                config=config,
                road_graph=road_graph,
            )

        if not isinstance(plan_result, RoutePlanResult):
            raise ValueError(f"plan_result must be a RoutePlanResult instance, got {type(plan_result).__name__}")

        if not plan_result.success or plan_result.primary_route is None:
            raise ValueError(
                f"Cannot evaluate unsuccessful RoutePlanResult (status={plan_result.status.value}): "
                f"{plan_result.status_message}"
            )

        return self.evaluate_routes(
            primary_route=plan_result.primary_route,
            alternative_routes=plan_result.alternative_routes,
            departure_time=departure_time,
            speed_provider=speed_provider,
            config=config,
            road_graph=road_graph,
        )

    @classmethod
    def evaluate_plan(
        cls,
        plan_result: RoutePlanResult,
        departure_time: datetime,
        speed_provider: BaseSpeedProvider,
        config: Optional[PredictiveEvaluatorConfig] = None,
        road_graph: Optional[RoadNetworkGraph] = None,
    ) -> RouteComparisonResult:
        """Convenience classmethod to evaluate a RoutePlanResult without prior instantiation."""
        evaluator = cls(speed_provider=speed_provider, config=config, road_graph=road_graph)
        return evaluator.evaluate_route_plan(
            plan_result=plan_result,
            departure_time=departure_time,
            speed_provider=speed_provider,
            config=config,
            road_graph=road_graph,
        )

