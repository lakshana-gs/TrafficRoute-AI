"""Strict canonical schemas for raw, cleaned, and aggregated traffic observations in TrafficRoute AI."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class QualityFlag(str, Enum):
    """Data quality classification for individual traffic observations."""

    VALID = "VALID"
    STOPPED = "STOPPED"  # Legitimate zero/near-zero speed traffic congestion stop
    INVALID_SPEED = "INVALID_SPEED"  # Negative speed or physically impossible speed
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"  # Outside Tamil Nadu geographic bounding box
    OUTLIER = "OUTLIER"  # Statistical speed spike outlier
    DUPLICATE = "DUPLICATE"  # Identical entity timestamp observation
    MISSING_COORDINATES = "MISSING_COORDINATES"
    INTERPOLATED = "INTERPOLATED"  # Derived/interpolated observation


class VehicleClass(str, Enum):
    """Standardized vehicle classes reflecting Indian heterogeneous traffic."""

    TWO_WHEELER = "TWO_WHEELER"
    THREE_WHEELER = "THREE_WHEELER"  # Auto-rickshaw
    CAR = "CAR"
    LCV = "LCV"  # Light Commercial Vehicle
    BUS = "BUS"
    HCV = "HCV"  # Heavy Commercial Vehicle / Multi-axle truck
    UNKNOWN = "UNKNOWN"


class RawTrafficObservation(BaseModel):
    """Canonical schema for raw, incoming traffic probe observations."""

    timestamp: datetime = Field(..., description="Timestamp of observation in UTC or explicit ISO 8601")
    traffic_entity_id: str = Field(..., description="Unique vehicle, probe, or trajectory identifier")
    latitude: float = Field(..., description="WGS84 Latitude in degrees")
    longitude: float = Field(..., description="WGS84 Longitude in degrees")
    observed_speed_kmh: float = Field(..., description="Observed vehicle speed in km/h")
    heading: Optional[float] = Field(default=None, description="Compass heading in degrees (0 - 360)")
    direction: Optional[str] = Field(default=None, description="General travel direction (e.g. Eastbound, Westbound)")
    vehicle_class: Optional[VehicleClass] = Field(default=VehicleClass.UNKNOWN, description="Vehicle category")
    road_name: Optional[str] = Field(default=None, description="Reported street or corridor name if known")
    road_reference: Optional[str] = Field(default=None, description="Reported highway reference (e.g. NH-544)")
    source: str = Field(default="Empirical_Telemetry", description="Originating data source identifier")

    @field_validator("latitude")
    @classmethod
    def check_latitude_range(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude {v} out of valid range [-90, 90]")
        return v

    @field_validator("longitude")
    @classmethod
    def check_longitude_range(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude {v} out of valid range [-180, 180]")
        return v


class CleanedTrafficObservation(RawTrafficObservation):
    """Schema for validated and cleaned traffic observations with quality flags."""

    quality_flag: QualityFlag = Field(default=QualityFlag.VALID, description="Cleaning and validation flag")
    is_retained: bool = Field(default=True, description="Whether observation is retained for downstream modeling")


class MatchedTrafficObservation(CleanedTrafficObservation):
    """Traffic observation matched to a P1 road network edge."""

    matched_edge_id: Optional[str] = Field(default=None, description="Matched P1 road edge ID ({u}_{v}_{key})")
    matched_u: Optional[int] = Field(default=None, description="Origin node of matched edge")
    matched_v: Optional[int] = Field(default=None, description="Target node of matched edge")
    matched_key: Optional[int] = Field(default=0, description="Multigraph key index")
    match_distance_m: Optional[float] = Field(default=None, description="Perpendicular distance to road centerline in meters")
    highway_type: Optional[str] = Field(default=None, description="Road classification of matched edge")
    is_matched: bool = Field(default=False, description="True if matched within threshold distance")


class SegmentAggregatedObservation(BaseModel):
    """Model-ready traffic state aggregated at road-segment and time-bucket level."""

    segment_id: str = Field(..., description="P1 road edge ID ({u}_{v}_{key})")
    time_bucket: datetime = Field(..., description="Floor timestamp of time interval (UTC)")
    mean_speed_kmh: float = Field(..., description="Mean observed speed in km/h during bucket")
    median_speed_kmh: float = Field(..., description="Median observed speed in km/h")
    speed_std_kmh: float = Field(default=0.0, description="Standard deviation of observed speed")
    min_speed_kmh: float = Field(..., description="Minimum observed speed in bucket")
    max_speed_kmh: float = Field(..., description="Maximum observed speed in bucket")
    observation_count: int = Field(..., description="Number of distinct observations aggregated in bucket")
    highway_type: str = Field(default="unclassified", description="Road classification")
    road_name: Optional[str] = Field(default=None, description="Road name if available")
    road_reference: Optional[str] = Field(default=None, description="Highway code (e.g. NH-544)")
    coverage_confidence: str = Field(default="HIGH", description="Confidence tier based on observation count")
