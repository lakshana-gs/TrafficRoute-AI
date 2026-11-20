"""Data quality validation and anomaly detection for TrafficRoute AI traffic observations."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TrafficDataQualityAudit(BaseModel):
    """Structured audit report summarizing raw traffic dataset quality metrics."""

    total_records: int = Field(..., description="Total raw input rows")
    valid_records: int = Field(..., description="Rows passing all sanity checks")
    invalid_records: int = Field(..., description="Rows failing one or more quality rules")
    missing_coordinates: int = Field(..., description="Null or NaN lat/lon")
    out_of_bounds_coordinates: int = Field(..., description="Coordinates outside geographic bounding box")
    negative_speeds: int = Field(..., description="Observed speeds < 0 km/h")
    extreme_high_speeds: int = Field(..., description="Observed speeds > maximum threshold (e.g. 150 km/h)")
    zero_speeds: int = Field(..., description="Observed speeds == 0 km/h (stopped traffic)")
    duplicate_records: int = Field(..., description="Duplicate entity-timestamp combinations")
    missing_timestamps: int = Field(..., description="Null or unparseable timestamps")
    min_speed_observed: float = Field(..., description="Minimum recorded speed in raw dataset")
    max_speed_observed: float = Field(..., description="Maximum recorded speed in raw dataset")
    median_speed_observed: float = Field(..., description="Median speed of valid observations")
    mean_speed_observed: float = Field(..., description="Mean speed of valid observations")
    quality_pass_rate_pct: float = Field(..., description="Percentage of valid records")


class TrafficValidator:
    """
    Validates raw traffic observation DataFrames against geographic, physical,
    and temporal domain rules.
    """

    def __init__(
        self,
        bounding_box: Optional[Dict[str, float]] = None,
        max_speed_kmh: float = 150.0,
        min_speed_kmh: float = 0.0,
    ):
        self.bbox = bounding_box or {
            "south": 10.9400,
            "north": 11.2300,
            "west": 76.9000,
            "east": 77.3000,
        }
        self.max_speed_kmh = max_speed_kmh
        self.min_speed_kmh = min_speed_kmh

    def audit_dataset(self, df: pd.DataFrame) -> TrafficDataQualityAudit:
        """
        Perform a comprehensive data quality audit on a traffic DataFrame.
        """
        total = len(df)
        if total == 0:
            return TrafficDataQualityAudit(
                total_records=0,
                valid_records=0,
                invalid_records=0,
                missing_coordinates=0,
                out_of_bounds_coordinates=0,
                negative_speeds=0,
                extreme_high_speeds=0,
                zero_speeds=0,
                duplicate_records=0,
                missing_timestamps=0,
                min_speed_observed=0.0,
                max_speed_observed=0.0,
                median_speed_observed=0.0,
                mean_speed_observed=0.0,
                quality_pass_rate_pct=0.0,
            )

        # Missing values
        missing_coords = df["latitude"].isna() | df["longitude"].isna()
        missing_ts = df["timestamp"].isna()

        # Out of bounds coordinates
        lat = df["latitude"].fillna(0.0)
        lon = df["longitude"].fillna(0.0)
        out_of_bounds = (
            (lat < self.bbox["south"])
            | (lat > self.bbox["north"])
            | (lon < self.bbox["west"])
            | (lon > self.bbox["east"])
        ) & ~missing_coords

        # Speed checks
        speeds = df["observed_speed_kmh"].fillna(-999.0)
        negative_spd = speeds < self.min_speed_kmh
        extreme_spd = speeds > self.max_speed_kmh
        zero_spd = speeds == 0.0

        # Duplicate checks
        duplicates = df.duplicated(subset=["traffic_entity_id", "timestamp"], keep="first")

        # Invalid mask
        invalid_mask = missing_coords | missing_ts | out_of_bounds | negative_spd | extreme_spd | duplicates
        valid_mask = ~invalid_mask

        valid_speeds = df.loc[valid_mask, "observed_speed_kmh"]
        min_spd = float(df["observed_speed_kmh"].min()) if not df["observed_speed_kmh"].isna().all() else 0.0
        max_spd = float(df["observed_speed_kmh"].max()) if not df["observed_speed_kmh"].isna().all() else 0.0
        median_spd = float(valid_speeds.median()) if len(valid_speeds) > 0 else 0.0
        mean_spd = float(valid_speeds.mean()) if len(valid_speeds) > 0 else 0.0

        return TrafficDataQualityAudit(
            total_records=total,
            valid_records=int(valid_mask.sum()),
            invalid_records=int(invalid_mask.sum()),
            missing_coordinates=int(missing_coords.sum()),
            out_of_bounds_coordinates=int(out_of_bounds.sum()),
            negative_speeds=int(negative_spd.sum()),
            extreme_high_speeds=int(extreme_spd.sum()),
            zero_speeds=int(zero_spd.sum()),
            duplicate_records=int(duplicates.sum()),
            missing_timestamps=int(missing_ts.sum()),
            min_speed_observed=round(min_spd, 2),
            max_speed_observed=round(max_spd, 2),
            median_speed_observed=round(median_spd, 2),
            mean_speed_observed=round(mean_spd, 2),
            quality_pass_rate_pct=round((int(valid_mask.sum()) / total) * 100, 2),
        )
