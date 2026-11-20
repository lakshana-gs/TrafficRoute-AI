"""Traffic speed cleaning, anomaly tagging, and deduplication for TrafficRoute AI."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.traffic.schemas import QualityFlag


class CleaningAccountingReport(BaseModel):
    """Accounting ledger tracking every observation through the cleaning pipeline."""

    raw_count: int = Field(..., description="Initial raw record count")
    valid_count: int = Field(..., description="Normal valid observations")
    stopped_count: int = Field(..., description="Legitimate zero/near-zero speed traffic stops retained")
    negative_speed_count: int = Field(..., description="Negative speed anomalies removed")
    extreme_speed_count: int = Field(..., description="Speeds exceeding max threshold removed")
    out_of_bounds_count: int = Field(..., description="GPS coordinates outside corridor removed")
    missing_coords_count: int = Field(..., description="Null/NaN coordinates removed")
    duplicate_count: int = Field(..., description="Duplicate entity-timestamp observations removed")
    total_removed: int = Field(..., description="Total invalid records discarded")
    total_retained: int = Field(..., description="Total clean + stopped records retained for modeling")
    retention_rate_pct: float = Field(..., description="Percentage of records retained")


class TrafficCleaner:
    """
    Cleans raw traffic observations by tagging anomalies, deduplicating,
    and filtering out corrupt records while preserving legitimate traffic stops.
    """

    def __init__(
        self,
        bounding_box: Optional[Dict[str, float]] = None,
        min_speed_kmh: float = 0.0,
        max_speed_kmh: float = 150.0,
        stopped_speed_threshold_kmh: float = 3.0,
    ):
        self.bbox = bounding_box or {
            "south": 10.9400,
            "north": 11.2300,
            "west": 76.9000,
            "east": 77.3000,
        }
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh
        self.stopped_threshold = stopped_speed_threshold_kmh

    def clean_observations(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, CleaningAccountingReport]:
        """
        Clean traffic observations DataFrame and produce an accounting ledger.

        Args:
            df: Raw traffic observations DataFrame.

        Returns:
            Tuple of (cleaned_retained_df, accounting_report).
        """
        raw_count = len(df)
        if raw_count == 0:
            empty_report = CleaningAccountingReport(
                raw_count=0,
                valid_count=0,
                stopped_count=0,
                negative_speed_count=0,
                extreme_speed_count=0,
                out_of_bounds_count=0,
                missing_coords_count=0,
                duplicate_count=0,
                total_removed=0,
                total_retained=0,
                retention_rate_pct=0.0,
            )
            return df.copy(), empty_report

        df_work = df.copy()

        # Step 1: Initialize quality flags
        df_work["quality_flag"] = QualityFlag.VALID.value
        df_work["is_retained"] = True

        # Step 2: Missing coordinates / timestamps
        missing_coords = df_work["latitude"].isna() | df_work["longitude"].isna()
        df_work.loc[missing_coords, "quality_flag"] = QualityFlag.MISSING_COORDINATES.value
        df_work.loc[missing_coords, "is_retained"] = False

        # Step 3: Out of bounds
        valid_coords = ~missing_coords
        lat = df_work["latitude"]
        lon = df_work["longitude"]
        out_of_bounds = valid_coords & (
            (lat < self.bbox["south"])
            | (lat > self.bbox["north"])
            | (lon < self.bbox["west"])
            | (lon > self.bbox["east"])
        )
        df_work.loc[out_of_bounds, "quality_flag"] = QualityFlag.OUT_OF_BOUNDS.value
        df_work.loc[out_of_bounds, "is_retained"] = False

        # Step 4: Speed anomalies
        speeds = df_work["observed_speed_kmh"].fillna(-999.0)
        negative_speeds = speeds < self.min_speed_kmh
        df_work.loc[negative_speeds, "quality_flag"] = QualityFlag.INVALID_SPEED.value
        df_work.loc[negative_speeds, "is_retained"] = False

        extreme_speeds = speeds > self.max_speed_kmh
        df_work.loc[extreme_speeds, "quality_flag"] = QualityFlag.OUTLIER.value
        df_work.loc[extreme_speeds, "is_retained"] = False

        # Step 5: Legitimate Stopped Traffic (0 <= v <= 3 km/h with valid coordinates)
        stopped_traffic = (
            (speeds >= self.min_speed_kmh)
            & (speeds <= self.stopped_threshold)
            & valid_coords
            & ~out_of_bounds
        )
        df_work.loc[stopped_traffic, "quality_flag"] = QualityFlag.STOPPED.value
        df_work.loc[stopped_traffic, "is_retained"] = True

        # Step 6: Deduplication (keep first occurrence of entity-timestamp)
        duplicates = df_work.duplicated(subset=["traffic_entity_id", "timestamp"], keep="first")
        df_work.loc[duplicates, "quality_flag"] = QualityFlag.DUPLICATE.value
        df_work.loc[duplicates, "is_retained"] = False

        # Compute summary accounting
        retained_mask = df_work["is_retained"] == True
        retained_df = df_work[retained_mask].copy()

        normal_valid = (df_work["quality_flag"] == QualityFlag.VALID.value).sum()
        stopped_cnt = (df_work["quality_flag"] == QualityFlag.STOPPED.value).sum()
        neg_spd_cnt = negative_speeds.sum()
        ext_spd_cnt = extreme_speeds.sum()
        oob_cnt = out_of_bounds.sum()
        missing_cnt = missing_coords.sum()
        dup_cnt = duplicates.sum()
        total_removed = raw_count - len(retained_df)
        total_retained = len(retained_df)
        retention_rate = round((total_retained / max(raw_count, 1)) * 100, 2)

        report = CleaningAccountingReport(
            raw_count=raw_count,
            valid_count=int(normal_valid),
            stopped_count=int(stopped_cnt),
            negative_speed_count=int(neg_spd_cnt),
            extreme_speed_count=int(ext_spd_cnt),
            out_of_bounds_count=int(oob_cnt),
            missing_coords_count=int(missing_cnt),
            duplicate_count=int(dup_cnt),
            total_removed=int(total_removed),
            total_retained=int(total_retained),
            retention_rate_pct=retention_rate,
        )

        return retained_df, report
