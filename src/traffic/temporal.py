"""Temporal normalization, interval analysis, and gap detection for TrafficRoute AI."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TemporalQualityReport(BaseModel):
    """Report detailing sampling consistency, interval distributions, and temporal coverage."""

    earliest_timestamp: str = Field(..., description="Earliest observation timestamp (UTC)")
    latest_timestamp: str = Field(..., description="Latest observation timestamp (UTC)")
    total_time_span_hours: float = Field(..., description="Total time span covered in hours")
    expected_sampling_interval_s: float = Field(..., description="Configured expected sampling interval in seconds")
    median_sampling_interval_s: float = Field(..., description="Observed median interval between consecutive observations")
    p95_sampling_interval_s: float = Field(..., description="95th percentile sampling interval")
    total_gaps_detected: int = Field(..., description="Number of significant time gaps exceeding threshold")
    longest_gap_hours: float = Field(..., description="Duration of longest telemetry blackout in hours")
    total_time_buckets: int = Field(..., description="Number of distinct discrete time buckets")


class TemporalNormalizer:
    """
    Normalizes timestamps, generates local IST time features, assesses sampling regularities,
    and assigns discrete time buckets.
    """

    def __init__(
        self,
        bucket_size_minutes: int = 15,
        expected_interval_s: float = 900.0,
        gap_multiplier: float = 3.0,
    ):
        self.bucket_size_min = bucket_size_minutes
        self.expected_interval_s = expected_interval_s
        self.gap_threshold_s = expected_interval_s * gap_multiplier

    def normalize_and_bucket(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, TemporalQualityReport]:
        """
        Normalize DataFrame timestamps, add IST calendar fields, assign time buckets,
        and generate a temporal quality audit.

        Args:
            df: Cleaned traffic observations DataFrame.

        Returns:
            Tuple of (temporally_normalized_df, temporal_quality_report).
        """
        if len(df) == 0:
            empty_report = TemporalQualityReport(
                earliest_timestamp="",
                latest_timestamp="",
                total_time_span_hours=0.0,
                expected_sampling_interval_s=self.expected_interval_s,
                median_sampling_interval_s=0.0,
                p95_sampling_interval_s=0.0,
                total_gaps_detected=0,
                longest_gap_hours=0.0,
                total_time_buckets=0,
            )
            return df.copy(), empty_report

        df_work = df.copy()

        # Step 1: Ensure UTC datetime
        if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
            df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], utc=True)

        # Step 2: Sort chronologically
        df_work = df_work.sort_values(by=["timestamp", "traffic_entity_id"]).reset_index(drop=True)

        # Step 3: Compute IST local time (+05:30)
        df_work["timestamp_ist"] = df_work["timestamp"] + pd.Timedelta(hours=5, minutes=30)
        df_work["hour_ist"] = df_work["timestamp_ist"].dt.hour
        df_work["day_of_week_ist"] = df_work["timestamp_ist"].dt.dayofweek  # 0=Monday, 6=Sunday
        df_work["is_weekend_ist"] = df_work["day_of_week_ist"].isin([5, 6]).astype(int)

        # Step 4: Assign discrete time bucket (floor to bucket_size_min)
        bucket_freq = f"{self.bucket_size_min}min"
        df_work["time_bucket"] = df_work["timestamp"].dt.floor(bucket_freq)

        # Step 5: Sampling interval & gap analysis
        earliest = df_work["timestamp"].min()
        latest = df_work["timestamp"].max()
        time_span_hours = round((latest - earliest).total_seconds() / 3600.0, 2)

        # Compute interval distribution per entity
        df_work["prev_timestamp"] = df_work.groupby("traffic_entity_id")["timestamp"].shift(1)
        intervals = (df_work["timestamp"] - df_work["prev_timestamp"]).dt.total_seconds().dropna()

        if len(intervals) > 0:
            median_interval = float(intervals.median())
            p95_interval = float(np.percentile(intervals, 95))
            gaps = intervals[intervals > self.gap_threshold_s]
            total_gaps = int(len(gaps))
            longest_gap_h = round(float(intervals.max()) / 3600.0, 2)
        else:
            median_interval = self.expected_interval_s
            p95_interval = self.expected_interval_s
            total_gaps = 0
            longest_gap_h = 0.0

        df_work = df_work.drop(columns=["prev_timestamp"])

        report = TemporalQualityReport(
            earliest_timestamp=earliest.isoformat(),
            latest_timestamp=latest.isoformat(),
            total_time_span_hours=time_span_hours,
            expected_sampling_interval_s=self.expected_interval_s,
            median_sampling_interval_s=round(median_interval, 1),
            p95_sampling_interval_s=round(p95_interval, 1),
            total_gaps_detected=total_gaps,
            longest_gap_hours=longest_gap_h,
            total_time_buckets=int(df_work["time_bucket"].nunique()),
        )

        return df_work, report
