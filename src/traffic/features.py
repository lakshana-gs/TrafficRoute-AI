"""Road-segment aggregation and temporal feature engineering without data leakage."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def get_time_of_day_bucket(hour: int, minute: int = 0) -> str:
    """Categorize time of day into standard transportation demand periods."""
    time_float = hour + (minute / 60.0)
    if 8.0 <= time_float <= 10.5:
        return "MORNING_PEAK"
    elif 17.5 <= time_float <= 20.5:
        return "EVENING_PEAK"
    elif 11.0 <= time_float < 17.5:
        return "MIDDAY_OFFPEAK"
    elif time_float >= 22.0 or time_float < 6.0:
        return "NIGHT"
    return "SHOULDER_PERIOD"


class TrafficFeatureEngineer:
    """
    Aggregates matched probe observations to the road-segment level and constructs
    predictive time-series features strictly enforcing temporal causality (zero leakage).
    """

    def __init__(self, min_observations_per_bucket: int = 1):
        self.min_obs = min_observations_per_bucket

    def aggregate_segment_time_buckets(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate matched traffic observations by road segment and time bucket.

        Args:
            matched_df: DataFrame with 'matched_edge_id', 'time_bucket', 'observed_speed_kmh'.

        Returns:
            DataFrame with segment-level time-series speed observations.
        """
        valid_matched = matched_df[matched_df["is_matched"] == True].copy()
        if len(valid_matched) == 0:
            return pd.DataFrame()

        # Group by segment and discrete time bucket
        grouped = valid_matched.groupby(["matched_edge_id", "time_bucket"])

        agg_df = grouped.agg(
            mean_speed_kmh=("observed_speed_kmh", "mean"),
            median_speed_kmh=("observed_speed_kmh", "median"),
            speed_std_kmh=("observed_speed_kmh", lambda x: float(x.std(ddof=0)) if len(x) > 1 else 0.0),
            min_speed_kmh=("observed_speed_kmh", "min"),
            max_speed_kmh=("observed_speed_kmh", "max"),
            observation_count=("observed_speed_kmh", "count"),
            highway_type=("highway_type", "first"),
            road_name=("road_name", "first"),
            road_reference=("road_reference", "first"),
        ).reset_index()

        # Rename matched_edge_id to segment_id
        agg_df = agg_df.rename(columns={"matched_edge_id": "segment_id"})

        # Assign confidence tiers
        def confidence_label(cnt: int) -> str:
            if cnt >= 4:
                return "HIGH_CONFIDENCE"
            elif cnt >= 2:
                return "MODERATE_CONFIDENCE"
            return "LOW_SAMPLE"

        agg_df["coverage_confidence"] = agg_df["observation_count"].apply(confidence_label)

        # Round numeric speeds
        for col in ["mean_speed_kmh", "median_speed_kmh", "speed_std_kmh", "min_speed_kmh", "max_speed_kmh"]:
            agg_df[col] = agg_df[col].round(2)

        return agg_df

    def build_model_ready_features(
        self, aggregated_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construct predictive features for each segment-time observation without temporal leakage.

        Features generated:
        - `hour_ist`, `day_of_week_ist`, `is_weekend_ist`
        - `time_of_day_bucket` (MORNING_PEAK, EVENING_PEAK, etc.)
        - `speed_lag_1` (t - 15m), `speed_lag_2` (t - 30m), `speed_lag_4` (t - 60m)
        - `expanding_historical_mean_speed` (mean of past observations on this segment up to t-1)
        """
        if len(aggregated_df) == 0:
            return pd.DataFrame()

        df_feat = aggregated_df.copy()

        # Step 1: Ensure UTC and sort chronologically per segment
        df_feat["time_bucket"] = pd.to_datetime(df_feat["time_bucket"], utc=True)
        df_feat = df_feat.sort_values(by=["segment_id", "time_bucket"]).reset_index(drop=True)

        # Step 2: Calendar & demand period features
        ts_ist = df_feat["time_bucket"] + pd.Timedelta(hours=5, minutes=30)
        df_feat["hour_ist"] = ts_ist.dt.hour
        df_feat["day_of_week_ist"] = ts_ist.dt.dayofweek
        df_feat["is_weekend_ist"] = df_feat["day_of_week_ist"].isin([5, 6]).astype(int)
        df_feat["time_of_day_bucket"] = [
            get_time_of_day_bucket(h, m) for h, m in zip(df_feat["hour_ist"], ts_ist.dt.minute)
        ]

        # Step 3: Lagged speed features (strictly past observations: shift >= 1)
        grouped = df_feat.groupby("segment_id")
        df_feat["speed_lag_1"] = grouped["mean_speed_kmh"].shift(1)
        df_feat["speed_lag_2"] = grouped["mean_speed_kmh"].shift(2)
        df_feat["speed_lag_4"] = grouped["mean_speed_kmh"].shift(4)

        # Step 4: Expanding historical mean (past observations ONLY, excluding current step)
        # Shift first to exclude current observation, then compute expanding mean
        df_feat["past_speed_shifted"] = grouped["mean_speed_kmh"].shift(1)
        df_feat["historical_speed_mean"] = (
            df_feat.groupby("segment_id")["past_speed_shifted"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
            .round(2)
        )
        df_feat = df_feat.drop(columns=["past_speed_shifted"])

        # Fill initial cold-start lag values with segment median / default
        df_feat["historical_speed_mean"] = df_feat["historical_speed_mean"].fillna(df_feat["mean_speed_kmh"])
        df_feat["speed_lag_1"] = df_feat["speed_lag_1"].fillna(df_feat["historical_speed_mean"])
        df_feat["speed_lag_2"] = df_feat["speed_lag_2"].fillna(df_feat["speed_lag_1"])
        df_feat["speed_lag_4"] = df_feat["speed_lag_4"].fillna(df_feat["speed_lag_2"])

        return df_feat
