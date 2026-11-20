"""Spatial road matching between traffic observations and P1 road network graph."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.network.graph import RoadNetworkGraph
from src.network.snapping import NetworkSpatialIndexer, SnapResult


class SpatialMatchingReport(BaseModel):
    """Report summarizing spatial linkage quality between traffic telemetry and P1 road network."""

    total_observations: int = Field(..., description="Total input observations")
    matched_observations: int = Field(..., description="Observations within snapping threshold")
    unmatched_observations: int = Field(..., description="Observations exceeding maximum snapping radius")
    match_rate_pct: float = Field(..., description="Percentage of observations successfully matched")
    mean_match_distance_m: float = Field(..., description="Average perpendicular snap distance in meters")
    median_match_distance_m: float = Field(..., description="Median perpendicular snap distance in meters")
    p95_match_distance_m: float = Field(..., description="95th percentile snap distance in meters")
    max_match_distance_m: float = Field(..., description="Maximum snap distance among matched records")
    unique_edges_matched: int = Field(..., description="Number of distinct P1 road segments receiving observations")


class TrafficSpatialMatcher:
    """
    Spatially links raw traffic coordinates to the P1 road graph using KD-Tree projection,
    ensuring unmatched observations are never artificially forced onto roads.
    """

    def __init__(
        self,
        road_graph: RoadNetworkGraph,
        spatial_indexer: Optional[NetworkSpatialIndexer] = None,
        max_snap_distance_m: float = 75.0,
    ):
        self.road_graph = road_graph
        self.indexer = spatial_indexer or NetworkSpatialIndexer(
            road_graph, max_snap_distance_m=max_snap_distance_m
        )
        self.max_distance_m = max_snap_distance_m

    def match_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, SpatialMatchingReport]:
        """
        Spatially match a DataFrame of traffic observations to P1 road edges.

        Args:
            df: Cleaned and normalized traffic observations DataFrame.

        Returns:
            Tuple of (matched_df, spatial_matching_report).
        """
        if len(df) == 0:
            empty_report = SpatialMatchingReport(
                total_observations=0,
                matched_observations=0,
                unmatched_observations=0,
                match_rate_pct=0.0,
                mean_match_distance_m=0.0,
                median_match_distance_m=0.0,
                p95_match_distance_m=0.0,
                max_match_distance_m=0.0,
                unique_edges_matched=0,
            )
            return df.copy(), empty_report

        df_work = df.copy()

        matched_edge_ids: List[Optional[str]] = []
        matched_us: List[Optional[int]] = []
        matched_vs: List[Optional[int]] = []
        matched_keys: List[Optional[int]] = []
        match_distances: List[Optional[float]] = []
        highway_types: List[Optional[str]] = []
        is_matched_flags: List[bool] = []

        # Cache coordinate queries to accelerate matching for repeated coordinates
        coord_cache: Dict[Tuple[float, float], SnapResult] = {}

        for _, row in df_work.iterrows():
            lat = round(float(row["latitude"]), 6)
            lon = round(float(row["longitude"]), 6)
            coord_key = (lat, lon)

            if coord_key not in coord_cache:
                snap = self.indexer.snap_coordinate(lat, lon, snap_to_edge=True)
                coord_cache[coord_key] = snap
            else:
                snap = coord_cache[coord_key]

            # Check if snap distance is within threshold
            if snap.is_valid and snap.distance_meters <= self.max_distance_m and snap.nearest_edge_id:
                matched_edge_ids.append(snap.nearest_edge_id)
                matched_us.append(snap.nearest_edge_u)
                matched_vs.append(snap.nearest_edge_v)
                matched_keys.append(snap.nearest_edge_key)
                match_distances.append(snap.distance_meters)
                highway_types.append(snap.highway_type)
                is_matched_flags.append(True)
            else:
                matched_edge_ids.append(None)
                matched_us.append(None)
                matched_vs.append(None)
                matched_keys.append(None)
                match_distances.append(snap.distance_meters if snap else None)
                highway_types.append(None)
                is_matched_flags.append(False)

        df_work["matched_edge_id"] = matched_edge_ids
        df_work["matched_u"] = matched_us
        df_work["matched_v"] = matched_vs
        df_work["matched_key"] = matched_keys
        df_work["match_distance_m"] = match_distances
        df_work["highway_type"] = highway_types
        df_work["is_matched"] = is_matched_flags

        # Compute matching statistics
        total = len(df_work)
        matched_count = int(sum(is_matched_flags))
        unmatched_count = total - matched_count
        match_rate = round((matched_count / max(total, 1)) * 100, 2)

        valid_distances = [d for d, m in zip(match_distances, is_matched_flags) if m and d is not None]
        if valid_distances:
            mean_dist = float(np.mean(valid_distances))
            median_dist = float(np.median(valid_distances))
            p95_dist = float(np.percentile(valid_distances, 95))
            max_dist = float(np.max(valid_distances))
        else:
            mean_dist, median_dist, p95_dist, max_dist = 0.0, 0.0, 0.0, 0.0

        unique_edges = int(df_work.loc[df_work["is_matched"], "matched_edge_id"].nunique())

        report = SpatialMatchingReport(
            total_observations=total,
            matched_observations=matched_count,
            unmatched_observations=unmatched_count,
            match_rate_pct=match_rate,
            mean_match_distance_m=round(mean_dist, 2),
            median_match_distance_m=round(median_dist, 2),
            p95_match_distance_m=round(p95_dist, 2),
            max_match_distance_m=round(max_dist, 2),
            unique_edges_matched=unique_edges,
        )

        return df_work, report
