"""End-to-end orchestrated Traffic Data Pipeline for TrafficRoute AI."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.network.graph import RoadNetworkGraph
from src.network.snapping import NetworkSpatialIndexer
from src.traffic.cleaning import CleaningAccountingReport, TrafficCleaner
from src.traffic.features import TrafficFeatureEngineer
from src.traffic.ingestion import generate_empirical_pilot_corridor_telemetry, load_raw_traffic_csv
from src.traffic.provenance import TrafficDatasetProvenance
from src.traffic.spatial import SpatialMatchingReport, TrafficSpatialMatcher
from src.traffic.temporal import TemporalNormalizer, TemporalQualityReport
from src.traffic.validation import TrafficDataQualityAudit, TrafficValidator


class PipelineExecutionSummary:
    """Container holding metrics across all pipeline stages."""

    def __init__(
        self,
        audit: TrafficDataQualityAudit,
        cleaning: CleaningAccountingReport,
        temporal: TemporalQualityReport,
        spatial: SpatialMatchingReport,
        model_ready_rows: int,
        unique_segments: int,
        train_rows: int,
        val_rows: int,
        test_rows: int,
        train_period: Tuple[str, str],
        val_period: Tuple[str, str],
        test_period: Tuple[str, str],
    ):
        self.audit = audit
        self.cleaning = cleaning
        self.temporal = temporal
        self.spatial = spatial
        self.model_ready_rows = model_ready_rows
        self.unique_segments = unique_segments
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.test_rows = test_rows
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit": self.audit.model_dump(),
            "cleaning": self.cleaning.model_dump(),
            "temporal": self.temporal.model_dump(),
            "spatial": self.spatial.model_dump(),
            "model_ready": {
                "total_rows": self.model_ready_rows,
                "unique_segments": self.unique_segments,
                "train_rows": self.train_rows,
                "val_rows": self.val_rows,
                "test_rows": self.test_rows,
                "train_period": self.train_period,
                "val_period": self.val_period,
                "test_period": self.test_period,
            },
        }


class TrafficDataPipeline:
    """
    Orchestrates the conversion of raw traffic observations into model-ready
    features and chronological dataset splits linked to the P1 road network.
    """

    def __init__(
        self,
        road_graph: RoadNetworkGraph,
        spatial_indexer: Optional[NetworkSpatialIndexer] = None,
        bucket_size_minutes: int = 15,
        max_snap_distance_m: float = 75.0,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        self.road_graph = road_graph
        self.validator = TrafficValidator()
        self.cleaner = TrafficCleaner()
        self.temporal_norm = TemporalNormalizer(bucket_size_minutes=bucket_size_minutes)
        self.spatial_matcher = TrafficSpatialMatcher(
            road_graph, spatial_indexer=spatial_indexer, max_snap_distance_m=max_snap_distance_m
        )
        self.feature_engineer = TrafficFeatureEngineer()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def run(
        self,
        raw_csv_path: Optional[Path | str] = None,
        output_dir: Path | str = "data/traffic",
    ) -> Tuple[pd.DataFrame, PipelineExecutionSummary]:
        """
        Execute full pipeline from raw ingestion to model-ready Parquet datasets.

        Args:
            raw_csv_path: Path to raw input CSV; if None, generates empirical corridor telemetry.
            output_dir: Base directory for storing processed datasets and metadata.

        Returns:
            Tuple of (model_ready_df, execution_summary).
        """
        out_base = Path(output_dir)
        raw_dir = out_base / "raw"
        interim_dir = out_base / "interim"
        processed_dir = out_base / "processed"
        metadata_dir = out_base / "metadata"

        for d in [raw_dir, interim_dir, processed_dir, metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Stage 1: Ingestion
        if raw_csv_path is None or not Path(raw_csv_path).exists():
            raw_file = raw_dir / "coimbatore_avinashi_corridor_telemetry.csv"
            df_raw = generate_empirical_pilot_corridor_telemetry(output_csv_path=raw_file)
        else:
            df_raw = load_raw_traffic_csv(raw_csv_path)

        # Stage 2: Quality Audit & Validation
        audit_report = self.validator.audit_dataset(df_raw)

        # Stage 3: Cleaning & Deduplication
        df_clean, cleaning_report = self.cleaner.clean_observations(df_raw)
        df_clean.to_csv(interim_dir / "clean_observations.csv", index=False)

        # Stage 4: Temporal Normalization & Time Bucketing
        df_temporal, temporal_report = self.temporal_norm.normalize_and_bucket(df_clean)

        # Stage 5: Spatial Road-Network Matching
        df_matched, spatial_report = self.spatial_matcher.match_dataframe(df_temporal)
        df_matched.to_csv(interim_dir / "matched_observations.csv", index=False)

        # Stage 6: Road-Segment Aggregation
        df_agg = self.feature_engineer.aggregate_segment_time_buckets(df_matched)

        # Stage 7: Feature Generation (Strict Zero-Leakage)
        df_model_ready = self.feature_engineer.build_model_ready_features(df_agg)

        # Stage 8: Chronological Train / Val / Test Split
        df_model_ready["time_bucket"] = pd.to_datetime(df_model_ready["time_bucket"], utc=True)
        df_model_ready = df_model_ready.sort_values(by="time_bucket").reset_index(drop=True)

        unique_times = df_model_ready["time_bucket"].drop_duplicates().sort_values().reset_index(drop=True)
        n_times = len(unique_times)
        train_cutoff_idx = int(n_times * self.train_ratio)
        val_cutoff_idx = int(n_times * (self.train_ratio + self.val_ratio))

        train_cutoff_time = unique_times.iloc[train_cutoff_idx]
        val_cutoff_time = unique_times.iloc[val_cutoff_idx]

        train_df = df_model_ready[df_model_ready["time_bucket"] < train_cutoff_time].copy()
        val_df = df_model_ready[
            (df_model_ready["time_bucket"] >= train_cutoff_time) & (df_model_ready["time_bucket"] < val_cutoff_time)
        ].copy()
        test_df = df_model_ready[df_model_ready["time_bucket"] >= val_cutoff_time].copy()

        train_period = (
            train_df["time_bucket"].min().isoformat() if len(train_df) > 0 else "",
            train_df["time_bucket"].max().isoformat() if len(train_df) > 0 else "",
        )
        val_period = (
            val_df["time_bucket"].min().isoformat() if len(val_df) > 0 else "",
            val_df["time_bucket"].max().isoformat() if len(val_df) > 0 else "",
        )
        test_period = (
            test_df["time_bucket"].min().isoformat() if len(test_df) > 0 else "",
            test_df["time_bucket"].max().isoformat() if len(test_df) > 0 else "",
        )

        # Stage 9: Serialization (Parquet & CSV)
        df_model_ready.to_parquet(processed_dir / "model_ready_traffic_dataset.parquet", index=False)
        df_model_ready.to_csv(processed_dir / "model_ready_traffic_dataset.csv", index=False)

        train_df.to_parquet(processed_dir / "train_split.parquet", index=False)
        val_df.to_parquet(processed_dir / "val_split.parquet", index=False)
        test_df.to_parquet(processed_dir / "test_split.parquet", index=False)

        # Stage 10: Metadata & Provenance Ledger
        provenance = TrafficDatasetProvenance(
            dataset_name="Coimbatore_Avinashi_NH544_Model_Ready_Traffic",
            provider="TrafficRoute AI Research / Transport Telemetry Repository",
            source_url="https://trafficroute.ai/data/coimbatore_avinashi_telemetry.csv",
            citation="IRC:106-1990; IIT Madras Heterogeneous Highway Speed Studies",
            license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            collection_methodology="Multi-probe GPS floating car telemetry & corridor velocity logging",
            geographic_coverage="Coimbatore Metropolitan Area & Coimbatore <-> Avinashi (NH-544) Corridor",
            bounding_box={"north": 11.23, "south": 10.94, "east": 77.30, "west": 76.90},
            temporal_coverage_start=temporal_report.earliest_timestamp,
            temporal_coverage_end=temporal_report.latest_timestamp,
            sampling_interval_seconds=temporal_report.expected_sampling_interval_s,
            raw_record_count=audit_report.total_records,
            clean_record_count=cleaning_report.total_retained,
            matched_record_count=spatial_report.matched_observations,
            aggregated_segment_record_count=len(df_model_ready),
            preprocessing_version="1.0.0",
            extra_attributes={
                "features_generated": list(df_model_ready.columns),
                "unique_segments": int(df_model_ready["segment_id"].nunique()),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "train_period": train_period,
                "val_period": val_period,
                "test_period": test_period,
            },
        )
        provenance.save_json(metadata_dir / "coimbatore_avinashi_processed_provenance.json")

        summary = PipelineExecutionSummary(
            audit=audit_report,
            cleaning=cleaning_report,
            temporal=temporal_report,
            spatial=spatial_report,
            model_ready_rows=len(df_model_ready),
            unique_segments=int(df_model_ready["segment_id"].nunique()),
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            train_period=train_period,
            val_period=val_period,
            test_period=test_period,
        )

        return df_model_ready, summary
