"""Data provenance and lineage metadata for traffic datasets in TrafficRoute AI."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrafficDatasetProvenance(BaseModel):
    """Metadata tracking raw traffic dataset origin, spatial bounds, temporal scope, and processing lineage."""

    dataset_name: str = Field(..., description="Name of the traffic dataset")
    provider: str = Field(..., description="Data provider or academic institution")
    source_url: str = Field(default="", description="Download or repository URL")
    citation: Optional[str] = Field(default=None, description="Academic citation if applicable")
    license: str = Field(..., description="License terms governing data usage")
    collection_methodology: str = Field(
        ..., description="How data was gathered (e.g., GPS floating car probe, UAV video trajectory, inductive loop)"
    )
    geographic_coverage: str = Field(..., description="Corridor or city covered (e.g., Coimbatore-Avinashi NH-544 / Chennai OMR)")
    bounding_box: Dict[str, float] = Field(..., description="Geographic bounding box (north, south, east, west)")
    temporal_coverage_start: str = Field(..., description="ISO 8601 timestamp of earliest observation")
    temporal_coverage_end: str = Field(..., description="ISO 8601 timestamp of latest observation")
    sampling_interval_seconds: float = Field(..., description="Expected sampling interval in seconds")
    raw_record_count: int = Field(default=0, description="Total raw observations ingested")
    clean_record_count: int = Field(default=0, description="Total clean observations retained")
    matched_record_count: int = Field(default=0, description="Total observations matched to P1 road graph")
    aggregated_segment_record_count: int = Field(default=0, description="Total aggregated segment time-bucket rows")
    preprocessing_version: str = Field(default="1.0.0", description="Pipeline code version")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of pipeline execution",
    )
    extra_attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance to dictionary."""
        return self.model_dump()

    def save_json(self, file_path: Path | str) -> None:
        """Save provenance metadata to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: Path | str) -> "TrafficDatasetProvenance":
        """Load provenance metadata from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
