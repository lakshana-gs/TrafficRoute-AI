"""Data provenance tracking and metadata management for TrafficRoute AI."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class NetworkProvenance(BaseModel):
    """Metadata detailing the origin, extraction, and licensing of network data."""

    dataset_name: str = Field(..., description="Human-readable name of the dataset")
    source: str = Field(default="OpenStreetMap", description="Originating data provider")
    source_url: str = Field(default="https://www.openstreetmap.org", description="Source URL")
    license: str = Field(
        default="Open Database License (ODbL) 1.0",
        description="License under which data is provided",
    )
    license_url: str = Field(
        default="https://opendatacommons.org/licenses/odbl/1-0/",
        description="License terms URL",
    )
    extraction_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of data extraction in ISO 8601 format",
    )
    extraction_method: str = Field(
        ...,
        description="Method/tool used to extract network (e.g. OSMnx bounding-box query, Geofabrik PBF)",
    )
    geographic_boundary: Dict[str, float] = Field(
        ...,
        description="Bounding box coordinates (north, south, east, west) in EPSG:4326",
    )
    osm_filter: str = Field(
        default='["highway"]["area"!~"yes"]["highway"!~"abandoned|construction|proposed|raceway"]',
        description="Custom OSM filter applied during extraction",
    )
    preprocessing_version: str = Field(
        default="1.0.0", description="Version of the preprocessing pipeline"
    )
    node_count: Optional[int] = Field(default=None, description="Total nodes in processed graph")
    edge_count: Optional[int] = Field(default=None, description="Total directed edges in processed graph")
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context or properties"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance object to standard dictionary."""
        return self.model_dump()

    def save_json(self, file_path: Path | str) -> None:
        """Save provenance metadata to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: Path | str) -> "NetworkProvenance":
        """Load provenance metadata from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
