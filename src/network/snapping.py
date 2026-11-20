"""Location-to-road-network snapping and spatial indexing for TrafficRoute AI."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point

from src.network.graph import RoadNetworkGraph


class SnapResult(BaseModel):
    """Result of snapping geographic coordinates onto the road network."""

    input_latitude: float = Field(..., description="Original query latitude")
    input_longitude: float = Field(..., description="Original query longitude")
    snapped_latitude: float = Field(..., description="Snapped point latitude on road")
    snapped_longitude: float = Field(..., description="Snapped point longitude on road")
    nearest_node_id: int = Field(..., description="Nearest topological graph node ID")
    nearest_edge_id: Optional[str] = Field(default=None, description="Nearest edge identifier if matched")
    nearest_edge_u: Optional[int] = Field(default=None, description="Edge origin node ID")
    nearest_edge_v: Optional[int] = Field(default=None, description="Edge target node ID")
    nearest_edge_key: Optional[int] = Field(default=0, description="Edge key index")
    road_name: Optional[str] = Field(default=None, description="Name of the snapped road")
    road_reference: Optional[str] = Field(default=None, description="Highway reference (e.g., NH-544)")
    highway_type: Optional[str] = Field(default=None, description="Road classification")
    distance_meters: float = Field(..., description="Perpendicular / straight-line snap distance in meters")
    is_valid: bool = Field(..., description="Whether snap distance is within max allowed threshold")
    warning: Optional[str] = Field(default=None, description="Warning if snap distance is high")


class NetworkSpatialIndexer:
    """
    High-performance spatial indexer using KD-Tree in projected radian space for O(log N) node queries,
    with local geometry edge-snapping.
    """

    def __init__(self, road_graph: RoadNetworkGraph, max_snap_distance_m: float = 1500.0):
        self.road_graph = road_graph
        self.max_snap_distance_m = max_snap_distance_m
        self.nodes_list: List[int] = []
        self.coords_radians: np.ndarray = np.empty((0, 2))
        self.kdtree: Optional[KDTree] = None
        self._build_node_index()

    def _build_node_index(self) -> None:
        """Construct KD-Tree over all graph nodes in radian space."""
        nodes = []
        coords = []
        for nid, data in self.road_graph.graph.nodes(data=True):
            lat = data.get("latitude", data.get("y"))
            lon = data.get("longitude", data.get("x"))
            if lat is not None and lon is not None:
                nodes.append(int(nid))
                coords.append([math.radians(float(lat)), math.radians(float(lon))])

        if nodes:
            self.nodes_list = nodes
            self.coords_radians = np.array(coords)
            self.kdtree = KDTree(self.coords_radians)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in meters between two WGS84 points."""
        r = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2.0) ** 2
        )
        return r * 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    def snap_coordinate(
        self,
        latitude: float,
        longitude: float,
        snap_to_edge: bool = True,
    ) -> SnapResult:
        """
        Snap an arbitrary latitude/longitude coordinate to the nearest road network node and edge.

        Args:
            latitude: Query latitude in WGS84 degrees.
            longitude: Query longitude in WGS84 degrees.
            snap_to_edge: If True, searches adjacent edges for nearest perpendicular point.

        Returns:
            SnapResult object containing snapped coordinates, node/edge IDs, and snap distance.
        """
        if not self.kdtree or not self.nodes_list:
            raise RuntimeError("Spatial indexer is empty. Graph has no nodes.")

        query_rad = np.array([math.radians(latitude), math.radians(longitude)])
        # Query nearest 5 candidate nodes from KDTree
        k_candidates = min(10, len(self.nodes_list))
        distances, indices = self.kdtree.query(query_rad, k=k_candidates)

        if np.isscalar(indices):
            indices = [indices]
            distances = [distances]

        nearest_node_id = self.nodes_list[indices[0]]
        nearest_node_data = self.road_graph.graph.nodes[nearest_node_id]
        best_lat = nearest_node_data["latitude"]
        best_lon = nearest_node_data["longitude"]
        min_dist_m = self._haversine_distance(latitude, longitude, best_lat, best_lon)

        matched_edge: Optional[Tuple[int, int, int, Dict[str, Any]]] = None

        if snap_to_edge:
            query_pt = Point(longitude, latitude)
            # Inspect all outgoing and incoming edges for candidate nodes
            candidate_node_ids = [self.nodes_list[idx] for idx in indices]
            candidate_edges = set()
            for cand_id in candidate_node_ids:
                for u, v, k in self.road_graph.graph.out_edges(cand_id, keys=True):
                    candidate_edges.add((u, v, k))
                for u, v, k in self.road_graph.graph.in_edges(cand_id, keys=True):
                    candidate_edges.add((u, v, k))

            for u, v, k in candidate_edges:
                edge_data = self.road_graph.graph.edges[u, v, k]
                geom = edge_data.get("geometry")
                if isinstance(geom, LineString):
                    # Project point onto LineString
                    proj_dist = geom.project(query_pt)
                    proj_pt = geom.interpolate(proj_dist)
                    d_m = self._haversine_distance(latitude, longitude, proj_pt.y, proj_pt.x)
                    if d_m < min_dist_m:
                        min_dist_m = d_m
                        best_lat = proj_pt.y
                        best_lon = proj_pt.x
                        matched_edge = (u, v, k, edge_data)

        # Fallback to nearest node's primary outgoing edge if no perpendicular edge found
        if matched_edge is None:
            out_edges = list(self.road_graph.graph.out_edges(nearest_node_id, keys=True, data=True))
            if out_edges:
                matched_edge = (out_edges[0][0], out_edges[0][1], out_edges[0][2], out_edges[0][3])
            else:
                in_edges = list(self.road_graph.graph.in_edges(nearest_node_id, keys=True, data=True))
                if in_edges:
                    matched_edge = (in_edges[0][0], in_edges[0][1], in_edges[0][2], in_edges[0][3])

        is_valid = min_dist_m <= self.max_snap_distance_m
        warning = None
        if not is_valid:
            warning = f"Snap distance ({min_dist_m:.1f}m) exceeds maximum threshold ({self.max_snap_distance_m}m)."
        elif min_dist_m > 300.0:
            warning = f"Moderate snap distance ({min_dist_m:.1f}m). Location is set back from main road."

        edge_id = matched_edge[3].get("edge_id") if matched_edge else None
        edge_u = matched_edge[0] if matched_edge else None
        edge_v = matched_edge[1] if matched_edge else None
        edge_k = matched_edge[2] if matched_edge else 0
        road_name = matched_edge[3].get("name") if matched_edge else None
        road_ref = matched_edge[3].get("road_reference") if matched_edge else None
        hw_type = matched_edge[3].get("highway_type") if matched_edge else None

        return SnapResult(
            input_latitude=latitude,
            input_longitude=longitude,
            snapped_latitude=best_lat,
            snapped_longitude=best_lon,
            nearest_node_id=nearest_node_id,
            nearest_edge_id=edge_id,
            nearest_edge_u=edge_u,
            nearest_edge_v=edge_v,
            nearest_edge_key=edge_k,
            road_name=road_name,
            road_reference=road_ref,
            highway_type=hw_type,
            distance_meters=round(min_dist_m, 2),
            is_valid=is_valid,
            warning=warning,
        )
