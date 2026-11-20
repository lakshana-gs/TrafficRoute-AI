"""Routing engine abstraction with pluggable cost evaluation for TrafficRoute AI."""

from datetime import datetime
from enum import Enum
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import networkx as nx
from pydantic import BaseModel, Field, field_validator, model_validator

from src.network.graph import RoadNetworkGraph
from src.network.snapping import NetworkSpatialIndexer, SnapResult


class CostMetric(str, Enum):
    """Supported edge cost evaluation metrics."""

    DISTANCE = "distance"
    FREE_FLOW_TIME = "free_flow_time"
    PREDICTED_TIME = "predicted_time"  # To be connected with ML speed providers in P5/P6


class RouteSegment(BaseModel):
    """Detailed metadata for a single traversed edge in a route."""

    u: int = Field(..., description="Origin node ID")
    v: int = Field(..., description="Target node ID")
    key: int = Field(default=0, description="Edge multigraph key")
    edge_id: str = Field(..., description="Unique edge identifier")
    name: Optional[str] = Field(default=None, description="Road name")
    road_reference: Optional[str] = Field(default=None, description="Highway code (e.g., NH-544)")
    highway_type: str = Field(..., description="Road classification")
    length_m: float = Field(..., gt=0.0, description="Segment length in meters")
    speed_kmh: float = Field(..., gt=0.0, description="Evaluated speed in km/h")
    traversal_time_s: float = Field(..., ge=0.0, description="Segment traversal time in seconds")

    @field_validator("length_m", mode="before")
    @classmethod
    def validate_length_m(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"length_m must be a finite, strictly positive number, got {v}")
        return float(v)

    @field_validator("speed_kmh", mode="before")
    @classmethod
    def validate_speed_kmh(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"speed_kmh must be a finite, strictly positive number, got {v}")
        return float(v)

    @field_validator("traversal_time_s", mode="before")
    @classmethod
    def validate_traversal_time_s(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
            raise ValueError(f"traversal_time_s must be a finite, non-negative number, got {v}")
        return float(v)


class Route(BaseModel):
    """Complete computed route between an origin and destination."""

    route_id: str = Field(..., description="Unique route identifier")
    name: str = Field(default="Main Route", description="Descriptive route label (e.g. Via NH-544)")
    origin_snap: SnapResult = Field(..., description="Snapping details for origin")
    destination_snap: SnapResult = Field(..., description="Snapping details for destination")
    node_sequence: List[int] = Field(..., min_length=1, description="Ordered sequence of traversed graph node IDs")
    edge_sequence: List[RouteSegment] = Field(..., description="Ordered sequence of traversed road segments")
    total_distance_m: float = Field(..., ge=0.0, description="Total route length in meters")
    total_time_s: float = Field(..., ge=0.0, description="Total route travel duration in seconds")
    cost_metric: CostMetric = Field(..., description="Cost metric used for route optimization")
    departure_time: Optional[datetime] = Field(default=None, description="Target departure timestamp")

    @field_validator("total_distance_m", mode="before")
    @classmethod
    def validate_total_distance_m(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
            raise ValueError(f"total_distance_m must be a finite, non-negative number, got {v}")
        return float(v)

    @field_validator("total_time_s", mode="before")
    @classmethod
    def validate_total_time_s(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
            raise ValueError(f"total_time_s must be a finite, non-negative number, got {v}")
        return float(v)

    @field_validator("node_sequence", mode="before")
    @classmethod
    def validate_node_sequence(cls, v: Any) -> List[int]:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("node_sequence must contain at least one node ID")
        for node in v:
            if isinstance(node, bool) or not isinstance(node, int):
                raise ValueError(f"Node IDs in node_sequence must be integers, got {node}")
        return v

    @model_validator(mode="after")
    def validate_route_invariants(self) -> "Route":
        expected_edge_count = len(self.node_sequence) - 1
        if len(self.edge_sequence) != expected_edge_count:
            raise ValueError(
                f"edge_sequence length ({len(self.edge_sequence)}) must equal len(node_sequence) - 1 ({expected_edge_count})"
            )

        # Single-node route (origin == destination)
        if len(self.node_sequence) == 1:
            if self.total_distance_m != 0.0 or self.total_time_s != 0.0:
                raise ValueError(
                    f"Single-node route must have zero distance and time, got dist={self.total_distance_m}, time={self.total_time_s}"
                )
            return self

        # Validate topological segment/node connectivity
        for i, seg in enumerate(self.edge_sequence):
            expected_u = self.node_sequence[i]
            expected_v = self.node_sequence[i + 1]
            if seg.u != expected_u or seg.v != expected_v:
                raise ValueError(
                    f"Segment at index {i} connects {seg.u} -> {seg.v}, but node_sequence expects {expected_u} -> {expected_v}"
                )

        # Validate totals against segment sums with floating-point tolerance
        # Tolerance accommodates per-segment rounding (e.g. 0.02 per segment + 1.0 baseline)
        tol = max(1.0, 0.02 * len(self.edge_sequence))

        seg_dist_sum = sum(seg.length_m for seg in self.edge_sequence)
        if abs(self.total_distance_m - seg_dist_sum) > tol:
            raise ValueError(
                f"total_distance_m ({self.total_distance_m}) deviates from sum of segment lengths ({seg_dist_sum:.2f}) beyond tolerance ({tol:.2f})"
            )

        # Total time is validated against segment traversal times for standard distance and free-flow metrics
        if self.cost_metric in (CostMetric.DISTANCE, CostMetric.FREE_FLOW_TIME):
            seg_time_sum = sum(seg.traversal_time_s for seg in self.edge_sequence)
            if abs(self.total_time_s - seg_time_sum) > tol:
                raise ValueError(
                    f"total_time_s ({self.total_time_s}) deviates from sum of segment traversal times ({seg_time_sum:.2f}) beyond tolerance ({tol:.2f})"
                )

        return self

    @property
    def total_distance_km(self) -> float:
        """Route distance in kilometers."""
        return round(self.total_distance_m / 1000.0, 2)

    @property
    def total_time_minutes(self) -> float:
        """Route travel duration in minutes."""
        return round(self.total_time_s / 60.0, 1)

    @property
    def segment_count(self) -> int:
        """Number of traversed road segments."""
        return len(self.edge_sequence)

    @property
    def average_speed_kmh(self) -> float:
        """Average journey speed across the entire route in km/h; returns 0.0 if total duration is 0."""
        if self.total_time_s <= 0.0 or self.total_distance_m <= 0.0:
            return 0.0
        dist_km = self.total_distance_m / 1000.0
        time_hours = self.total_time_s / 3600.0
        return round(dist_km / time_hours, 1)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of key route KPIs."""
        metric_str = self.cost_metric.value if isinstance(self.cost_metric, CostMetric) else str(self.cost_metric)
        dep_str = self.departure_time.isoformat() if self.departure_time is not None else None
        return {
            "route_id": self.route_id,
            "name": self.name,
            "cost_metric": metric_str,
            "total_distance_m": self.total_distance_m,
            "total_distance_km": self.total_distance_km,
            "total_time_s": self.total_time_s,
            "total_time_minutes": self.total_time_minutes,
            "average_speed_kmh": self.average_speed_kmh,
            "segment_count": self.segment_count,
            "node_count": len(self.node_sequence),
            "departure_time": dep_str,
        }


class RouteStatus(str, Enum):
    """Diagnostic outcome status for route planning queries."""

    SUCCESS = "success"
    SAME_LOCATION = "same_location"
    SNAP_FAILED = "snap_failed"
    NO_PATH_FOUND = "no_path_found"
    INVALID_INPUT = "invalid_input"


class RoutePlanResult(BaseModel):
    """Canonical response contract for unified route planning operations."""

    success: bool = Field(..., description="True if a valid route was determined, False otherwise")
    status: RouteStatus = Field(..., description="Diagnostic classification of planning outcome")
    status_message: str = Field(..., description="Human-readable explanation of outcome")
    primary_route: Optional[Route] = Field(default=None, description="Primary recommended route if found")
    alternative_routes: List[Route] = Field(default_factory=list, description="Diverse alternative corridors excluding primary route")
    summary: Optional[Dict[str, Any]] = Field(default=None, description="Convenience KPI summary of primary route")

    @model_validator(mode="after")
    def populate_summary(self) -> "RoutePlanResult":
        """Ensure summary is populated from primary_route if not explicitly provided."""
        if self.summary is None and self.primary_route is not None:
            self.summary = self.primary_route.summary()
        return self


class RoutingEngine:
    """
    Routing engine supporting shortest distance, free-flow time, and extensible dynamic cost routing.
    """

    def __init__(self, road_graph: RoadNetworkGraph, spatial_indexer: Optional[NetworkSpatialIndexer] = None):
        self.road_graph = road_graph
        self.indexer = spatial_indexer or NetworkSpatialIndexer(road_graph)

    @staticmethod
    def _haversine_heuristic(u_node: Dict[str, Any], v_node: Dict[str, Any]) -> float:
        """Admissible A* heuristic: great-circle free-flow time estimate."""
        r = 6371000.0
        lat1 = math.radians(u_node["latitude"])
        lon1 = math.radians(u_node["longitude"])
        lat2 = math.radians(v_node["latitude"])
        lon2 = math.radians(v_node["longitude"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        dist_m = r * 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
        max_possible_speed_mps = 120.0 * (1000.0 / 3600.0)  # 120 km/h upper bound
        return dist_m / max_possible_speed_mps

    @staticmethod
    def _is_valid_coordinate(lat: Any, lon: Any) -> bool:
        """Validate that latitude and longitude are finite numbers within standard WGS84 ranges."""
        if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
            return False
        if isinstance(lat, bool) or isinstance(lon, bool):
            return False
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False
        if not (-90.0 <= lat <= 90.0):
            return False
        if not (-180.0 <= lon <= 180.0):
            return False
        return True

    @staticmethod
    def _normalize_metric(metric: Any) -> Optional[CostMetric]:
        """Validate and normalize cost metric, returning None if invalid or unsupported."""
        if isinstance(metric, CostMetric):
            return metric
        if isinstance(metric, str):
            try:
                return CostMetric(metric)
            except ValueError:
                return None
        return None

    def get_edge_weight(
        self,
        u: int,
        v: int,
        edge_data: Dict[str, Any],
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
        speed_provider: Optional[Any] = None,
    ) -> Optional[float]:
        """
        Compute edge weight based on selected metric.
        Returns positive finite float if edge is valid, or None if edge data is invalid.
        """
        if not isinstance(edge_data, dict):
            return None

        # Validate length_m: strictly positive and finite
        length_m = edge_data.get("length_m")
        if (
            not isinstance(length_m, (int, float))
            or isinstance(length_m, bool)
            or not math.isfinite(length_m)
            or length_m <= 0.0
        ):
            return None

        norm_metric = self._normalize_metric(metric)
        if norm_metric is None:
            return None

        if norm_metric == CostMetric.DISTANCE:
            return float(length_m)

        # Dynamic or Baseline Time Calculation
        if norm_metric == CostMetric.PREDICTED_TIME and speed_provider is not None:
            speed_kmh = speed_provider.get_speed(u, v, edge_data, departure_time)
        else:
            speed_kmh = edge_data.get("free_flow_speed_kmh", 30.0)

        # Validate speed_kmh: strictly positive and finite
        if (
            not isinstance(speed_kmh, (int, float))
            or isinstance(speed_kmh, bool)
            or not math.isfinite(speed_kmh)
            or speed_kmh <= 0.0
        ):
            return None

        speed_mps = float(speed_kmh) * (1000.0 / 3600.0)
        if speed_mps <= 0.0:
            return None

        return float(length_m) / speed_mps

    def _assemble_route_segments(
        self,
        node_path: List[int],
        metric: CostMetric,
        departure_time: Optional[datetime] = None,
    ) -> Optional[Tuple[List[RouteSegment], float, float]]:
        """
        Reconstruct valid traversed edge segments along a node path.
        Returns None if any step contains no valid edge.
        """
        is_multi = self.road_graph.graph.is_multigraph()
        edge_segments: List[RouteSegment] = []
        total_dist_m = 0.0
        total_time_s = 0.0

        for i in range(len(node_path) - 1):
            curr_u = node_path[i]
            curr_v = node_path[i + 1]

            if is_multi:
                edge_dict = self.road_graph.graph[curr_u][curr_v]
                valid_candidates = []
                for k, edge_attrs in edge_dict.items():
                    cost = self.get_edge_weight(curr_u, curr_v, edge_attrs, metric=metric, departure_time=departure_time)
                    if cost is not None:
                        valid_candidates.append((cost, k, edge_attrs))
                if not valid_candidates:
                    return None
                # Select candidate with minimum valid cost
                valid_candidates.sort(key=lambda item: item[0])
                _, best_k, data = valid_candidates[0]
            else:
                cost = self.get_edge_weight(curr_u, curr_v, self.road_graph.graph[curr_u][curr_v], metric=metric, departure_time=departure_time)
                if cost is None:
                    return None
                best_k = 0
                data = self.road_graph.graph[curr_u][curr_v]

            # Validate segment length
            raw_len = data.get("length_m")
            if (
                not isinstance(raw_len, (int, float))
                or isinstance(raw_len, bool)
                or not math.isfinite(raw_len)
                or raw_len <= 0.0
            ):
                return None
            seg_len = float(raw_len)

            # Validate segment speed: use positive finite float or fall back to standardized default 30.0 km/h
            raw_speed = data.get("free_flow_speed_kmh")
            if (
                isinstance(raw_speed, (int, float))
                and not isinstance(raw_speed, bool)
                and math.isfinite(raw_speed)
                and raw_speed > 0.0
            ):
                seg_speed = float(raw_speed)
            else:
                seg_speed = 30.0

            # Validate precomputed free_flow_time_s from graph if present and valid; otherwise compute
            raw_time = data.get("free_flow_time_s")
            if (
                isinstance(raw_time, (int, float))
                and not isinstance(raw_time, bool)
                and math.isfinite(raw_time)
                and raw_time > 0.0
            ):
                seg_time = float(raw_time)
            else:
                speed_mps = seg_speed * (1000.0 / 3600.0)
                seg_time = seg_len / speed_mps

            total_dist_m += seg_len
            total_time_s += seg_time

            k_int = best_k if isinstance(best_k, int) else 0

            raw_name = data.get("name")
            name_str = " / ".join(str(n) for n in raw_name if n) if isinstance(raw_name, list) else (str(raw_name) if raw_name is not None else None)
            raw_ref = data.get("road_reference")
            ref_str = " / ".join(str(r) for r in raw_ref if r) if isinstance(raw_ref, list) else (str(raw_ref) if raw_ref is not None else None)

            edge_segments.append(
                RouteSegment(
                    u=curr_u,
                    v=curr_v,
                    key=k_int,
                    edge_id=str(data.get("edge_id", f"{curr_u}_{curr_v}_{best_k}")),
                    name=name_str,
                    road_reference=ref_str,
                    highway_type=str(data.get("highway_type") or data.get("highway") or "unclassified"),
                    length_m=round(seg_len, 2),
                    speed_kmh=round(seg_speed, 1),
                    traversal_time_s=round(seg_time, 2),
                )
            )

        return edge_segments, total_dist_m, total_time_s

    def calculate_route(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
        route_name: str = "Fastest Route",
        route_id: str = "primary_route",
    ) -> Optional[Route]:
        """
        Compute optimal route between arbitrary origin and destination coordinates.

        Args:
            origin_lat: Origin latitude in WGS84.
            origin_lon: Origin longitude in WGS84.
            destination_lat: Destination latitude in WGS84.
            destination_lon: Destination longitude in WGS84.
            metric: Optimization cost metric (DISTANCE, FREE_FLOW_TIME, PREDICTED_TIME).
            departure_time: Optional planned departure timestamp.
            route_name: Human-readable route label.
            route_id: Identifier for route.

        Returns:
            Route object or None if no path exists or inputs are invalid.
        """
        # Validate metric
        norm_metric = self._normalize_metric(metric)
        if norm_metric is None:
            return None

        # Validate coordinate types, finiteness, and WGS84 bounds
        if not self._is_valid_coordinate(origin_lat, origin_lon) or not self._is_valid_coordinate(destination_lat, destination_lon):
            return None

        origin_snap = self.indexer.snap_coordinate(origin_lat, origin_lon)
        dest_snap = self.indexer.snap_coordinate(destination_lat, destination_lon)

        if not origin_snap.is_valid or not dest_snap.is_valid:
            return None

        u_start = origin_snap.nearest_node_id
        v_target = dest_snap.nearest_node_id

        if u_start == v_target:
            # Origin and destination map to the exact same node
            return Route(
                route_id=route_id,
                name=route_name,
                origin_snap=origin_snap,
                destination_snap=dest_snap,
                node_sequence=[u_start],
                edge_sequence=[],
                total_distance_m=0.0,
                total_time_s=0.0,
                cost_metric=norm_metric,
                departure_time=departure_time,
            )

        is_multi = self.road_graph.graph.is_multigraph()

        # Weight function for NetworkX shortest path
        def weight_fn(u, v, d):
            if is_multi and isinstance(d, dict):
                best_cost = None
                for k_idx, edge_attrs in d.items():
                    cost = self.get_edge_weight(u, v, edge_attrs, metric=norm_metric, departure_time=departure_time)
                    if cost is not None:
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                return best_cost
            if isinstance(d, dict):
                return self.get_edge_weight(u, v, d, metric=norm_metric, departure_time=departure_time)
            return None

        try:
            node_path = nx.shortest_path(self.road_graph.graph, source=u_start, target=v_target, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        # Assemble traversed edge segments
        assembled = self._assemble_route_segments(node_path, norm_metric, departure_time)
        if assembled is None:
            return None
        edge_segments, total_dist_m, total_time_s = assembled

        return Route(
            route_id=route_id,
            name=route_name,
            origin_snap=origin_snap,
            destination_snap=dest_snap,
            node_sequence=node_path,
            edge_sequence=edge_segments,
            total_distance_m=round(total_dist_m, 2),
            total_time_s=round(total_time_s, 2),
            cost_metric=norm_metric,
            departure_time=departure_time,
        )

    def _node_to_snap_result(self, node_id: int) -> SnapResult:
        """Construct a direct SnapResult for an existing graph node without spatial snapping."""
        node_data = self.road_graph.graph.nodes[node_id]
        raw_lat = node_data.get("latitude") if node_data.get("latitude") is not None else node_data.get("y")
        raw_lon = node_data.get("longitude") if node_data.get("longitude") is not None else node_data.get("x")

        if not self._is_valid_coordinate(raw_lat, raw_lon):
            return SnapResult(
                input_latitude=0.0,
                input_longitude=0.0,
                snapped_latitude=0.0,
                snapped_longitude=0.0,
                nearest_node_id=node_id,
                distance_meters=0.0,
                is_valid=False,
                warning=f"Node {node_id} has invalid coordinates",
            )

        lat = float(raw_lat)
        lon = float(raw_lon)
        return SnapResult(
            input_latitude=lat,
            input_longitude=lon,
            snapped_latitude=lat,
            snapped_longitude=lon,
            nearest_node_id=node_id,
            distance_meters=0.0,
            is_valid=True,
        )

    def calculate_route_by_nodes(
        self,
        origin_node_id: int,
        destination_node_id: int,
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
        route_name: str = "Fastest Route",
        route_id: str = "primary_route",
    ) -> Optional[Route]:
        """
        Compute optimal route directly between existing graph node IDs without coordinate snapping.

        Args:
            origin_node_id: Origin graph node ID.
            destination_node_id: Destination graph node ID.
            metric: Optimization cost metric (DISTANCE, FREE_FLOW_TIME).
            departure_time: Optional planned departure timestamp.
            route_name: Human-readable route label.
            route_id: Identifier for route.

        Returns:
            Route object or None if origin/destination is missing, disconnected, or no valid path exists.
        """
        # Validate metric
        norm_metric = self._normalize_metric(metric)
        if norm_metric is None:
            return None

        # Validate node IDs: must be int and strictly not bool
        if (
            not isinstance(origin_node_id, int)
            or isinstance(origin_node_id, bool)
            or not isinstance(destination_node_id, int)
            or isinstance(destination_node_id, bool)
        ):
            return None

        # Node presence check: return None if origin or destination does not exist in graph
        if (
            origin_node_id not in self.road_graph.graph
            or destination_node_id not in self.road_graph.graph
        ):
            return None

        origin_snap = self._node_to_snap_result(origin_node_id)
        dest_snap = self._node_to_snap_result(destination_node_id)
        if not origin_snap.is_valid or not dest_snap.is_valid:
            return None

        # Same existing node: return a zero-distance, zero-time route with one node and no edges
        if origin_node_id == destination_node_id:
            return Route(
                route_id=route_id,
                name=route_name,
                origin_snap=origin_snap,
                destination_snap=dest_snap,
                node_sequence=[origin_node_id],
                edge_sequence=[],
                total_distance_m=0.0,
                total_time_s=0.0,
                cost_metric=norm_metric,
                departure_time=departure_time,
            )

        is_multi = self.road_graph.graph.is_multigraph()

        # Weight function for NetworkX shortest path
        def weight_fn(u, v, d):
            if is_multi and isinstance(d, dict):
                best_cost = None
                for k_idx, edge_attrs in d.items():
                    cost = self.get_edge_weight(u, v, edge_attrs, metric=norm_metric, departure_time=departure_time)
                    if cost is not None:
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                return best_cost
            if isinstance(d, dict):
                return self.get_edge_weight(u, v, d, metric=norm_metric, departure_time=departure_time)
            return None

        try:
            node_path = nx.shortest_path(
                self.road_graph.graph,
                source=origin_node_id,
                target=destination_node_id,
                weight=weight_fn,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        # Assemble traversed edge segments
        assembled = self._assemble_route_segments(node_path, norm_metric, departure_time)
        if assembled is None:
            return None
        edge_segments, total_dist_m, total_time_s = assembled

        return Route(
            route_id=route_id,
            name=route_name,
            origin_snap=origin_snap,
            destination_snap=dest_snap,
            node_sequence=node_path,
            edge_sequence=edge_segments,
            total_distance_m=round(total_dist_m, 2),
            total_time_s=round(total_time_s, 2),
            cost_metric=norm_metric,
            departure_time=departure_time,
        )

    def plan_route(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
        num_alternatives: int = 0,
        route_name: str = "Fastest Route",
        route_id: str = "primary_route",
    ) -> RoutePlanResult:
        """
        Unified route planning entry point for geographic coordinate queries.
        Coordinates snapping, primary route finding, and optional alternative generation.
        """
        if isinstance(num_alternatives, bool) or not isinstance(num_alternatives, int) or num_alternatives < 0:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"num_alternatives must be a non-negative integer, got {num_alternatives}",
            )

        norm_metric = self._normalize_metric(metric)
        if norm_metric is None:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"Invalid or unsupported cost metric: {metric}",
            )

        if not self._is_valid_coordinate(origin_lat, origin_lon):
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"Invalid origin coordinates: ({origin_lat}, {origin_lon})",
            )

        if not self._is_valid_coordinate(destination_lat, destination_lon):
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"Invalid destination coordinates: ({destination_lat}, {destination_lon})",
            )

        origin_snap = self.indexer.snap_coordinate(origin_lat, origin_lon)
        dest_snap = self.indexer.snap_coordinate(destination_lat, destination_lon)

        if not origin_snap.is_valid or not dest_snap.is_valid:
            failed_pt = "origin" if not origin_snap.is_valid else "destination"
            return RoutePlanResult(
                success=False,
                status=RouteStatus.SNAP_FAILED,
                status_message=f"Snapping failed for {failed_pt} coordinate under established spatial policy",
            )

        u_start = origin_snap.nearest_node_id
        v_target = dest_snap.nearest_node_id

        if u_start == v_target:
            primary = Route(
                route_id=route_id,
                name=route_name,
                origin_snap=origin_snap,
                destination_snap=dest_snap,
                node_sequence=[u_start],
                edge_sequence=[],
                total_distance_m=0.0,
                total_time_s=0.0,
                cost_metric=norm_metric,
                departure_time=departure_time,
            )
            return RoutePlanResult(
                success=True,
                status=RouteStatus.SAME_LOCATION,
                status_message="Origin and destination snapped to the same graph node; zero-distance route returned",
                primary_route=primary,
                alternative_routes=[],
            )

        primary = self.calculate_route(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
            metric=norm_metric,
            departure_time=departure_time,
            route_name=route_name,
            route_id=route_id,
        )

        if primary is None:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.NO_PATH_FOUND,
                status_message="No viable path found between origin and destination coordinates",
            )

        alternatives: List[Route] = []
        if num_alternatives > 0:
            from src.network.alternatives import AlternativeRouteGenerator

            alt_gen = AlternativeRouteGenerator(self)
            all_diverse = alt_gen.generate_alternative_routes(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                destination_lat=destination_lat,
                destination_lon=destination_lon,
                k_routes=num_alternatives + 1,
                metric=norm_metric,
                departure_time=departure_time,
            )
            for r in all_diverse:
                if r.node_sequence != primary.node_sequence:
                    alternatives.append(r)
                    if len(alternatives) >= num_alternatives:
                        break

        return RoutePlanResult(
            success=True,
            status=RouteStatus.SUCCESS,
            status_message="Route planning completed successfully",
            primary_route=primary,
            alternative_routes=alternatives,
        )

    def plan_route_by_nodes(
        self,
        origin_node_id: int,
        destination_node_id: int,
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
        num_alternatives: int = 0,
        route_name: str = "Fastest Route",
        route_id: str = "primary_route",
    ) -> RoutePlanResult:
        """
        Unified route planning entry point for direct graph node ID queries.
        Coordinates node verification, primary route finding, and optional alternative generation.
        """
        if isinstance(num_alternatives, bool) or not isinstance(num_alternatives, int) or num_alternatives < 0:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"num_alternatives must be a non-negative integer, got {num_alternatives}",
            )

        norm_metric = self._normalize_metric(metric)
        if norm_metric is None:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"Invalid or unsupported cost metric: {metric}",
            )

        if (
            isinstance(origin_node_id, bool)
            or not isinstance(origin_node_id, int)
            or isinstance(destination_node_id, bool)
            or not isinstance(destination_node_id, int)
        ):
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message="Node IDs must be non-boolean integers",
            )

        if origin_node_id not in self.road_graph.graph or destination_node_id not in self.road_graph.graph:
            missing = origin_node_id if origin_node_id not in self.road_graph.graph else destination_node_id
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message=f"Node ID {missing} does not exist in road network graph",
            )

        origin_snap = self._node_to_snap_result(origin_node_id)
        dest_snap = self._node_to_snap_result(destination_node_id)

        if not origin_snap.is_valid or not dest_snap.is_valid:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.INVALID_INPUT,
                status_message="Node coordinates are non-finite or outside valid bounds",
            )

        if origin_node_id == destination_node_id:
            primary = Route(
                route_id=route_id,
                name=route_name,
                origin_snap=origin_snap,
                destination_snap=dest_snap,
                node_sequence=[origin_node_id],
                edge_sequence=[],
                total_distance_m=0.0,
                total_time_s=0.0,
                cost_metric=norm_metric,
                departure_time=departure_time,
            )
            return RoutePlanResult(
                success=True,
                status=RouteStatus.SAME_LOCATION,
                status_message="Origin and destination are the exact same node; zero-distance route returned",
                primary_route=primary,
                alternative_routes=[],
            )

        primary = self.calculate_route_by_nodes(
            origin_node_id=origin_node_id,
            destination_node_id=destination_node_id,
            metric=norm_metric,
            departure_time=departure_time,
            route_name=route_name,
            route_id=route_id,
        )

        if primary is None:
            return RoutePlanResult(
                success=False,
                status=RouteStatus.NO_PATH_FOUND,
                status_message="No viable path found between origin and destination nodes",
            )

        alternatives: List[Route] = []
        if num_alternatives > 0:
            from src.network.alternatives import AlternativeRouteGenerator

            alt_gen = AlternativeRouteGenerator(self)
            all_diverse = alt_gen.generate_alternative_routes_by_nodes(
                origin_node_id=origin_node_id,
                destination_node_id=destination_node_id,
                k_routes=num_alternatives + 1,
                metric=norm_metric,
                departure_time=departure_time,
            )
            for r in all_diverse:
                if r.node_sequence != primary.node_sequence:
                    alternatives.append(r)
                    if len(alternatives) >= num_alternatives:
                        break

        return RoutePlanResult(
            success=True,
            status=RouteStatus.SUCCESS,
            status_message="Route planning completed successfully",
            primary_route=primary,
            alternative_routes=alternatives,
        )
