"""Alternative route generation and route diversity enforcement for TrafficRoute AI."""

from datetime import datetime
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import networkx as nx
from pydantic import ValidationError

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RouteSegment, RoutingEngine
from src.network.snapping import NetworkSpatialIndexer, SnapResult


def compute_route_overlap(route_a: Route, route_b: Route) -> float:
    """
    Compute length-weighted shared edge overlap ratio between two routes.
    Returns value between 0.0 (completely distinct) and 1.0 (identical).
    """
    if route_a.total_distance_m <= 0 or route_b.total_distance_m <= 0:
        return 1.0

    edges_a = {(seg.u, seg.v): seg.length_m for seg in route_a.edge_sequence}
    edges_b = {(seg.u, seg.v): seg.length_m for seg in route_b.edge_sequence}

    shared_length = 0.0
    for edge, length in edges_a.items():
        if edge in edges_b:
            shared_length += length

    min_total_len = min(route_a.total_distance_m, route_b.total_distance_m)
    return shared_length / max(min_total_len, 1.0)


class AlternativeRouteGenerator:
    """
    Generates meaningful, distinct alternative routes between origin and destination,
    preventing duplicate or trivial residential variations.
    """

    def __init__(
        self,
        routing_engine: RoutingEngine,
        max_shared_ratio: float = 0.75,
        penalty_factor: float = 0.40,
    ):
        self.engine = routing_engine
        self.max_shared_ratio = (
            max_shared_ratio
            if isinstance(max_shared_ratio, (int, float))
            and not isinstance(max_shared_ratio, bool)
            and math.isfinite(max_shared_ratio)
            and 0.0 <= max_shared_ratio <= 1.0
            else 0.75
        )
        self.penalty_factor = (
            penalty_factor
            if isinstance(penalty_factor, (int, float))
            and not isinstance(penalty_factor, bool)
            and math.isfinite(penalty_factor)
            and penalty_factor >= 0.0
            else 0.40
        )

    @staticmethod
    def _generate_descriptive_name(route: Route, rank: int) -> str:
        """Derive an intuitive human-readable route name from prominent highways."""
        highways = []
        for seg in route.edge_sequence:
            if seg.road_reference and seg.road_reference not in highways:
                highways.append(seg.road_reference)
            elif seg.name and seg.name not in ("Unnamed Road", "unknown", None) and seg.name not in highways:
                if seg.highway_type in ("trunk", "primary", "secondary", "motorway"):
                    highways.append(seg.name)

        if highways:
            primary_corridor = highways[0]
            if len(highways) > 1:
                return f"Via {primary_corridor} & {highways[1]}"
            return f"Via {primary_corridor}"

        if rank == 1:
            return "Primary Fastest Route"
        return f"Alternative Route {rank}"

    def _find_diverse_routes(
        self,
        origin_snap: SnapResult,
        dest_snap: SnapResult,
        u_start: int,
        v_target: int,
        k_routes: int,
        norm_metric: CostMetric,
        departure_time: Optional[datetime],
    ) -> List[Route]:
        """Core iterative penalization search across directed graphs."""
        is_multi = self.engine.road_graph.graph.is_multigraph()
        selected_routes: List[Route] = []
        edge_usage_counts: Dict[Tuple[int, int], int] = {}

        def get_penalized_edge_cost(u: int, v: int, edge_attrs: Dict[str, Any]) -> Optional[float]:
            base_cost = self.engine.get_edge_weight(u, v, edge_attrs, metric=norm_metric, departure_time=departure_time)
            if base_cost is None or not math.isfinite(base_cost) or base_cost < 0.0:
                return None
            times_used = edge_usage_counts.get((u, v), 0)
            if times_used > 0:
                multiplier = 1.0 + (self.penalty_factor * times_used)
                return base_cost * multiplier
            return base_cost

        def penalized_weight_fn(u: int, v: int, d: Any) -> Optional[float]:
            if is_multi and isinstance(d, dict):
                best_cost = None
                for k_idx, edge_attrs in d.items():
                    if not isinstance(edge_attrs, dict):
                        continue
                    cost = get_penalized_edge_cost(u, v, edge_attrs)
                    if cost is not None:
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                return best_cost
            if isinstance(d, dict):
                return get_penalized_edge_cost(u, v, d)
            return None

        # Explore diverse corridors iteratively
        max_iterations = max(10, k_routes * 5)
        for _ in range(max_iterations):
            if len(selected_routes) >= k_routes:
                break

            try:
                node_path = nx.shortest_path(
                    self.engine.road_graph.graph,
                    source=u_start,
                    target=v_target,
                    weight=penalized_weight_fn,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                break

            # Assemble segments using the hardened RoutingEngine assembly
            assembled = self.engine._assemble_route_segments(node_path, norm_metric, departure_time)
            if assembled is None:
                # Corrupt edge encountered; penalize this path so alternatives can be explored
                for i in range(len(node_path) - 1):
                    pair = (node_path[i], node_path[i + 1])
                    edge_usage_counts[pair] = edge_usage_counts.get(pair, 0) + 1
                continue

            edge_segments, total_dist_m, total_time_s = assembled

            try:
                candidate_route = Route(
                    route_id=f"route_{len(selected_routes) + 1}",
                    name=f"Candidate {len(selected_routes) + 1}",
                    origin_snap=origin_snap,
                    destination_snap=dest_snap,
                    node_sequence=node_path,
                    edge_sequence=edge_segments,
                    total_distance_m=round(total_dist_m, 2),
                    total_time_s=round(total_time_s, 2),
                    cost_metric=norm_metric,
                    departure_time=departure_time,
                )
            except (ValueError, ValidationError):
                for i in range(len(node_path) - 1):
                    pair = (node_path[i], node_path[i + 1])
                    edge_usage_counts[pair] = edge_usage_counts.get(pair, 0) + 1
                continue

            # Diversity check against all already selected routes
            is_diverse = True
            for accepted in selected_routes:
                overlap = compute_route_overlap(candidate_route, accepted)
                if overlap > self.max_shared_ratio:
                    is_diverse = False
                    break

            if is_diverse or len(selected_routes) == 0:
                rank = len(selected_routes) + 1
                candidate_route.name = self._generate_descriptive_name(candidate_route, rank)
                candidate_route.route_id = f"route_{rank}"
                selected_routes.append(candidate_route)

            # Increment edge usage for penalty in next iteration
            for seg in edge_segments:
                edge_usage_counts[(seg.u, seg.v)] = edge_usage_counts.get((seg.u, seg.v), 0) + 1

        # Sort selected routes by primary cost (distance or time)
        if norm_metric == CostMetric.DISTANCE:
            selected_routes.sort(key=lambda r: (r.total_distance_m, r.total_time_s))
        else:
            selected_routes.sort(key=lambda r: (r.total_time_s, r.total_distance_m))

        for idx, r in enumerate(selected_routes):
            r.route_id = f"route_{idx + 1}"
            r.name = self._generate_descriptive_name(r, idx + 1)

        return selected_routes

    def generate_alternative_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        k_routes: int = 3,
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
    ) -> List[Route]:
        """
        Generate up to k distinct, diverse routes between origin and destination coordinates.

        Args:
            origin_lat: Origin latitude in WGS84.
            origin_lon: Origin longitude in WGS84.
            destination_lat: Destination latitude in WGS84.
            destination_lon: Destination longitude in WGS84.
            k_routes: Maximum number of alternative routes to return.
            metric: Cost optimization metric.
            departure_time: Optional departure timestamp.

        Returns:
            List of diverse Route objects sorted by estimated duration/distance.
        """
        if isinstance(k_routes, bool) or not isinstance(k_routes, int) or k_routes <= 0:
            return []

        norm_metric = self.engine._normalize_metric(metric)
        if norm_metric is None:
            return []

        if not self.engine._is_valid_coordinate(origin_lat, origin_lon) or not self.engine._is_valid_coordinate(
            destination_lat, destination_lon
        ):
            return []

        origin_snap = self.engine.indexer.snap_coordinate(origin_lat, origin_lon)
        dest_snap = self.engine.indexer.snap_coordinate(destination_lat, destination_lon)

        if not origin_snap.is_valid or not dest_snap.is_valid:
            return []

        u_start = origin_snap.nearest_node_id
        v_target = dest_snap.nearest_node_id

        if u_start == v_target:
            primary = self.engine.calculate_route(
                origin_lat, origin_lon, destination_lat, destination_lon, metric=norm_metric, departure_time=departure_time
            )
            return [primary] if primary is not None else []

        return self._find_diverse_routes(
            origin_snap=origin_snap,
            dest_snap=dest_snap,
            u_start=u_start,
            v_target=v_target,
            k_routes=k_routes,
            norm_metric=norm_metric,
            departure_time=departure_time,
        )

    def generate_alternative_routes_by_nodes(
        self,
        origin_node_id: int,
        destination_node_id: int,
        k_routes: int = 3,
        metric: CostMetric = CostMetric.FREE_FLOW_TIME,
        departure_time: Optional[datetime] = None,
    ) -> List[Route]:
        """
        Generate up to k distinct, diverse routes directly between existing graph node IDs.

        Args:
            origin_node_id: Origin graph node ID.
            destination_node_id: Destination graph node ID.
            k_routes: Maximum number of alternative routes to return.
            metric: Cost optimization metric.
            departure_time: Optional departure timestamp.

        Returns:
            List of diverse Route objects sorted by estimated duration/distance.
        """
        if isinstance(k_routes, bool) or not isinstance(k_routes, int) or k_routes <= 0:
            return []

        norm_metric = self.engine._normalize_metric(metric)
        if norm_metric is None:
            return []

        if (
            isinstance(origin_node_id, bool)
            or not isinstance(origin_node_id, int)
            or isinstance(destination_node_id, bool)
            or not isinstance(destination_node_id, int)
        ):
            return []

        if (
            origin_node_id not in self.engine.road_graph.graph
            or destination_node_id not in self.engine.road_graph.graph
        ):
            return []

        origin_snap = self.engine._node_to_snap_result(origin_node_id)
        dest_snap = self.engine._node_to_snap_result(destination_node_id)

        if not origin_snap.is_valid or not dest_snap.is_valid:
            return []

        if origin_node_id == destination_node_id:
            primary = self.engine.calculate_route_by_nodes(
                origin_node_id, destination_node_id, metric=norm_metric, departure_time=departure_time
            )
            return [primary] if primary is not None else []

        return self._find_diverse_routes(
            origin_snap=origin_snap,
            dest_snap=dest_snap,
            u_start=origin_node_id,
            v_target=destination_node_id,
            k_routes=k_routes,
            norm_metric=norm_metric,
            departure_time=departure_time,
        )
