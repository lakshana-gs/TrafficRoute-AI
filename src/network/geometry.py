"""Route geometry construction and GeoJSON serialization for TrafficRoute AI."""

from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import LineString, Point, mapping

from src.network.graph import RoadNetworkGraph
from src.network.routing import Route, RouteSegment


class RouteGeometryBuilder:
    """
    Constructs high-precision, map-ready coordinate sequences and GeoJSON objects
    from computed routes for frontend map rendering.
    """

    def __init__(self, road_graph: RoadNetworkGraph):
        self.road_graph = road_graph

    def build_route_geojson(self, route: Route) -> Dict[str, Any]:
        """
        Convert a Route object into a standardized GeoJSON FeatureCollection.

        Includes:
        - Route polyline Feature (LineString)
        - Origin snap Feature (Point)
        - Destination snap Feature (Point)
        - Step-by-step turn and highway maneuvers
        - Bounding box and summary metrics
        """
        ordered_coords: List[List[float]] = []
        steps: List[Dict[str, Any]] = []
        distinct_road_names: List[str] = []
        distinct_road_classes: List[str] = []

        min_lat, min_lon = float("inf"), float("inf")
        max_lat, max_lon = float("-inf"), float("-inf")

        # Step 1: Add origin point
        orig_lat = route.origin_snap.snapped_latitude
        orig_lon = route.origin_snap.snapped_longitude
        ordered_coords.append([orig_lon, orig_lat])
        min_lat, max_lat = min(min_lat, orig_lat), max(max_lat, orig_lat)
        min_lon, max_lon = min(min_lon, orig_lon), max(max_lon, orig_lon)

        current_road_name = None
        accumulated_dist_m = 0.0
        accumulated_time_s = 0.0
        step_coords: List[List[float]] = []

        for seg in route.edge_sequence:
            # Retrieve true geometry if stored on edge
            edge_data = self.road_graph.get_edge(seg.u, seg.v, seg.key) or {}
            geom = edge_data.get("geometry")

            seg_coords: List[Tuple[float, float]] = []
            if isinstance(geom, LineString):
                seg_coords = list(geom.coords)
            else:
                u_node = self.road_graph.get_node(seg.u) or {}
                v_node = self.road_graph.get_node(seg.v) or {}
                u_lat = u_node.get("latitude", 0.0)
                u_lon = u_node.get("longitude", 0.0)
                v_lat = v_node.get("latitude", 0.0)
                v_lon = v_node.get("longitude", 0.0)
                seg_coords = [(u_lon, u_lat), (v_lon, v_lat)]

            # Accumulate ordered coordinates
            for lon, lat in seg_coords:
                if not ordered_coords or (ordered_coords[-1][0] != lon or ordered_coords[-1][1] != lat):
                    ordered_coords.append([lon, lat])
                min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)

            # Track highway tags
            road_label = seg.name or seg.road_reference or "Unnamed Road"
            if seg.name and seg.name not in distinct_road_names:
                distinct_road_names.append(seg.name)
            if seg.road_reference and seg.road_reference not in distinct_road_names:
                distinct_road_names.append(seg.road_reference)
            if seg.highway_type and seg.highway_type not in distinct_road_classes:
                distinct_road_classes.append(seg.highway_type)

            # Maneuver step aggregation
            if current_road_name is None:
                current_road_name = road_label
                accumulated_dist_m = seg.length_m
                accumulated_time_s = seg.traversal_time_s
                step_coords = [list(c) for c in seg_coords]
            elif road_label == current_road_name:
                accumulated_dist_m += seg.length_m
                accumulated_time_s += seg.traversal_time_s
                step_coords.extend([list(c) for c in seg_coords])
            else:
                # Flush previous step
                steps.append({
                    "instruction": f"Continue on {current_road_name}",
                    "road_name": current_road_name,
                    "distance_m": round(accumulated_dist_m, 1),
                    "distance_km": round(accumulated_dist_m / 1000.0, 2),
                    "duration_s": round(accumulated_time_s, 1),
                    "duration_mins": round(accumulated_time_s / 60.0, 1),
                })
                current_road_name = road_label
                accumulated_dist_m = seg.length_m
                accumulated_time_s = seg.traversal_time_s
                step_coords = [list(c) for c in seg_coords]

        # Flush final step
        if current_road_name is not None:
            steps.append({
                "instruction": f"Proceed on {current_road_name} to destination",
                "road_name": current_road_name,
                "distance_m": round(accumulated_dist_m, 1),
                "distance_km": round(accumulated_dist_m / 1000.0, 2),
                "duration_s": round(accumulated_time_s, 1),
                "duration_mins": round(accumulated_time_s / 60.0, 1),
            })

        # Append destination point if not identical
        dest_lat = route.destination_snap.snapped_latitude
        dest_lon = route.destination_snap.snapped_longitude
        if not ordered_coords or (ordered_coords[-1][0] != dest_lon or ordered_coords[-1][1] != dest_lat):
            ordered_coords.append([dest_lon, dest_lat])

        bbox = [min_lon, min_lat, max_lon, max_lat] if min_lat != float("inf") else [0.0, 0.0, 0.0, 0.0]

        # Construct GeoJSON Features
        route_line_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": ordered_coords,
            },
            "properties": {
                "route_id": route.route_id,
                "name": route.name,
                "total_distance_m": route.total_distance_m,
                "total_distance_km": route.total_distance_km,
                "total_time_s": route.total_time_s,
                "total_time_minutes": route.total_time_minutes,
                "cost_metric": route.cost_metric.value,
                "distinct_road_names": distinct_road_names,
                "distinct_road_classes": distinct_road_classes,
                "steps": steps,
            },
        }

        origin_point_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [route.origin_snap.input_longitude, route.origin_snap.input_latitude],
            },
            "properties": {
                "type": "origin",
                "snapped_coordinates": [route.origin_snap.snapped_longitude, route.origin_snap.snapped_latitude],
                "snap_distance_m": route.origin_snap.distance_meters,
                "nearest_road": route.origin_snap.road_name or route.origin_snap.road_reference,
            },
        }

        dest_point_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [route.destination_snap.input_longitude, route.destination_snap.input_latitude],
            },
            "properties": {
                "type": "destination",
                "snapped_coordinates": [route.destination_snap.snapped_longitude, route.destination_snap.snapped_latitude],
                "snap_distance_m": route.destination_snap.distance_meters,
                "nearest_road": route.destination_snap.road_name or route.destination_snap.road_reference,
            },
        }

        return {
            "type": "FeatureCollection",
            "bbox": bbox,
            "features": [route_line_feature, origin_point_feature, dest_point_feature],
            "properties": {
                "route_id": route.route_id,
                "name": route.name,
                "total_distance_km": route.total_distance_km,
                "total_time_minutes": route.total_time_minutes,
                "segment_count": len(route.edge_sequence),
                "coordinate_count": len(ordered_coords),
            },
        }
