"""Road network graph representation, schema enforcement, and serialization for TrafficRoute AI."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
from shapely.geometry import LineString, Point, mapping, shape
from shapely import wkt

from src.network.provenance import NetworkProvenance


# Standard road hierarchy with engineering-calibrated fallback free-flow speeds (km/h) for Indian networks
ROAD_CLASSIFICATION_HIERARCHY = {
    "motorway": {"tier": 1, "default_speed_kmh": 100.0, "description": "High-speed controlled-access highway"},
    "motorway_link": {"tier": 1, "default_speed_kmh": 60.0, "description": "Motorway slip road/ramp"},
    "trunk": {"tier": 2, "default_speed_kmh": 80.0, "description": "National Highway (NH) corridor"},
    "trunk_link": {"tier": 2, "default_speed_kmh": 50.0, "description": "Trunk road slip road/ramp"},
    "primary": {"tier": 3, "default_speed_kmh": 60.0, "description": "State Highway (SH) or primary urban arterial"},
    "primary_link": {"tier": 3, "default_speed_kmh": 40.0, "description": "Primary road slip road/ramp"},
    "secondary": {"tier": 4, "default_speed_kmh": 45.0, "description": "Major District Road (MDR) / connector"},
    "secondary_link": {"tier": 4, "default_speed_kmh": 35.0, "description": "Secondary road link"},
    "tertiary": {"tier": 5, "default_speed_kmh": 35.0, "description": "Other District Road / collector road"},
    "tertiary_link": {"tier": 5, "default_speed_kmh": 25.0, "description": "Tertiary road link"},
    "unclassified": {"tier": 6, "default_speed_kmh": 30.0, "description": "Local interconnecting road"},
    "residential": {"tier": 7, "default_speed_kmh": 25.0, "description": "Neighborhood residential street"},
    "service": {"tier": 8, "default_speed_kmh": 15.0, "description": "Access road / alleyway / parking access"},
    "living_street": {"tier": 9, "default_speed_kmh": 15.0, "description": "Pedestrian-priority shared street"},
}

DEFAULT_FALLBACK_SPEED_KMH = 30.0


def parse_maxspeed_value(val: Any) -> Optional[float]:
    """Parse raw OSM maxspeed string/number into numeric km/h."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    if isinstance(val, list):
        for item in val:
            parsed = parse_maxspeed_value(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if not val_clean or val_clean in ("none", "null", "signals", "variable", "unknown"):
            return None
        # Handle formats like "80 km/h", "80 mph", "80"
        parts = val_clean.split()
        try:
            speed = float(parts[0])
            if len(parts) > 1 and "mph" in parts[1]:
                speed *= 1.60934
            return speed if speed > 0 else None
        except (ValueError, IndexError):
            return None
    return None


def parse_lanes_value(val: Any) -> Optional[int]:
    """Parse raw OSM lanes tag into integer."""
    if val is None:
        return None
    if isinstance(val, int):
        return val if val > 0 else None
    if isinstance(val, float):
        return int(val) if val > 0 else None
    if isinstance(val, list):
        for item in val:
            parsed = parse_lanes_value(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(val, str):
        try:
            return int(val.split(";")[0].strip())
        except (ValueError, IndexError):
            return None
    return None


def normalize_highway_type(val: Any) -> str:
    """Normalize highway tag into primary classification string."""
    if isinstance(val, list):
        # Pick highest priority hierarchy if multiple listed
        for candidate in val:
            if candidate in ROAD_CLASSIFICATION_HIERARCHY:
                return str(candidate)
        return str(val[0]) if val else "unclassified"
    if isinstance(val, str):
        return val.strip().lower()
    return "unclassified"


class RoadNetworkGraph:
    """
    Encapsulates a directed spatial road graph for routing and speed integration.
    Wraps a NetworkX MultiDiGraph with schema validation, geometry indexing, and serialization.
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None, provenance: Optional[NetworkProvenance] = None):
        self.graph: nx.MultiDiGraph = graph if graph is not None else nx.MultiDiGraph()
        self.provenance: Optional[NetworkProvenance] = provenance
        self._standardize_schema()

    def _standardize_schema(self) -> None:
        """Standardize node and edge attributes across the graph."""
        # Process Nodes
        for node_id, data in self.graph.nodes(data=True):
            data["node_id"] = node_id
            # Standardize latitude / longitude
            if "y" in data and "latitude" not in data:
                data["latitude"] = float(data["y"])
            elif "latitude" in data:
                data["latitude"] = float(data["latitude"])
                data["y"] = data["latitude"]

            if "x" in data and "longitude" not in data:
                data["longitude"] = float(data["x"])
            elif "longitude" in data:
                data["longitude"] = float(data["longitude"])
                data["x"] = data["longitude"]

        # Process Edges
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            # Edge identifier
            data["u"] = u
            data["v"] = v
            data["key"] = k
            if "edge_id" not in data:
                data["edge_id"] = f"{u}_{v}_{k}"

            # Standardize highway classification
            raw_hw = data.get("highway", "unclassified")
            norm_hw = normalize_highway_type(raw_hw)
            data["highway_type"] = norm_hw

            # Standardize road name and reference
            raw_name = data.get("name")
            if isinstance(raw_name, list):
                data["name"] = " / ".join(str(n) for n in raw_name if n)
            elif isinstance(raw_name, str) and raw_name.strip():
                data["name"] = raw_name.strip()
            else:
                data["name"] = None

            raw_ref = data.get("ref")
            if isinstance(raw_ref, list):
                data["road_reference"] = " / ".join(str(r) for r in raw_ref if r)
            elif isinstance(raw_ref, str) and raw_ref.strip():
                data["road_reference"] = raw_ref.strip()
            else:
                data["road_reference"] = None

            # Standardize length in meters
            if "length" in data:
                data["length_m"] = float(data["length"])
            elif "length_m" in data:
                data["length_m"] = float(data["length_m"])
            else:
                # Compute geodesic length if geometry exists
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                data["length_m"] = self._compute_haversine_distance(
                    u_data["latitude"], u_data["longitude"], v_data["latitude"], v_data["longitude"]
                )

            # Standardize oneway
            raw_oneway = data.get("oneway", False)
            if isinstance(raw_oneway, str):
                data["oneway"] = raw_oneway.lower() in ("yes", "true", "1", "-1")
            elif isinstance(raw_oneway, bool):
                data["oneway"] = raw_oneway
            else:
                data["oneway"] = False

            # Standardize maxspeed
            data["maxspeed"] = parse_maxspeed_value(data.get("maxspeed"))

            # Derive free_flow_speed_kmh (fallback heuristic from hierarchy if maxspeed not in OSM)
            if data["maxspeed"] is not None and data["maxspeed"] > 0:
                data["free_flow_speed_kmh"] = data["maxspeed"]
                data["speed_source"] = "OSM_MAXSPEED"
            else:
                hw_info = ROAD_CLASSIFICATION_HIERARCHY.get(norm_hw)
                fallback = hw_info["default_speed_kmh"] if hw_info else DEFAULT_FALLBACK_SPEED_KMH
                data["free_flow_speed_kmh"] = fallback
                data["speed_source"] = "DERIVED_ROAD_HIERARCHY"

            # Derive free_flow_travel_time_sec (T = L / v)
            speed_mps = data["free_flow_speed_kmh"] * (1000.0 / 3600.0)
            data["free_flow_time_s"] = data["length_m"] / max(speed_mps, 0.1)

            # Standardize lanes, bridge, tunnel
            data["lanes"] = parse_lanes_value(data.get("lanes"))
            data["bridge"] = bool(data.get("bridge")) if "bridge" in data and data["bridge"] not in (False, "no", None) else None
            data["tunnel"] = bool(data.get("tunnel")) if "tunnel" in data and data["tunnel"] not in (False, "no", None) else None
            data["source"] = "OpenStreetMap"

            # Standardize geometry
            if "geometry" in data:
                geom = data["geometry"]
                if isinstance(geom, str):
                    try:
                        data["geometry"] = wkt.loads(geom)
                    except Exception:
                        pass
                elif isinstance(geom, dict):
                    data["geometry"] = shape(geom)
            else:
                # Default straight LineString between u and v
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                data["geometry"] = LineString([(u_data["longitude"], u_data["latitude"]), (v_data["longitude"], v_data["latitude"])])

    @staticmethod
    def _compute_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two WGS84 points in meters."""
        from math import asin, cos, radians, sin, sqrt

        r = 6371000.0  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
        c = 2.0 * asin(sqrt(a))
        return r * c

    @property
    def node_count(self) -> int:
        """Number of vertices in the road network."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of directed edges in the road network."""
        return self.graph.number_of_edges()

    def get_node(self, node_id: Any) -> Optional[Dict[str, Any]]:
        """Retrieve node attribute dictionary."""
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None

    def get_edge(self, u: Any, v: Any, key: int = 0) -> Optional[Dict[str, Any]]:
        """Retrieve edge attribute dictionary."""
        if self.graph.has_edge(u, v, key):
            return dict(self.graph.edges[u, v, key])
        return None

    def save_graphml(self, file_path: Path | str) -> None:
        """Save graph to GraphML format with sanitized attributes (no None values)."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        g_copy = nx.MultiDiGraph()

        # Sanitize nodes
        for node_id, data in self.graph.nodes(data=True):
            clean_node_data = {}
            for k, v in data.items():
                if v is not None:
                    if isinstance(v, (int, float, str, bool)):
                        clean_node_data[str(k)] = v
                    else:
                        clean_node_data[str(k)] = str(v)
            g_copy.add_node(node_id, **clean_node_data)

        # Sanitize edges
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            clean_edge_data = {}
            for attr_k, attr_v in data.items():
                if attr_k == "key":
                    continue
                if attr_v is not None:
                    if attr_k == "geometry" and isinstance(attr_v, LineString):
                        clean_edge_data["geometry_wkt"] = attr_v.wkt
                    elif isinstance(attr_v, (int, float, str, bool)):
                        clean_edge_data[str(attr_k)] = attr_v
                    elif isinstance(attr_v, LineString):
                        clean_edge_data["geometry_wkt"] = attr_v.wkt
                    else:
                        clean_edge_data[str(attr_k)] = str(attr_v)
            g_copy.add_edge(u, v, key=k, **clean_edge_data)

        nx.write_graphml(g_copy, str(path))

    @classmethod
    def load_graphml(cls, file_path: Path | str, provenance: Optional[NetworkProvenance] = None) -> "RoadNetworkGraph":
        """Load graph from a GraphML file and restore numeric types and geometries."""
        path = Path(file_path)
        g_raw = nx.read_graphml(str(path))
        g = nx.MultiDiGraph()

        # Restore nodes with integer IDs if possible
        for node_id, data in g_raw.nodes(data=True):
            nid = int(node_id) if str(node_id).isdigit() else node_id
            node_data = dict(data)
            if "latitude" in node_data:
                node_data["latitude"] = float(node_data["latitude"])
                node_data["y"] = node_data["latitude"]
            elif "y" in node_data:
                node_data["latitude"] = float(node_data["y"])
                node_data["y"] = node_data["latitude"]

            if "longitude" in node_data:
                node_data["longitude"] = float(node_data["longitude"])
                node_data["x"] = node_data["longitude"]
            elif "x" in node_data:
                node_data["longitude"] = float(node_data["x"])
                node_data["x"] = node_data["longitude"]

            g.add_node(nid, **node_data)

        # Restore edges
        for u, v, k, data in g_raw.edges(keys=True, data=True):
            u_id = int(u) if str(u).isdigit() else u
            v_id = int(v) if str(v).isdigit() else v
            k_id = int(k) if str(k).isdigit() else k
            edge_data = dict(data)

            if "length_m" in edge_data:
                edge_data["length_m"] = float(edge_data["length_m"])
            if "free_flow_speed_kmh" in edge_data:
                edge_data["free_flow_speed_kmh"] = float(edge_data["free_flow_speed_kmh"])
            if "free_flow_time_s" in edge_data:
                edge_data["free_flow_time_s"] = float(edge_data["free_flow_time_s"])
            if "maxspeed" in edge_data:
                edge_data["maxspeed"] = float(edge_data["maxspeed"])
            if "lanes" in edge_data:
                edge_data["lanes"] = int(edge_data["lanes"])
            if "oneway" in edge_data:
                edge_data["oneway"] = str(edge_data["oneway"]).lower() in ("true", "1", "yes")

            if "geometry_wkt" in edge_data:
                try:
                    edge_data["geometry"] = wkt.loads(edge_data["geometry_wkt"])
                except Exception:
                    pass
                del edge_data["geometry_wkt"]

    def save_binary(self, file_path: Path | str) -> None:
        """Save graph and provenance to optimized binary format for rapid loading."""
        import pickle
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self.graph,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_binary(cls, file_path: Path | str) -> "RoadNetworkGraph":
        """Load graph and provenance from optimized binary file in sub-second time."""
        import pickle
        path = Path(file_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        prov = NetworkProvenance(**payload["provenance"]) if payload.get("provenance") else None
        return cls(graph=payload["graph"], provenance=prov)

    def to_geojson(self) -> Dict[str, Any]:
        """Convert all edges in the graph to a GeoJSON FeatureCollection."""
        features = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            geom = data.get("geometry")
            if isinstance(geom, LineString):
                geom_dict = mapping(geom)
            else:
                u_node = self.graph.nodes[u]
                v_node = self.graph.nodes[v]
                geom_dict = {
                    "type": "LineString",
                    "coordinates": [
                        [u_node["longitude"], u_node["latitude"]],
                        [v_node["longitude"], v_node["latitude"]],
                    ],
                }

            props = {
                "edge_id": data.get("edge_id", f"{u}_{v}_{k}"),
                "u": u,
                "v": v,
                "key": k,
                "name": data.get("name"),
                "road_reference": data.get("road_reference"),
                "highway_type": data.get("highway_type"),
                "length_m": round(data.get("length_m", 0.0), 2),
                "free_flow_speed_kmh": data.get("free_flow_speed_kmh"),
                "free_flow_time_s": round(data.get("free_flow_time_s", 0.0), 2),
                "oneway": data.get("oneway", False),
                "lanes": data.get("lanes"),
            }

            features.append({
                "type": "Feature",
                "geometry": geom_dict,
                "properties": props,
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def save_geojson(self, file_path: Path | str) -> None:
        """Export edges to a GeoJSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, ensure_ascii=False)
