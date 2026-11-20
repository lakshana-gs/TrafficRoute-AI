"""Reproducible OpenStreetMap road network extraction and parsing for TrafficRoute AI."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import requests
from shapely.geometry import LineString

from src.network.graph import RoadNetworkGraph
from src.network.provenance import NetworkProvenance


# Core pilot bounding box: Coimbatore Metro + Coimbatore -> Avinashi (NH-544) + Annur / Somanur corridors
COIMBATORE_AVINASHI_BBOX = {
    "south": 10.9400,
    "north": 11.2300,
    "west": 76.9000,
    "east": 77.3000,
}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

REQUEST_HEADERS = {
    "User-Agent": "TrafficRouteAI-PredictiveSystem/1.0 (TamilNadu Route Research; contact@trafficroute.ai)",
    "Accept": "application/json",
}

MOTORABLE_HIGHWAY_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}

EXCLUDED_HIGHWAY_TYPES = {
    "footway",
    "pedestrian",
    "path",
    "steps",
    "cycleway",
    "bridleway",
    "corridor",
    "track",
    "proposed",
    "construction",
    "abandoned",
    "raceway",
}


def compute_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in meters."""
    r = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
    return r * c


def fetch_osm_overpass_data(
    bbox: Optional[Dict[str, float]] = None,
    timeout_s: int = 90,
) -> Dict[str, Any]:
    """
    Fetch road elements from Overpass API within bounding box.
    Tries multiple global mirror endpoints with proper headers.
    """
    bounds = bbox or COIMBATORE_AVINASHI_BBOX
    query = f"""[out:json][timeout:{timeout_s}];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
);
out body;
>;
out skel qt;
"""
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(
                endpoint, data={"data": query}, headers=REQUEST_HEADERS, timeout=timeout_s
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("elements"):
                    return data
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to fetch OSM data from all Overpass endpoints: {last_error}")


def parse_osm_json_to_graph(
    osm_data: Dict[str, Any],
    provenance: Optional[NetworkProvenance] = None,
    retain_largest_component: bool = True,
) -> RoadNetworkGraph:
    """
    Convert raw Overpass OSM JSON payload into a directed NetworkX MultiDiGraph.

    Args:
        osm_data: Dictionary returned from Overpass JSON endpoint.
        provenance: Associated NetworkProvenance metadata.
        retain_largest_component: If True, filters out small disconnected disconnected components.

    Returns:
        Standardized RoadNetworkGraph instance.
    """
    elements = osm_data.get("elements", [])
    nodes_dict: Dict[int, Dict[str, Any]] = {}
    ways_list: List[Dict[str, Any]] = []

    for elem in elements:
        elem_type = elem.get("type")
        if elem_type == "node":
            nodes_dict[elem["id"]] = {
                "latitude": float(elem["lat"]),
                "longitude": float(elem["lon"]),
                "tags": elem.get("tags", {}),
            }
        elif elem_type == "way":
            ways_list.append(elem)

    g = nx.MultiDiGraph()

    for way in ways_list:
        tags = way.get("tags", {})
        hw = tags.get("highway")
        if not hw or hw in EXCLUDED_HIGHWAY_TYPES:
            continue

        way_nodes = way.get("nodes", [])
        if len(way_nodes) < 2:
            continue

        # Check accessibility tags
        if tags.get("motor_vehicle") == "no" or tags.get("motorcar") == "no" or tags.get("access") in ("private", "no"):
            continue

        oneway_tag = tags.get("oneway", "no")
        junction_tag = tags.get("junction")
        is_oneway = (
            oneway_tag in ("yes", "1", "true")
            or hw in ("motorway", "motorway_link")
            or junction_tag in ("roundabout", "circular")
        )
        is_reverse_oneway = oneway_tag in ("-1", "reverse")

        name = tags.get("name")
        ref = tags.get("ref")
        maxspeed = tags.get("maxspeed")
        lanes = tags.get("lanes")
        bridge = tags.get("bridge")
        tunnel = tags.get("tunnel")
        surface = tags.get("surface")

        for i in range(len(way_nodes) - 1):
            u_id = way_nodes[i]
            v_id = way_nodes[i + 1]

            if u_id not in nodes_dict or v_id not in nodes_dict:
                continue

            u_node = nodes_dict[u_id]
            v_node = nodes_dict[v_id]

            # Add nodes to graph if not present
            if u_id not in g.nodes:
                g.add_node(u_id, latitude=u_node["latitude"], longitude=u_node["longitude"], y=u_node["latitude"], x=u_node["longitude"])
            if v_id not in g.nodes:
                g.add_node(v_id, latitude=v_node["latitude"], longitude=v_node["longitude"], y=v_node["latitude"], x=v_node["longitude"])

            seg_len = compute_haversine_distance(
                u_node["latitude"], u_node["longitude"], v_node["latitude"], v_node["longitude"]
            )
            if seg_len <= 0.01:
                seg_len = 1.0  # minimal baseline to avoid division by zero

            geom_forward = LineString([
                (u_node["longitude"], u_node["latitude"]),
                (v_node["longitude"], v_node["latitude"]),
            ])

            base_edge_attrs = {
                "osmid": way.get("id"),
                "highway": hw,
                "name": name,
                "ref": ref,
                "maxspeed": maxspeed,
                "lanes": lanes,
                "bridge": bridge,
                "tunnel": tunnel,
                "surface": surface,
                "length": seg_len,
            }

            if is_reverse_oneway:
                # Backward only: v -> u
                geom_rev = LineString([
                    (v_node["longitude"], v_node["latitude"]),
                    (u_node["longitude"], u_node["latitude"]),
                ])
                rev_attrs = dict(base_edge_attrs, geometry=geom_rev, oneway=True)
                g.add_edge(v_id, u_id, **rev_attrs)
            elif is_oneway:
                # Forward only: u -> v
                fwd_attrs = dict(base_edge_attrs, geometry=geom_forward, oneway=True)
                g.add_edge(u_id, v_id, **fwd_attrs)
            else:
                # Two-way: u -> v and v -> u
                geom_rev = LineString([
                    (v_node["longitude"], v_node["latitude"]),
                    (u_node["longitude"], u_node["latitude"]),
                ])
                fwd_attrs = dict(base_edge_attrs, geometry=geom_forward, oneway=False)
                rev_attrs = dict(base_edge_attrs, geometry=geom_rev, oneway=False)
                g.add_edge(u_id, v_id, **fwd_attrs)
                g.add_edge(v_id, u_id, **rev_attrs)

    if retain_largest_component and g.number_of_nodes() > 0:
        # Retain largest strongly connected or weakly connected component
        # In directed street networks, weakly connected components ensure route connectivity
        components = list(nx.strongly_connected_components(g))
        if components:
            largest_scc = max(components, key=len)
            # If largest SCC contains majority of nodes, use it; otherwise use largest WCC
            if len(largest_scc) >= 0.5 * g.number_of_nodes():
                g = g.subgraph(largest_scc).copy()
            else:
                largest_wcc = max(nx.weakly_connected_components(g), key=len)
                g = g.subgraph(largest_wcc).copy()

    road_graph = RoadNetworkGraph(graph=g, provenance=provenance)
    return road_graph


def load_or_extract_pilot_network(
    data_dir: Path | str = "data",
    force_reextract: bool = False,
) -> RoadNetworkGraph:
    """
    Load the Coimbatore-Avinashi pilot network from disk if available, or extract it from OSM.

    Args:
        data_dir: Base directory containing 'raw', 'processed', and 'metadata' subdirectories.
        force_reextract: If True, re-fetches from Overpass API regardless of existing files.

    Returns:
        Loaded/Extracted RoadNetworkGraph instance.
    """
    base_path = Path(data_dir)
    raw_path = base_path / "raw" / "coimbatore_avinashi_osm.json"
    bin_path = base_path / "processed" / "coimbatore_avinashi_network.bin"
    graphml_path = base_path / "processed" / "coimbatore_avinashi_network.graphml"
    geojson_path = base_path / "processed" / "coimbatore_avinashi_edges.geojson"
    metadata_path = base_path / "metadata" / "coimbatore_avinashi_provenance.json"

    # Step 1: Check if fast binary cache exists
    if not force_reextract and bin_path.exists():
        try:
            return RoadNetworkGraph.load_binary(bin_path)
        except Exception:
            pass

    # Step 2: Check if processed graphml already exists and is valid
    if not force_reextract and graphml_path.exists() and metadata_path.exists():
        try:
            provenance = NetworkProvenance.load_json(metadata_path)
            road_graph = RoadNetworkGraph.load_graphml(graphml_path, provenance=provenance)
            if road_graph.node_count > 0:
                road_graph.save_binary(bin_path)
                return road_graph
        except Exception:
            pass

    # Step 3: Check if raw OSM JSON exists; if not, fetch from Overpass API
    if not raw_path.exists() or force_reextract:
        osm_data = fetch_osm_overpass_data(bbox=COIMBATORE_AVINASHI_BBOX)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(osm_data, f)
    else:
        with open(raw_path, "r", encoding="utf-8") as f:
            osm_data = json.load(f)

    # Step 4: Create provenance record
    provenance = NetworkProvenance(
        dataset_name="Coimbatore_Avinashi_Pilot_Road_Network",
        source="OpenStreetMap",
        source_url="https://www.openstreetmap.org",
        license="Open Database License (ODbL) 1.0",
        license_url="https://opendatacommons.org/licenses/odbl/1-0/",
        extraction_date=datetime.now(timezone.utc).isoformat(),
        extraction_method="Overpass API Bounding Box Extraction",
        geographic_boundary=COIMBATORE_AVINASHI_BBOX,
        osm_filter="highway~motorway|trunk|primary|secondary|tertiary|unclassified|residential",
        preprocessing_version="1.0.0",
        extra_metadata={
            "corridor": "Coimbatore Metropolitan Area <-> Avinashi (NH-544)",
            "bounds": COIMBATORE_AVINASHI_BBOX,
        },
    )

    # Step 5: Parse and construct graph
    road_graph = parse_osm_json_to_graph(osm_data, provenance=provenance, retain_largest_component=True)

    # Step 6: Save processed artifacts (both GraphML and fast binary format)
    road_graph.save_binary(bin_path)
    road_graph.save_graphml(graphml_path)
    road_graph.save_geojson(geojson_path)
    if road_graph.provenance:
        road_graph.provenance.node_count = road_graph.node_count
        road_graph.provenance.edge_count = road_graph.edge_count
        road_graph.provenance.save_json(metadata_path)

    return road_graph
