"""End-to-end integration test for the real Coimbatore -> Avinashi routing pipeline."""

import pytest

from src.network.alternatives import AlternativeRouteGenerator, compute_route_overlap
from src.network.extraction import load_or_extract_pilot_network
from src.network.geometry import RouteGeometryBuilder
from src.network.routing import CostMetric, RoutingEngine


@pytest.fixture(scope="module")
def pilot_graph():
    """Load the real extracted Coimbatore-Avinashi road network."""
    return load_or_extract_pilot_network(force_reextract=False)


def test_coimbatore_to_avinashi_end_to_end(pilot_graph):
    # Real World Coordinates:
    # Origin: Coimbatore Railway Station
    cbe_station_lat, cbe_station_lon = 10.9990, 76.9630
    # Destination: Avinashi (New Bus Stand / Central Junction)
    avinashi_lat, avinashi_lon = 11.1930, 77.2690

    # Step 1: Initialize routing components
    engine = RoutingEngine(pilot_graph)
    alt_gen = AlternativeRouteGenerator(engine, max_shared_ratio=0.80)
    geom_builder = RouteGeometryBuilder(pilot_graph)

    # Step 2: Location Snapping Verification
    orig_snap = engine.indexer.snap_coordinate(cbe_station_lat, cbe_station_lon)
    dest_snap = engine.indexer.snap_coordinate(avinashi_lat, avinashi_lon)

    assert orig_snap.is_valid is True
    assert orig_snap.distance_meters < 100.0  # Railway station should snap to nearby approach road within 100m
    assert dest_snap.is_valid is True
    assert dest_snap.distance_meters < 100.0

    # Step 3: Compute Primary Fastest Route
    primary_route = engine.calculate_route(
        origin_lat=cbe_station_lat,
        origin_lon=cbe_station_lon,
        destination_lat=avinashi_lat,
        destination_lon=avinashi_lon,
        metric=CostMetric.FREE_FLOW_TIME,
        route_name="Primary Route (NH-544)",
    )

    assert primary_route is not None
    # Real physical distance between Coimbatore Junction and Avinashi is ~42 - 45 km
    assert 40.0 <= primary_route.total_distance_km <= 48.0
    # Free-flow time is ~30 - 45 minutes
    assert 30.0 <= primary_route.total_time_minutes <= 50.0
    assert len(primary_route.edge_sequence) > 100

    # Verify prominent road names along the corridor
    corridor_names = [seg.name for seg in primary_route.edge_sequence if seg.name]
    corridor_refs = [seg.road_reference for seg in primary_route.edge_sequence if seg.road_reference]
    all_labels = set(corridor_names + corridor_refs)

    # NH-544 or Avinashi Road should be present
    assert any("NH" in label or "Avinashi" in label or "Salem" in label for label in all_labels)

    # Step 4: Generate Alternative Routes
    alt_routes = alt_gen.generate_alternative_routes(
        origin_lat=cbe_station_lat,
        origin_lon=cbe_station_lon,
        destination_lat=avinashi_lat,
        destination_lon=avinashi_lon,
        k_routes=3,
        metric=CostMetric.FREE_FLOW_TIME,
    )

    assert len(alt_routes) >= 2  # At least 2 distinct feasible corridors
    for r in alt_routes:
        assert 38.0 <= r.total_distance_km <= 65.0
        assert r.total_time_minutes > 0

    if len(alt_routes) >= 2:
        overlap = compute_route_overlap(alt_routes[0], alt_routes[1])
        assert overlap <= 0.85  # Proven diverse alternatives

    # Step 5: Construct GeoJSON Output
    geojson_output = geom_builder.build_route_geojson(primary_route)
    assert geojson_output["type"] == "FeatureCollection"
    assert len(geojson_output["features"]) == 3
    assert geojson_output["features"][0]["geometry"]["type"] == "LineString"
    assert len(geojson_output["features"][0]["geometry"]["coordinates"]) > 200
    assert "bbox" in geojson_output
    assert len(geojson_output["features"][0]["properties"]["steps"]) > 0

    # Step 6: Verify Provenance Metadata
    assert pilot_graph.provenance is not None
    assert pilot_graph.provenance.source == "OpenStreetMap"
    assert "ODbL" in pilot_graph.provenance.license
