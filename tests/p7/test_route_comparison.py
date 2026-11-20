"""Unit tests for P7.3: Predictive Multi-Route Comparison & Recommendation Facade."""

import copy
from datetime import datetime, timezone
import networkx as nx
import pytest

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RoutePlanResult, RouteSegment, RouteStatus
from src.network.snapping import SnapResult
from src.traffic.congestion import CongestionLevel
from src.traffic.predictive_evaluation import (
    MockSpeedProvider,
    PredictiveEvaluatorConfig,
    PredictiveRouteEvaluator,
    RouteComparisonResult,
)


@pytest.fixture
def base_snap():
    return SnapResult(
        input_latitude=11.00,
        input_longitude=77.00,
        snapped_latitude=11.00,
        snapped_longitude=77.00,
        nearest_node_id=1,
        distance_meters=0.0,
        is_valid=True,
    )


def make_test_route(segments, base_snap, route_id="route_1", name="Corridor 1"):
    """Helper to assemble a valid Route object from a list of RouteSegments."""
    nodes = [segments[0].u] + [s.v for s in segments]
    tot_dist = sum(s.length_m for s in segments)
    tot_time = sum(s.traversal_time_s for s in segments)
    return Route(
        route_id=route_id,
        name=name,
        origin_snap=base_snap,
        destination_snap=base_snap,
        node_sequence=nodes,
        edge_sequence=segments,
        total_distance_m=tot_dist,
        total_time_s=tot_time,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )


# =====================================================================
# 1. Multi-Route Ranking by Predicted Travel Time
# =====================================================================

def test_multi_route_ranking_by_predicted_travel_time(base_snap):
    """
    Verify routes are ranked ascending by forward-propagated travel time,
    the fastest alternative is recommended, and savings are calculated
    strictly relative to the primary route.
    """
    # Primary: 2000m, free-flow 60 km/h (120s free-flow)
    prim_seg = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=2000.0, speed_kmh=60.0,
        traversal_time_s=120.0, name="Primary Highway", road_reference="NH-544", highway_type="trunk"
    )
    primary = make_test_route([prim_seg], base_snap, route_id="primary", name="Primary Route")

    # Alt 1: 2500m, free-flow 60 km/h (150s free-flow)
    alt1_seg = RouteSegment(
        edge_id="1_3_0", u=1, v=3, key=0, length_m=2500.0, speed_kmh=60.0,
        traversal_time_s=150.0, name="Arterial Bypass", road_reference="SH-15", highway_type="primary"
    )
    alt1 = make_test_route([alt1_seg], base_snap, route_id="alt1", name="Arterial Bypass")

    # Alt 2: 3000m, free-flow 60 km/h (180s free-flow)
    alt2_seg = RouteSegment(
        edge_id="1_4_0", u=1, v=4, key=0, length_m=3000.0, speed_kmh=60.0,
        traversal_time_s=180.0, name="Outer Ring", road_reference=None, highway_type="trunk"
    )
    alt2 = make_test_route([alt2_seg], base_snap, route_id="alt2", name="Outer Ring")

    # Dynamic speeds:
    # Primary: 20 km/h (severe peak congestion) -> trav = 2000 / (20 * 1000/3600) = 360.0s (6.0 min)
    # Alt 1: 40 km/h (moderate traffic) -> trav = 2500 / (40 * 1000/3600) = 225.0s (3.75 min)
    # Alt 2: 60 km/h (free flowing) -> trav = 3000 / (60 * 1000/3600) = 180.0s (3.0 min)
    edge_speeds = {
        (1, 2, 0): 20.0,
        (1, 3, 0): 40.0,
        (1, 4, 0): 60.0,
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 8, 30, tzinfo=timezone.utc)

    evaluator = PredictiveRouteEvaluator(speed_provider=provider)
    result = evaluator.evaluate_routes(
        primary_route=primary,
        alternative_routes=[alt1, alt2],
        departure_time=dep_time,
    )

    assert isinstance(result, RouteComparisonResult)
    # 1. Ranking: Alt 2 (180s) < Alt 1 (225s) < Primary (360s)
    assert len(result.ranked_routes) == 3
    assert result.ranked_routes[0].route_id == "alt2"
    assert result.ranked_routes[1].route_id == "alt1"
    assert result.ranked_routes[2].route_id == "primary"

    # 2. Recommendation: Alt 2 recommended, saves 360 - 180 = 180s (3.0 min)
    assert result.recommended_route_id == "alt2"
    assert pytest.approx(result.time_savings_seconds, abs=0.1) == 180.0
    assert pytest.approx(result.time_savings_minutes, abs=0.1) == 3.0
    assert "saves 3.0 mins" in result.recommendation_reason


# =====================================================================
# 2. Minimum Time Savings Threshold Behavior
# =====================================================================

def test_minimum_time_savings_threshold_retains_primary(base_snap):
    """
    Verify that if an alternative is faster but saves less than
    min_time_savings_seconds (default 60.0s), the recommendation remains
    with the primary route to prevent recommending detours for trivial gains.
    """
    # Primary: travel time 600.0s
    prim_seg = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=5000.0, speed_kmh=50.0,
        traversal_time_s=360.0, name="Main Road", road_reference=None, highway_type="primary"
    )
    primary = make_test_route([prim_seg], base_snap, route_id="primary", name="Main Road")

    # Alt: saves only 25s (travel time 575.0s)
    alt_seg = RouteSegment(
        edge_id="1_3_0", u=1, v=3, key=0, length_m=4791.67, speed_kmh=50.0,
        traversal_time_s=345.0, name="Side Street", road_reference=None, highway_type="secondary"
    )
    alt = make_test_route([alt_seg], base_snap, route_id="alt", name="Side Street")

    # Speeds:
    # Primary: 30 km/h -> 5000 / (30 * 1000/3600) = 600.0s
    # Alt: 30 km/h -> 4791.67 / (30 * 1000/3600) = 575.0s (saves 25.0s)
    edge_speeds = {
        (1, 2, 0): 30.0,
        (1, 3, 0): 30.0,
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    # Default config has min_time_savings_seconds = 60.0
    evaluator = PredictiveRouteEvaluator(speed_provider=provider)
    result = evaluator.evaluate_routes(primary, [alt], dep_time)

    # Alt is faster in ranked_routes
    assert result.ranked_routes[0].route_id == "alt"

    # But primary route is retained as recommendation because savings (25.0s) < 60.0s
    assert result.recommended_route_id == "primary"
    assert result.time_savings_seconds == 0.0
    assert "below the minimum recommendation threshold" in result.recommendation_reason


# =====================================================================
# 3. Tie-Breaking Behavior (Travel Time Ties)
# =====================================================================

def test_tie_breaking_behavior(base_snap):
    """
    Test tie-breaking when routes have travel times within tie_threshold_seconds:
    - Prefer shorter physical distance when distance difference > 50m.
    - Prefer primary route when distances are equal or within 50m.
    """
    # 3a: Equal travel time (300s), but Alt is shorter by 500m (> 50m)
    # Configured with min_time_savings_seconds=0.0 and tie_threshold_seconds=30.0
    prim_seg = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=5000.0, speed_kmh=60.0,
        traversal_time_s=300.0, name="Highway", road_reference=None, highway_type="trunk"
    )
    primary = make_test_route([prim_seg], base_snap, route_id="primary", name="Longer Path")

    alt_seg = RouteSegment(
        edge_id="1_3_0", u=1, v=3, key=0, length_m=4000.0, speed_kmh=48.0,
        traversal_time_s=300.0, name="Cut Through", road_reference=None, highway_type="primary"
    )
    alt_shorter = make_test_route([alt_seg], base_snap, route_id="alt_shorter", name="Shorter Path")

    edge_speeds = {
        (1, 2, 0): 60.0,   # 5000 / (60 * 1000/3600) = 300.0s
        (1, 3, 0): 48.0,   # 4000 / (48 * 1000/3600) = 300.0s
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    cfg = PredictiveEvaluatorConfig(min_time_savings_seconds=0.0, tie_threshold_seconds=30.0)
    evaluator = PredictiveRouteEvaluator(speed_provider=provider, config=cfg)

    # 3b: Equal travel times and equal distances (within 50m): primary preferred
    alt_same_dist = RouteSegment(
        edge_id="1_4_0", u=1, v=4, key=0, length_m=5020.0, speed_kmh=60.24,
        traversal_time_s=300.0, name="Parallel Path", road_reference=None, highway_type="trunk"
    )
    alt_equal = make_test_route([alt_same_dist], base_snap, route_id="alt_equal", name="Parallel Path")

    res_equal = evaluator.evaluate_routes(primary, [alt_equal], dep_time)
    assert res_equal.recommended_route_id == "primary"
    assert "distance difference is within 50.0m" in res_equal.recommendation_reason


# =====================================================================
# 4. No Alternatives Provided
# =====================================================================

def test_no_alternatives_evaluates_sole_primary(base_snap):
    """Verify evaluation when no alternative routes are provided."""
    seg = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Only Road", road_reference=None, highway_type="primary"
    )
    primary = make_test_route([seg], base_snap, route_id="sole_route", name="Only Road")

    provider = MockSpeedProvider(default_speed_kmh=40.0)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    evaluator = PredictiveRouteEvaluator(speed_provider=provider)
    res = evaluator.evaluate_routes(primary, alternative_routes=[], departure_time=dep_time)

    assert res.recommended_route_id == "sole_route"
    assert res.time_savings_seconds == 0.0
    assert len(res.alternative_routes) == 0
    assert len(res.ranked_routes) == 1
    assert "sole viable candidate" in res.recommendation_reason


# =====================================================================
# 5. Same Location / Single-Node Route Handling
# =====================================================================

def test_same_location_zero_travel_time(base_snap):
    """Verify single-node route returns zero travel time and appropriate reason."""
    single_route = Route(
        route_id="stationery",
        name="Stationary Point",
        origin_snap=base_snap,
        destination_snap=base_snap,
        node_sequence=[1],
        edge_sequence=[],
        total_distance_m=0.0,
        total_time_s=0.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )
    provider = MockSpeedProvider(default_speed_kmh=50.0)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    evaluator = PredictiveRouteEvaluator(speed_provider=provider)
    res = evaluator.evaluate_routes(single_route, departure_time=dep_time)

    assert res.recommended_route_id == "stationery"
    assert res.time_savings_seconds == 0.0
    assert res.primary_route.predicted_travel_time_seconds == 0.0
    assert "same location" in res.recommendation_reason


# =====================================================================
# 6. Unsuccessful or Invalid RoutePlanResult Fails Safely
# =====================================================================

def test_unavailable_or_unsuccessful_plan_fails_safely(base_snap):
    """Verify unsuccessful RoutePlanResults raise descriptive ValueError."""
    failed_plan = RoutePlanResult(
        success=False,
        status=RouteStatus.NO_PATH_FOUND,
        status_message="No viable path connects origin and destination",
        primary_route=None,
        alternative_routes=[],
    )
    provider = MockSpeedProvider()
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)
    evaluator = PredictiveRouteEvaluator(speed_provider=provider)

    with pytest.raises(ValueError, match="Cannot evaluate unsuccessful RoutePlanResult"):
        evaluator.evaluate_route_plan(failed_plan, dep_time)

    # Non-timezone-aware timestamp rejection
    seg = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Road", road_reference=None, highway_type="primary"
    )
    route = make_test_route([seg], base_snap)
    naive_dt = datetime(2026, 10, 21, 9, 0)  # no tzinfo

    with pytest.raises(ValueError, match="departure_time must be timezone-aware"):
        evaluator.evaluate_routes(route, departure_time=naive_dt)


# =====================================================================
# 7. Non-Safety Disclaimer & Data Provenance Integrity
# =====================================================================

def test_provenance_and_non_safety_disclaimer(base_snap):
    """
    Verify mandatory non-safety disclaimer is included verbatim
    and data provenance accurately tracks provider and KPIs.
    """
    seg = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Link", road_reference=None, highway_type="primary"
    )
    route = make_test_route([seg], base_snap)
    provider = MockSpeedProvider(default_speed_kmh=50.0)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    evaluator = PredictiveRouteEvaluator(speed_provider=provider)
    res = evaluator.evaluate_routes(route, departure_time=dep_time)

    # Mandatory disclaimer check
    expected_disclaimer = (
        "DISCLAIMER: This recommendation optimizes estimated travel time only based on simulated traffic impedance. "
        "It does NOT evaluate road safety, accident risk, pavement condition, lighting, or weather hazards, and must not "
        "be interpreted as a safety recommendation."
    )
    assert res.evaluation_disclaimer == expected_disclaimer

    # Provenance summary check
    prov = res.data_provenance
    assert prov["speed_provider"] == "MockSpeedProvider"
    assert prov["route_count_evaluated"] == 1
    assert prov["primary_route_id"] == "route_1"
    assert len(prov["routes_summary"]) == 1
    assert prov["routes_summary"][0]["route_id"] == "route_1"


# =====================================================================
# 8. Unchanged Departure Time & Non-Mutation Invariants
# =====================================================================

def test_unchanged_departure_time_and_non_mutation(base_snap):
    """
    Verify all routes evaluate against the exact same initial departure timestamp
    and that input Route objects and road graph are not mutated.
    """
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.01, longitude=77.01)
    g.add_node(3, latitude=11.02, longitude=77.02)
    g.add_edge(1, 2, key=0, edge_id="1_2_0", length_m=1000.0, free_flow_speed_kmh=60.0, highway_type="primary")
    g.add_edge(1, 3, key=0, edge_id="1_3_0", length_m=1200.0, free_flow_speed_kmh=60.0, highway_type="primary")
    road_graph = RoadNetworkGraph(graph=g)

    seg_prim = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Route A", road_reference=None, highway_type="primary"
    )
    seg_alt = RouteSegment(
        edge_id="1_3_0", u=1, v=3, key=0, length_m=1200.0, speed_kmh=60.0,
        traversal_time_s=72.0, name="Route B", road_reference=None, highway_type="primary"
    )
    prim_route = make_test_route([seg_prim], base_snap, route_id="prim")
    alt_route = make_test_route([seg_alt], base_snap, route_id="alt")

    prim_backup = copy.deepcopy(prim_route)
    alt_backup = copy.deepcopy(alt_route)

    dep_time = datetime(2026, 10, 21, 10, 15, tzinfo=timezone.utc)
    provider = MockSpeedProvider(default_speed_kmh=45.0)

    evaluator = PredictiveRouteEvaluator(speed_provider=provider, road_graph=road_graph)
    res = evaluator.evaluate_routes(prim_route, [alt_route], dep_time)

    # 1. Departure time invariant
    assert res.departure_time == dep_time
    assert res.primary_route.departure_time == dep_time
    assert res.alternative_routes[0].departure_time == dep_time
    assert res.primary_route.evaluated_segments[0].arrival_time == dep_time
    assert res.alternative_routes[0].evaluated_segments[0].arrival_time == dep_time

    # 2. Non-mutation invariant
    assert prim_route.model_dump() == prim_backup.model_dump()
    assert alt_route.model_dump() == alt_backup.model_dump()
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


# =====================================================================
# 9. Facade Integration with RoutePlanResult & JSON Summary
# =====================================================================

def test_evaluate_route_plan_facade_and_summary(base_snap):
    """
    Verify PredictiveRouteEvaluator.evaluate_route_plan integrates with P6 RoutePlanResult,
    supports classmethod execution, and produces JSON-serializable summary.
    """
    seg_prim = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=2000.0, speed_kmh=60.0,
        traversal_time_s=120.0, name="Corridor 1", road_reference="NH-544", highway_type="trunk"
    )
    seg_alt = RouteSegment(
        edge_id="1_3_0", u=1, v=3, key=0, length_m=1800.0, speed_kmh=60.0,
        traversal_time_s=108.0, name="Corridor 2", road_reference=None, highway_type="primary"
    )
    prim_route = make_test_route([seg_prim], base_snap, route_id="prim", name="Corridor 1")
    alt_route = make_test_route([seg_alt], base_snap, route_id="alt", name="Corridor 2")

    plan = RoutePlanResult(
        success=True,
        status=RouteStatus.SUCCESS,
        status_message="Path found",
        primary_route=prim_route,
        alternative_routes=[alt_route],
    )

    # Speeds: Primary 20 km/h (360s), Alt 50 km/h (129.6s) -> saves 230.4s (3.8 min)
    edge_speeds = {
        (1, 2, 0): 20.0,
        (1, 3, 0): 50.0,
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 8, 30, tzinfo=timezone.utc)

    # Execute via classmethod
    res = PredictiveRouteEvaluator.evaluate_plan(
        plan_result=plan,
        departure_time=dep_time,
        speed_provider=provider,
    )

    assert res.recommended_route_id == "alt"
    assert res.time_savings_minutes > 3.0

    # Summary verification
    summary = res.summary()
    assert isinstance(summary, dict)
    assert summary["recommended_route_id"] == "alt"
    assert summary["time_savings_seconds"] > 200.0
    assert len(summary["ranked_routes"]) == 2
    assert summary["ranked_routes"][0]["is_recommended"] is True
    assert summary["ranked_routes"][1]["is_recommended"] is False
    assert "DISCLAIMER" in summary["evaluation_disclaimer"]
