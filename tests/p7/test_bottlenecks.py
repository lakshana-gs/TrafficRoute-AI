"""Unit tests for P7.2: Route Congestion Scoring & Bottleneck Attribution."""

from datetime import datetime, timezone
import networkx as nx
import pytest

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RouteSegment, RoutingEngine
from src.network.snapping import SnapResult
from src.traffic.congestion import CongestionLevel, CongestionStatus
from src.traffic.predictive_evaluation import (
    EvaluatedRoute,
    EvaluatedRouteSegment,
    MockSpeedProvider,
    RouteBottleneck,
    extract_route_bottlenecks,
    forward_propagate_route,
    generate_bottleneck_display_title,
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


def make_route(segments, base_snap, route_id="test_route", name="Test Route"):
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
# 1. Signed Delay Sum, Gross Positive Delay & Decomposition Invariant
# =====================================================================

def test_signed_delay_sum_and_gross_positive_delay_decomposition(base_snap):
    """
    Verify net signed route delay equals sum of segment signed delays,
    gross positive delay sums strictly positive contributions,
    and net delay satisfies: Delta T_net = Delta T_pos - Delta T_neg <= Delta T_pos.
    """
    # 4 segments of 1000m each, free-flow 60 km/h (ff_time = 60s per segment)
    seg1 = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Fast Link", road_reference=None, highway_type="trunk"
    )
    seg2 = RouteSegment(
        edge_id="2_3_0", u=2, v=3, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Congested Link A", road_reference=None, highway_type="primary"
    )
    seg3 = RouteSegment(
        edge_id="3_4_0", u=3, v=4, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Congested Link B", road_reference=None, highway_type="primary"
    )
    seg4 = RouteSegment(
        edge_id="4_5_0", u=4, v=5, key=0, length_m=1000.0, speed_kmh=60.0,
        traversal_time_s=60.0, name="Neutral Link", road_reference=None, highway_type="secondary"
    )
    route = make_route([seg1, seg2, seg3, seg4], base_snap)

    # Dynamic speeds:
    # Seg 1: 90 km/h -> trav = 40.0s -> delay = -20.0s (faster than free-flow)
    # Seg 2: 20 km/h -> trav = 180.0s -> delay = +120.0s (congested)
    # Seg 3: 30 km/h -> trav = 120.0s -> delay = +60.0s (congested)
    # Seg 4: 60 km/h -> trav = 60.0s -> delay = 0.0s (free-flow)
    edge_speeds = {
        (1, 2, 0): 90.0,
        (2, 3, 0): 20.0,
        (3, 4, 0): 30.0,
        (4, 5, 0): 60.0,
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    eval_route = forward_propagate_route(route, dep_time, provider)

    # 1. Net signed delay sum
    expected_net_delay = -20.0 + 120.0 + 60.0 + 0.0  # 160.0s
    assert pytest.approx(eval_route.total_delay_seconds, abs=0.1) == expected_net_delay

    # 2. Gross positive delay sum
    expected_pos_delay = 0.0 + 120.0 + 60.0 + 0.0     # 180.0s
    assert pytest.approx(eval_route.total_positive_delay_seconds, abs=0.1) == expected_pos_delay

    # 3. Decomposition invariant
    gross_neg_delay = eval_route.data_provenance["gross_negative_delay_seconds"]
    assert pytest.approx(gross_neg_delay, abs=0.1) == 20.0
    assert pytest.approx(eval_route.total_delay_seconds, abs=0.1) == (
        eval_route.total_positive_delay_seconds - gross_neg_delay
    )
    assert eval_route.total_delay_seconds < eval_route.total_positive_delay_seconds

    # 4. Bottleneck sum is bounded by gross positive delay (NOT net signed delay)
    bottleneck_delay_sum = sum(b.delay_seconds for b in eval_route.bottlenecks)
    assert bottleneck_delay_sum <= eval_route.total_positive_delay_seconds
    assert bottleneck_delay_sum > eval_route.total_delay_seconds  # 180.0 > 160.0


# =====================================================================
# 2. Non-Double-Counting Invariant & Disjoint Segment Attribution
# =====================================================================

def test_bottleneck_non_double_counting_invariant(base_snap):
    """
    Verify that every segment's positive delay is attributed at most once,
    bottlenecks partition qualifying segments into disjoint subsets,
    and total bottleneck delay <= total positive route delay.
    """
    segs = [
        RouteSegment(edge_id=f"{i}_{i+1}_0", u=i, v=i+1, key=0, length_m=500.0, speed_kmh=50.0,
                     traversal_time_s=36.0, name=f"Street {i}", road_reference=None, highway_type="primary")
        for i in range(1, 6)
    ]
    route = make_route(segs, base_snap)

    # All edges have heavy delay (15 km/h -> delay = 120s - 36s = 84s per segment)
    provider = MockSpeedProvider(default_speed_kmh=15.0)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    eval_route = forward_propagate_route(route, dep_time, provider)
    assert len(eval_route.bottlenecks) == 5

    total_bottleneck_delay = sum(b.delay_seconds for b in eval_route.bottlenecks)
    assert pytest.approx(total_bottleneck_delay, abs=0.1) == eval_route.total_positive_delay_seconds

    # Verify disjoint attribution: each segment is attributed to exactly one bottleneck
    attributed_segments = set()
    for b in eval_route.bottlenecks:
        # segment_id must not overlap
        assert b.segment_id not in attributed_segments
        attributed_segments.add(b.segment_id)

    assert len(attributed_segments) == 5


# =====================================================================
# 3. Continuous Corridor Clustering & Deduplication
# =====================================================================

def test_corridor_clustering_consecutive_same_road(base_snap):
    """
    Verify consecutive qualifying segments with the same road name
    are clustered into a single named corridor bottleneck, while non-consecutive
    segments with the same name remain separate.
    """
    # Route layout:
    # Seg 0: Avinashi Road (qualifies)
    # Seg 1: Avinashi Road (qualifies)
    # Seg 2: Avinashi Road (qualifies)
    # Seg 3: Trichy Road (qualifies)
    # Seg 4: Avinashi Road (qualifies, but separated by Trichy Road)
    segs = [
        RouteSegment(edge_id="1_2_0", u=1, v=2, key=0, length_m=600.0, speed_kmh=60.0,
                     traversal_time_s=36.0, name="Avinashi Road", road_reference="NH-544", highway_type="primary"),
        RouteSegment(edge_id="2_3_0", u=2, v=3, key=0, length_m=400.0, speed_kmh=60.0,
                     traversal_time_s=24.0, name="Avinashi Road", road_reference="NH-544", highway_type="primary"),
        RouteSegment(edge_id="3_4_0", u=3, v=4, key=0, length_m=500.0, speed_kmh=60.0,
                     traversal_time_s=30.0, name="Avinashi Road", road_reference="NH-544", highway_type="primary"),
        RouteSegment(edge_id="4_5_0", u=4, v=5, key=0, length_m=800.0, speed_kmh=60.0,
                     traversal_time_s=48.0, name="Trichy Road", road_reference="SH-15", highway_type="primary"),
        RouteSegment(edge_id="5_6_0", u=5, v=6, key=0, length_m=700.0, speed_kmh=60.0,
                     traversal_time_s=42.0, name="Avinashi Road", road_reference="NH-544", highway_type="primary"),
    ]
    route = make_route(segs, base_snap)

    # Congested speeds (20 km/h across all edges -> delay > 30s on each)
    provider = MockSpeedProvider(default_speed_kmh=20.0)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    eval_route = forward_propagate_route(route, dep_time, provider)

    # Segments 0, 1, 2 cluster into 1 bottleneck.
    # Segment 3 is Trichy Road (1 bottleneck).
    # Segment 4 is Avinashi Road (1 bottleneck, not merged with 0-2 because separated by Seg 3).
    # Total bottlenecks = 3
    assert len(eval_route.bottlenecks) == 3

    # Clustered corridor checks
    corridor_b = next(b for b in eval_route.bottlenecks if b.u == 1 and b.v == 4)
    assert corridor_b.road_name == "Avinashi Road"
    assert corridor_b.road_reference == "NH-544"
    assert corridor_b.segment_id == "1_2_0...3_4_0"
    assert corridor_b.arrival_time == dep_time

    # Clustered delay must equal sum of constituent delays
    seg_0_delay = eval_route.evaluated_segments[0].signed_delay_seconds
    seg_1_delay = eval_route.evaluated_segments[1].signed_delay_seconds
    seg_2_delay = eval_route.evaluated_segments[2].signed_delay_seconds
    assert pytest.approx(corridor_b.delay_seconds, abs=0.01) == (seg_0_delay + seg_1_delay + seg_2_delay)

    # Ranking check: Clustered corridor (1500m) should have highest delay, so rank 1
    assert corridor_b.rank == 1
    assert "Avinashi Road (NH-544)" in corridor_b.display_title


# =====================================================================
# 4. Stopped Traffic Elevation to Critical Severe Bottleneck
# =====================================================================

def test_stopped_traffic_elevation_to_bottleneck(base_snap):
    """
    Verify that a segment experiencing stopped traffic (0.0 km/h) is
    elevated to a critical bottleneck with CongestionLevel.SEVERE
    and CongestionStatus.STOPPED, using the crawl speed floor.
    """
    seg1 = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=300.0, speed_kmh=50.0,
        traversal_time_s=21.6, name="Gridlock Link", road_reference=None, highway_type="primary"
    )
    seg2 = RouteSegment(
        edge_id="2_3_0", u=2, v=3, key=0, length_m=500.0, speed_kmh=50.0,
        traversal_time_s=36.0, name="Moving Link", road_reference=None, highway_type="primary"
    )
    route = make_route([seg1, seg2], base_snap)

    edge_speeds = {
        (1, 2, 0): 0.0,   # Stopped traffic
        (2, 3, 0): 50.0,  # Free flow
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    eval_route = forward_propagate_route(route, dep_time, provider, crawl_speed_kmh=3.0)

    assert len(eval_route.bottlenecks) == 1
    bn = eval_route.bottlenecks[0]
    assert bn.segment_id == "1_2_0"
    assert bn.congestion_level == CongestionLevel.SEVERE
    assert bn.observed_speed_kmh == 3.0
    assert bn.delay_seconds > 0.0

    # Evaluated segment must reflect stopped status and bottleneck flag
    seg_res = eval_route.evaluated_segments[0]
    assert seg_res.is_bottleneck is True
    assert seg_res.congestion_result.congestion_status == CongestionStatus.STOPPED
    assert seg_res.congestion_result.congestion_level == CongestionLevel.SEVERE


# =====================================================================
# 5. Missing Road Names & Display Title Formatting
# =====================================================================

def test_missing_road_names_and_display_title_formatting(base_snap):
    """
    Test display title generation rules across named, referenced, and unnamed corridors,
    and verify consecutive unnamed links remain distinct.
    """
    # 5a: Both name and reference present
    t1 = generate_bottleneck_display_title("Avinashi Road", "NH-544", "trunk", 100, 4.2)
    assert t1 == "Avinashi Road (NH-544): +4.2m delay"

    # 5b: Only road name present
    t2 = generate_bottleneck_display_title("Mettupalayam Road", None, "primary", 200, 1.5)
    assert t2 == "Mettupalayam Road: +1.5m delay"

    # 5c: Only road reference present
    t3 = generate_bottleneck_display_title(None, "SH-172", "secondary", 300, 2.0)
    assert t3 == "SH-172: +2.0m delay"

    # 5d: Unnamed corridor
    t4 = generate_bottleneck_display_title(None, None, "primary_link", 401, 1.25)
    assert t4 == "Primary Link corridor near Node 401: +1.2m delay"

    # 5e: Consecutive unnamed segments without reference remain distinct
    segs = [
        RouteSegment(edge_id="10_11_0", u=10, v=11, key=0, length_m=500.0, speed_kmh=50.0,
                     traversal_time_s=36.0, name=None, road_reference=None, highway_type="residential"),
        RouteSegment(edge_id="11_12_0", u=11, v=12, key=0, length_m=500.0, speed_kmh=50.0,
                     traversal_time_s=36.0, name=None, road_reference=None, highway_type="residential"),
    ]
    route = make_route(segs, base_snap)
    provider = MockSpeedProvider(default_speed_kmh=15.0)  # heavy delay
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    eval_route = forward_propagate_route(route, dep_time, provider)

    # Unnamed consecutive links without reference must NOT merge into ambiguous cluster
    assert len(eval_route.bottlenecks) == 2
    assert eval_route.bottlenecks[0].display_title.startswith("Residential corridor near Node")
    assert eval_route.bottlenecks[1].display_title.startswith("Residential corridor near Node")


# =====================================================================
# 6. Routes with No Positive-Delay Bottlenecks
# =====================================================================

def test_routes_with_no_positive_delay_bottlenecks(base_snap):
    """
    Verify routes with free flow, sub-threshold delay, or zero distance
    return empty bottleneck lists and valid non-negative metrics.
    """
    # 6a: Free-flow route (speeds >= free-flow speed)
    segs = [
        RouteSegment(edge_id="1_2_0", u=1, v=2, key=0, length_m=600.0, speed_kmh=60.0,
                     traversal_time_s=36.0, name="Smooth Highway", road_reference=None, highway_type="trunk"),
        RouteSegment(edge_id="2_3_0", u=2, v=3, key=0, length_m=400.0, speed_kmh=40.0,
                     traversal_time_s=36.0, name="Smooth Avenue", road_reference=None, highway_type="primary"),
    ]
    route = make_route(segs, base_snap)
    provider = MockSpeedProvider(default_speed_kmh=80.0)  # faster than free-flow
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    eval_route = forward_propagate_route(route, dep_time, provider)
    assert eval_route.bottlenecks == []
    assert eval_route.total_positive_delay_seconds == 0.0
    assert eval_route.total_delay_seconds < 0.0

    # 6b: Sub-threshold delay route (delay < 30.0s threshold)
    # Length: 300m, free-flow: 60 km/h (18.0s). Speed: 30 km/h (36.0s). Delay: 18.0s < 30s.
    seg_minor = RouteSegment(
        edge_id="1_2_0", u=1, v=2, key=0, length_m=300.0, speed_kmh=60.0,
        traversal_time_s=18.0, name="Minor Queue Link", road_reference=None, highway_type="primary"
    )
    route_minor = make_route([seg_minor], base_snap)
    provider_minor = MockSpeedProvider(default_speed_kmh=30.0)
    eval_minor = forward_propagate_route(route_minor, dep_time, provider_minor, min_bottleneck_delay_seconds=30.0)
    assert eval_minor.bottlenecks == []
    assert eval_minor.total_positive_delay_seconds == pytest.approx(18.0, abs=0.1)

    # 6c: Single-node route (zero distance/delay)
    single_route = Route(
        route_id="single_node",
        name="Zero Distance",
        origin_snap=base_snap,
        destination_snap=base_snap,
        node_sequence=[1],
        edge_sequence=[],
        total_distance_m=0.0,
        total_time_s=0.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )
    eval_single = forward_propagate_route(single_route, dep_time, provider)
    assert eval_single.bottlenecks == []
    assert eval_single.total_positive_delay_seconds == 0.0
    assert eval_single.total_delay_seconds == 0.0


# =====================================================================
# 7. Ranked Bottlenecks and max_bottlenecks Limit
# =====================================================================

def test_ranked_bottlenecks_and_max_limit(base_snap):
    """
    Verify bottlenecks are ranked strictly descending by delay contribution
    and max_bottlenecks limits the surfaced output while updating segment flags.
    """
    segs = [
        RouteSegment(edge_id="1_2_0", u=1, v=2, key=0, length_m=1000.0, speed_kmh=60.0,
                     traversal_time_s=60.0, name="Corridor 1", road_reference=None, highway_type="primary"),
        RouteSegment(edge_id="2_3_0", u=2, v=3, key=0, length_m=1000.0, speed_kmh=60.0,
                     traversal_time_s=60.0, name="Corridor 2", road_reference=None, highway_type="primary"),
        RouteSegment(edge_id="3_4_0", u=3, v=4, key=0, length_m=1000.0, speed_kmh=60.0,
                     traversal_time_s=60.0, name="Corridor 3", road_reference=None, highway_type="primary"),
    ]
    route = make_route(segs, base_snap)

    # Edge delays:
    # Seg 1: 15 km/h -> trav = 240s -> delay = 180s
    # Seg 2: 25 km/h -> trav = 144s -> delay = 84s
    # Seg 3: 20 km/h -> trav = 180s -> delay = 120s
    edge_speeds = {
        (1, 2, 0): 15.0,
        (2, 3, 0): 25.0,
        (3, 4, 0): 20.0,
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)

    # Test full ranking without limit
    eval_all = forward_propagate_route(route, dep_time, provider)
    assert len(eval_all.bottlenecks) == 3
    assert eval_all.bottlenecks[0].rank == 1
    assert eval_all.bottlenecks[0].segment_id == "1_2_0"
    assert eval_all.bottlenecks[0].delay_seconds == pytest.approx(180.0, abs=0.1)

    assert eval_all.bottlenecks[1].rank == 2
    assert eval_all.bottlenecks[1].segment_id == "3_4_0"
    assert eval_all.bottlenecks[1].delay_seconds == pytest.approx(120.0, abs=0.1)

    assert eval_all.bottlenecks[2].rank == 3
    assert eval_all.bottlenecks[2].segment_id == "2_3_0"
    assert eval_all.bottlenecks[2].delay_seconds == pytest.approx(84.0, abs=0.1)

    # Test truncation with max_bottlenecks=2
    eval_top2 = forward_propagate_route(route, dep_time, provider, max_bottlenecks=2)
    assert len(eval_top2.bottlenecks) == 2
    assert eval_top2.bottlenecks[0].segment_id == "1_2_0"
    assert eval_top2.bottlenecks[1].segment_id == "3_4_0"

    # Verify is_bottleneck flags on evaluated segments
    # Top 2 segments must be True, 3rd (surpassed) must be False
    seg_by_id = {s.segment.edge_id: s for s in eval_top2.evaluated_segments}
    assert seg_by_id["1_2_0"].is_bottleneck is True
    assert seg_by_id["3_4_0"].is_bottleneck is True
    assert seg_by_id["2_3_0"].is_bottleneck is False
