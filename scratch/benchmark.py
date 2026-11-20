"""Performance benchmark script for TrafficRoute AI P1 milestone."""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
import numpy as np

from src.network.extraction import load_or_extract_pilot_network
from src.network.snapping import NetworkSpatialIndexer
from src.network.routing import RoutingEngine, CostMetric
from src.network.alternatives import AlternativeRouteGenerator
from src.network.geometry import RouteGeometryBuilder


def run_benchmarks():
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Graph Loading Benchmark
    t0 = time.time()
    road_graph = load_or_extract_pilot_network(force_reextract=False)
    load_time_s = time.time() - t0
    mem_after_graph_mb = process.memory_info().rss / (1024 * 1024)
    graph_mem_mb = mem_after_graph_mb - mem_before_mb

    # 2. Spatial Indexer Building Benchmark
    t0 = time.time()
    indexer = NetworkSpatialIndexer(road_graph)
    indexer_build_time_s = time.time() - t0

    # 3. Location Snapping Benchmark (100 random coordinate queries)
    lats = np.random.uniform(10.98, 11.20, size=100)
    lons = np.random.uniform(76.95, 77.25, size=100)

    snap_times = []
    for lat, lon in zip(lats, lons):
        t0 = time.perf_counter()
        indexer.snap_coordinate(lat, lon, snap_to_edge=True)
        snap_times.append((time.perf_counter() - t0) * 1000.0)  # in ms

    # 4. Routing Engine Latency (Coimbatore Railway Station -> Avinashi)
    engine = RoutingEngine(road_graph, spatial_indexer=indexer)
    cbe_lat, cbe_lon = 10.9990, 76.9630
    avi_lat, avi_lon = 11.1930, 77.2690

    # Warmup
    engine.calculate_route(cbe_lat, cbe_lon, avi_lat, avi_lon)

    route_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        route = engine.calculate_route(cbe_lat, cbe_lon, avi_lat, avi_lon, metric=CostMetric.FREE_FLOW_TIME)
        route_times.append((time.perf_counter() - t0) * 1000.0)

    # 5. Alternative Route Generation (3 diverse routes)
    alt_gen = AlternativeRouteGenerator(engine, max_shared_ratio=0.80)
    alt_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        alt_routes = alt_gen.generate_alternative_routes(cbe_lat, cbe_lon, avi_lat, avi_lon, k_routes=3)
        alt_times.append((time.perf_counter() - t0) * 1000.0)

    # 6. GeoJSON Generation Latency
    geom_builder = RouteGeometryBuilder(road_graph)
    t0 = time.perf_counter()
    geojson_data = geom_builder.build_route_geojson(route)
    geojson_time_ms = (time.perf_counter() - t0) * 1000.0

    total_mem_mb = process.memory_info().rss / (1024 * 1024)

    print("=" * 60)
    print("TRAFFICROUTE AI — P1 PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print(f"Network Scale: {road_graph.node_count:,} nodes | {road_graph.edge_count:,} edges")
    print(f"Binary Graph Load Time: {load_time_s:.3f} s")
    print(f"Spatial KD-Tree Index Build: {indexer_build_time_s:.3f} s")
    print(f"Memory Usage (Graph in RAM): {graph_mem_mb:.1f} MB (Total Process: {total_mem_mb:.1f} MB)")
    print(f"Location Snapping Latency (Mean): {np.mean(snap_times):.2f} ms (p95: {np.percentile(snap_times, 95):.2f} ms)")
    print(f"Single Route Calculation Latency: {np.mean(route_times):.2f} ms")
    print(f"3-Alternative Route Generation Latency: {np.mean(alt_times):.2f} ms")
    print(f"Map-Ready GeoJSON Serialization Latency: {geojson_time_ms:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
