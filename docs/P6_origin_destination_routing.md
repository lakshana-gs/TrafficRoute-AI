# P6 — Origin–Destination Route Engine Technical Specification

**Milestone:** P6 — Origin–Destination Route Engine  
**Packages:** `src.network.routing`, `src.network.alternatives`  
**Scope:** Unified Point-to-Point Routing, Graph Pathfinding, Defensive Validation, and Diverse Multi-Path Corridor Engine  
**Status:** COMPLETE (All P6.1–P6.5 sub-milestones verified; 153/153 full-suite tests passed)

---

## 1. Executive Summary & Objective

Milestone **P6** delivers the production-ready static origin–destination (OD) routing layer for the TrafficRoute AI road network graph (specifically validated on the Coimbatore Metropolitan Area $\leftrightarrow$ Avinashi NH-544 pilot corridor, comprising 251,867 nodes and 549,919 directed edges).

Prior milestones provided:
- **P1:** Road network extraction, OpenStreetMap attribute standardization, and binary graph serialization.
- **P2:** Real traffic probe ingestion, cleaning, temporal bucketing, and spatial snapping.
- **P3 & P4/P4.1:** Historical baseline speed prediction and DCRNN spatio-temporal deep learning models.
- **P5:** Segment-level congestion classification and delay estimation primitives.

**P6 connects network graph topology and spatial indexers to robust pathfinding:**
1. Direct node-to-node pathfinding bypassing coordinate snapping when explicit vertex IDs are available (**P6.1**).
2. Defensive input validation and non-fabricating edge-cost evaluation handling corrupt/non-finite attributes (**P6.2**).
3. Strictly validated structured route results enforcing physical invariants, segment/node connectivity, and native JSON summary serialization (**P6.3**).
4. Diverse multi-corridor alternative generation with safe edge penalization and length-weighted overlap filtering (**P6.4**).
5. A unified route-planning facade with comprehensive diagnostic outcome classification (**P6.5**).

---

## 2. Supported Routing Entry Points

The routing engine exposes two tiers of APIs: a high-level diagnostic planning facade (`plan_route` / `plan_route_by_nodes`), and lower-level direct routing methods (`calculate_route` / `calculate_route_by_nodes`).

### 2.1 Unified Diagnostic Planning Facade (Recommended)

```python
engine = RoutingEngine(road_network_graph)

# 1. Geographic coordinate planning (WGS84)
result: RoutePlanResult = engine.plan_route(
    origin_lat=11.00,
    origin_lon=76.96,
    destination_lat=11.19,
    destination_lon=77.26,
    metric=CostMetric.FREE_FLOW_TIME,
    departure_time=datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc),
    num_alternatives=2,
    route_name="Fastest Route",
    route_id="primary_route",
)

# 2. Direct graph node-ID planning
node_result: RoutePlanResult = engine.plan_route_by_nodes(
    origin_node_id=101,
    destination_node_id=505,
    metric=CostMetric.FREE_FLOW_TIME,
    num_alternatives=1,
)
```

- **Output Contract:** Always returns a [`RoutePlanResult`](#32-routeplanresult) with a diagnostic [`RouteStatus`](#31-routestatus) enum, human-readable `status_message`, `primary_route`, and non-duplicated `alternative_routes`.
- **Alternatives:** When `num_alternatives > 0`, lazily invokes `AlternativeRouteGenerator` to discover structurally diverse corridors.

### 2.2 Direct Optimal Route APIs

```python
# Direct coordinate routing: returns Optional[Route] (None if unroutable/invalid)
route: Optional[Route] = engine.calculate_route(origin_lat, origin_lon, dest_lat, dest_lon, metric)

# Direct node-ID routing: returns Optional[Route]
node_route: Optional[Route] = engine.calculate_route_by_nodes(origin_node_id, dest_node_id, metric)
```

### 2.3 Standalone Multi-Corridor Generator

```python
alt_gen = AlternativeRouteGenerator(engine, max_shared_ratio=0.75, penalty_factor=0.40)

# Coordinate-based alternatives
routes: List[Route] = alt_gen.generate_alternative_routes(origin_lat, origin_lon, dest_lat, dest_lon, k_routes=3)

# Node-ID-based alternatives
node_routes: List[Route] = alt_gen.generate_alternative_routes_by_nodes(origin_node_id, dest_node_id, k_routes=3)
```

---

## 3. Schemas & Data Contracts

All data contracts are implemented using Pydantic (V2) models with strict validation.

### 3.1 `RouteStatus` Enum

```python
class RouteStatus(str, Enum):
    SUCCESS = "success"            # Usable primary route determined
    SAME_LOCATION = "same_location"# Origin and destination represent the same location (0-distance route)
    SNAP_FAILED = "snap_failed"    # Valid coordinate failed spatial snapping (e.g. >1500m off-network)
    NO_PATH_FOUND = "no_path_found"# Valid snapped endpoints but topologically disconnected in graph
    INVALID_INPUT = "invalid_input"# Malformed coordinates, booleans, non-finite values, or invalid node IDs
```

### 3.2 `RoutePlanResult`

```python
class RoutePlanResult(BaseModel):
    success: bool                          # True for SUCCESS and SAME_LOCATION, False otherwise
    status: RouteStatus                    # Diagnostic outcome classification
    status_message: str                    # Detailed explanation of status
    primary_route: Optional[Route] = None  # Primary optimal route if found
    alternative_routes: List[Route] = []   # Non-duplicated diverse alternatives
    summary: Optional[Dict[str, Any]]      # JSON-serializable KPI summary of primary route
```

### 3.3 `Route` Model & Invariants

```python
class Route(BaseModel):
    route_id: str
    name: str = "Main Route"
    origin_snap: SnapResult
    destination_snap: SnapResult
    node_sequence: List[int]               # Length >= 1
    edge_sequence: List[RouteSegment]
    total_distance_m: float                # Finite, non-negative
    total_time_s: float                    # Finite, non-negative
    cost_metric: CostMetric
    departure_time: Optional[datetime] = None

    @property
    def total_distance_km(self) -> float: ...
    @property
    def total_time_minutes(self) -> float: ...
    @property
    def segment_count(self) -> int: ...
    @property
    def average_speed_kmh(self) -> float: ... # 0.0 when time <= 0
    def summary(self) -> Dict[str, Any]: ...
```

**Validated Route Invariants:**
1. **Sequence Length:** `len(edge_sequence) == len(node_sequence) - 1`.
2. **Topological Connectivity:** For each segment index $i$, `seg.u == node_sequence[i]` and `seg.v == node_sequence[i+1]`.
3. **Single-Node Route:** When `len(node_sequence) == 1`, `edge_sequence == []`, `total_distance_m == 0.0`, and `total_time_s == 0.0`.
4. **Distance Consistency:** `abs(total_distance_m - sum(seg.length_m)) <= max(1.0, 0.02 * N)`.
5. **Time Consistency:** For `DISTANCE` and `FREE_FLOW_TIME`, `abs(total_time_s - sum(seg.traversal_time_s)) <= max(1.0, 0.02 * N)`. Deliberately decoupled for `PREDICTED_TIME` so dynamic congestion delay is not conflated with static free-flow sum.

### 3.4 `RouteSegment`

```python
class RouteSegment(BaseModel):
    u: int
    v: int
    key: int = 0
    edge_id: str
    name: Optional[str] = None
    road_reference: Optional[str] = None
    highway_type: str
    length_m: float          # Strictly positive (> 0.0), finite
    speed_kmh: float         # Strictly positive (> 0.0), finite
    traversal_time_s: float  # Non-negative (>= 0.0), finite
```

---

## 4. Cost Metrics & Edge Evaluation Flow

The engine supports three optimization metrics defined in `CostMetric`:

| CostMetric | Evaluation Formula | Edge Requirement |
|---|---|---|
| `DISTANCE` | $\text{cost} = \text{length\_m}$ | Requires finite, positive `length_m`. |
| `FREE_FLOW_TIME` | $\text{cost} = \frac{\text{length\_m}}{\text{speed\_mps}} = \frac{\text{length\_m}}{\text{speed\_kmh} \times (1000 / 3600)}$ | Uses `free_flow_time_s` if precomputed, otherwise computes from `free_flow_speed_kmh` or OSM hierarchy fallback. |
| `PREDICTED_TIME` | Static free-flow fallback; dynamic ML hook reserved for live integration. | Same validation as free-flow time. |

### Defensive Edge-Cost Rules:
- **No Clamping / No Fabrication:** An edge with `length_m <= 0`, `NaN`, `inf`, or `None` returns `None` as its weight. NetworkX automatically excludes it from the pathfinding search graph.
- **Hierarchy Fallback:** If `free_flow_speed_kmh` is missing or invalid on an edge with valid length, the engine applies the standardized OSM hierarchy default (`motorway: 100`, `trunk: 80`, `primary: 60`, `secondary: 50`, `tertiary: 40`, `residential: 25`, other: `30` km/h) rather than failing the route.
- **Parallel Edge Resolution:** On multigraphs, the edge key minimizing valid cost is evaluated for pathfinding and consistently selected during segment assembly.

---

## 5. Diagnostic Semantics & Error Classification

`RoutingEngine.plan_route` and `plan_route_by_nodes` classify outcomes deterministically:

```
[Query Received]
       │
       ├─► Invalid coordinates / node IDs / metric / num_alternatives?
       │     └─► RouteStatus.INVALID_INPUT (success=False)
       │
       ├─► Coordinate distance to nearest road > 1500m?
       │     └─► RouteStatus.SNAP_FAILED (success=False)
       │
       ├─► Origin and destination resolve to the same node?
       │     └─► RouteStatus.SAME_LOCATION (success=True, primary_route=0m route)
       │
       ├─► Dijkstra path search raises NetworkXNoPath?
       │     └─► RouteStatus.NO_PATH_FOUND (success=False)
       │
       └─► Route found and segments assembled?
             └─► RouteStatus.SUCCESS (success=True, primary_route=Route, alternatives=[...])
```

---

## 6. Multi-Corridor Alternative Generation & Diversity

Alternative route exploration implements iterative edge-cost penalization:

### 6.1 Overlap Metric Formulation
Given candidate route $R_a$ and accepted route $R_b$, length-weighted shared edge overlap ratio is:
$$\text{overlap}(R_a, R_b) = \frac{\sum_{e \in R_a \cap R_b} L(e)}{\min(L(R_a), L(R_b))}$$
where $L(R)$ is total route length in meters.

### 6.2 Iterative Penalization Algorithm
1. First candidate found via shortest path is accepted as the primary fastest route ($k=1$).
2. For each subsequent iteration:
   - Traversed edges $(u, v)$ in accepted routes receive an increased cost multiplier:
     $$c_{\text{penalized}}(u, v) = c_{\text{base}}(u, v) \times \left(1.0 + \text{penalty\_factor} \times \text{times\_used}(u, v)\right)$$
   - Invalid edges returning `None` are never multiplied or compared.
   - NetworkX finds a candidate path minimizing penalized cost.
   - Segments are verified via `RoutingEngine._assemble_route_segments`.
   - The candidate is accepted if and only if $\text{overlap}(R_{\text{candidate}}, R_{\text{accepted}}) \le \text{max\_shared\_ratio}$ against **all** previously accepted routes.
   - Iteration stops when $k$ routes are found or `max_iterations = max(10, k_routes * 5)` is reached.
3. Returned alternatives are sorted by primary cost (`DISTANCE` or `FREE_FLOW_TIME`).
4. In `plan_route`, the primary route is separated and only diverse secondary routes are returned in `alternative_routes`.

---

## 7. Verification Results & Regression Baselines

All test suites were verified under Python 3.13.7 and pytest 9.1.1:

| Verification Step | Test Suite | Tests Passed | Duration |
|---|---|---|---|
| **Step 1** | `tests/p6/test_facade.py` | 13 / 13 | 1.44s |
| **Step 2** | `tests/p6/test_alternatives.py` | 11 / 11 | 0.79s |
| **Step 3** | `tests/p6/test_structured_results.py` | 18 / 18 | 0.90s |
| **Step 4** | `tests/p6/test_defensive_validation.py` | 17 / 17 | 0.75s |
| **Step 5** | `tests/p6/test_route_by_nodes.py` | 11 / 11 | 0.78s |
| **Step 6** | `tests/test_routing.py` | 2 / 2 | 0.81s |
| **Step 7** | `tests/test_alternatives.py` | 1 / 1 | 0.70s |
| **Step 8** | `tests/test_integration.py` (Coimbatore $\to$ Avinashi Pilot Corridor) | 1 / 1 | 12.23s |
| **Step 9** | Full Test Suite (`pytest -v`) | **153 / 153** | **64.50s** |

### Verified File Checksums (Preservation Guarantee):
- `data/processed/coimbatore_avinashi_network.bin`: `7EB09D976159987CB356201BC1E8884C4BA9EB39D1A716A67A80C5A84BFC9C30`
- `artifacts/models/p4_dcrnn_best.pt`: `BF1A9B90C795C890996B423A3DD0CE4B5A8A6C519DC44E09D27D843925CC8410`
- `src/network/graph.py`: `274AA01AFBE0876DA53420B9B7D21B23F9F7C5A5D11FFFB0FE7A9A4EAD690D70`
- `src/network/snapping.py`: `D909B5BE850A89759948406BAA80A4D57C451FD9613C773B67C20C78AF5B8F46`
- `src/traffic/congestion.py`: `AC72A6E11520BA97940F427276B429AB0EA634D261BFD32697798BB474FFB905`
