# P7 — Predictive Route Evaluation Technical Specification

**Milestone:** P7 — Predictive Route Evaluation  
**Packages:** `src.traffic.predictive_evaluation`, `src.models`  
**Scope:** Forward Spatio-Temporal Travel Time Propagation, Route Congestion Scoring, Bottleneck Attribution, Multi-Route Predictive Comparison, and Pluggable Speed Provider Layer  
**Status:** SPECIFICATION / ARCHITECTURE DESIGN (Implementation Pending Approval)  
**Upstream Baselines:** P0–P6 Complete (153/153 Automated Regression Tests Passing)  

---

## 1. Executive Summary & Objective

Milestone **P6** delivered the static origin–destination (OD) routing layer, providing validated topological pathfinding and structurally diverse corridor generation (`RoutingEngine.plan_route` and `plan_route_by_nodes`).

Milestone **P7** bridges static network topology with dynamic traffic intelligence. In real-world urban and highway networks, vehicle speed varies dynamically over time: a commuter departing Coimbatore at 08:30 AM will encounter different traffic states when reaching Chitra Junction (at 08:45 AM) compared to reaching Avinashi (at 09:15 AM). A static snapshot speed calculation is scientifically invalid for multi-segment trips.

**P7 delivers a forward spatio-temporal route evaluation engine that answers core user questions:**
1. *"How long will my trip take if I depart at time $t_{\text{dep}}$?"* (Forward time-integrated Estimated Travel Time).
2. *"How much delay should I expect due to traffic congestion?"* (Signed delay vs. free-flow baseline).
3. *"Where are the bottlenecks along my route?"* (Ranked, named congestion hotspots).
4. *"Which route option has lower predicted travel time?"* (Predictive multi-corridor comparison and time savings quantification).
5. *"What data sources inform this prediction?"* (Transparent, honest data provenance without fabricated sensor claims).

---

## 2. Upstream Capabilities vs. Planned P7 Scope

To maintain architectural integrity and prevent ambiguity, the boundary between implemented upstream milestones and planned P7 capabilities is defined explicitly:

| Capability Area | Implemented Upstream Baseline (P0–P6) | Planned Milestone P7 Scope |
|---|---|---|
| **Road Network Topology** | 251,867 nodes, 549,919 edges in `RoadNetworkGraph` (P1). | Consumed read-only; no graph mutation or topology modification. |
| **Pathfinding & Alternatives** | Shortest/fastest corridor discovery via Dijkstra and penalized search (`plan_route`, `plan_route_by_nodes`) (P6). | Consumes `RoutePlanResult` candidate corridors; does **not** re-run Dijkstra graph pathfinding. |
| **Segment Congestion Intelligence** | Deterministic segment delay calculation and LoS classification (`DelayCalculator`, `CongestionClassifier`, `evaluate_road_network_edge`) (P5). | Applied sequentially at simulated arrival times $t_{\text{arr}}(e)$ to evaluate dynamic route performance. |
| **Speed Models** | METR-LA DCRNN PyTorch model (`p4_dcrnn_best.pt`) on 207 sensors (P4); Tabular lag XGBoost (P3); Heuristic generator in `src/traffic/ingestion.py`. | Abstract `BaseSpeedProvider` interface; `HeuristicDiurnalSpeedProvider` for Coimbatore; `DCRNNSpeedProvider` for sensorized subgraphs; `StaticFreeFlowSpeedProvider` fallback. |
| **Time Propagation** | Static segment traversal: $\tau_{\text{static}} = L / v_{\text{free}}$ (P6). | Dynamic sequential integration: $\tau_i = L_i / v_i(t_{\text{arr}})$, updating arrival timestamp for each subsequent segment. |
| **Bottleneck Attribution** | Segment-level flag (`is_bottleneck` in synthetic generator) (P2). | Grouped, ranked, named corridor bottlenecks with signed delay attribution and no double-counting. |
| **Multi-Route Comparison** | Structural overlap ratio $\text{overlap}(R_a, R_b)$ and static length/time sorting (P6.4). | Side-by-side predicted travel time comparison at identical $t_{\text{dep}}$, recommendation rationale, and configurable time savings thresholds. |

---

## 3. Architecture & Data Contracts

### 3.1 System Integration Architecture

The predictive evaluation engine operates downstream of P6 route generation and upstream of future P8 backend APIs:

```
[ P6 RoutingEngine.plan_route(...) ]
               │
               ▼ Returns RoutePlanResult
    (primary_route + alternative_routes)
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│             PredictiveRouteEvaluator (Milestone P7)         │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 1. Forward Spatio-Temporal Simulation Loop          │   │
│   │    t_arr(e_{i+1}) = t_arr(e_i) + L(e_i) / v(e_i, t) │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │ Queries                       │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 2. Pluggable Speed Provider Layer                   │   │
│   │    - HeuristicDiurnalSpeedProvider (TN Pilot)       │   │
│   │    - DCRNNSpeedProvider (P4 Sensor Subgraph)        │   │
│   │    - StaticFreeFlowSpeedProvider (Fallback)         │   │
│   │    - MockSpeedProvider (Deterministic Unit Tests)   │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │ Feeds                         │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 3. P5 Delay & Congestion Intelligence Engine        │   │
│   │    - DelayCalculator (signed delay seconds/minutes) │   │
│   │    - CongestionClassifier (LoS classification)      │   │
│   │    - Bottleneck Extractor (top-k ranked hotspots)   │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │ Outputs                       │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 4. RouteComparisonResult                            │   │
│   │    - Evaluated Primary Route                        │   │
│   │    - Evaluated Alternative Routes                   │   │
│   │    - Recommended Route ID & Expected Time Savings   │   │
│   │    - Explicit Non-Safety Travel-Time Disclaimer     │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
               │
               ▼ Future P8 FastAPI Consumption
     /api/v1/routes/evaluate
```

---

### 3.2 Core Data Contracts (Pydantic V2)

All schemas enforce strict data validation, explicit units, timezone-aware timestamps, and transparent provenance.

#### A. Speed Query & Provenance Contract
```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class SpeedProvenanceType(str, Enum):
    """Originating methodology for segment speed estimation."""
    ASSUMPTION_BASED_DIURNAL = "assumption_based_diurnal"  # Heuristic diurnal curve based on peak/off-peak assumptions
    MODEL_PREDICTED = "model_predicted"                    # Spatio-temporal ML/DCRNN multi-horizon forecast (METR-LA only)
    HISTORICAL_PROFILE = "historical_profile"              # Historical time-of-day tabular sensor mean (P3)
    FREE_FLOW_FALLBACK = "free_flow_fallback"              # Static OSM speed limit / road hierarchy default
    SYNTHETIC_TEST = "synthetic_test"                      # Deterministic test fixture only (explicitly marked)

class ForecastHorizonStatus(str, Enum):
    """Status of forecast horizon relative to model capabilities."""
    WITHIN_HORIZON = "within_horizon"                      # Arrival time is within valid forecast horizon (e.g. <= 60 min)
    BEYOND_MODEL_HORIZON = "beyond_model_horizon"          # Arrival time exceeds model horizon (> 60 min); fallback invoked
    NOT_APPLICABLE = "not_applicable"                      # Speed source does not use a horizon (diurnal, free-flow, synthetic)

class SpeedQueryResult(BaseModel):
    """Result of querying a speed provider for an edge at a given target arrival time."""
    speed_kmh: float = Field(..., gt=0.0, description="Estimated vehicle speed in km/h (strictly positive)")
    provenance_type: SpeedProvenanceType = Field(..., description="Classification of speed derivation")
    model_source: str = Field(..., description="Identifier of model, provider, or fallback utilized")
    forecast_horizon_minutes: Optional[int] = Field(default=None, description="Forecast horizon step in minutes if model-derived")
    forecast_horizon_status: ForecastHorizonStatus = Field(
        default=ForecastHorizonStatus.NOT_APPLICABLE,
        description="Horizon availability classification"
    )
    confidence_tier: str = Field(..., description="Documented confidence tier: HEURISTIC_ESTIMATE, BENCHMARK_SENSOR, or BASELINE_UNMONITORED")
    is_fallback: bool = Field(default=False, description="True if primary model was unavailable or horizon was exceeded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic notes, limitations, or lineage")
```

#### B. Evaluated Route Segment Contract
```python
from src.network.routing import RouteSegment
from src.traffic.congestion import SegmentCongestionResult

class EvaluatedRouteSegment(BaseModel):
    """Traversed edge segment enriched with forward arrival time and dynamic congestion metrics."""
    segment: RouteSegment = Field(..., description="Underlying P6 structural route segment")
    arrival_time: datetime = Field(..., description="Timezone-aware arrival timestamp at segment start")
    traversal_time_seconds: float = Field(..., gt=0.0, description="Dynamic traversal duration in seconds")
    free_flow_time_seconds: float = Field(..., gt=0.0, description="Static free-flow reference time in seconds")
    signed_delay_seconds: float = Field(..., description="Signed delay in seconds: traversal_time - free_flow_time (positive=delay, negative=speedup)")
    speed_result: SpeedQueryResult = Field(..., description="Speed evaluation details, horizon status, and provenance")
    congestion_result: SegmentCongestionResult = Field(..., description="P5 canonical segment congestion result")
    is_bottleneck: bool = Field(default=False, description="True if delay contribution qualifies as an actionable bottleneck")
```

#### C. Route Bottleneck Contract
```python
from src.traffic.congestion import CongestionLevel

class RouteBottleneck(BaseModel):
    """Identified and attributed congestion hotspot along a candidate route."""
    rank: int = Field(..., ge=1, description="Rank ordered descending by delay contribution (1 = highest delay)")
    segment_id: str = Field(..., description="Graph edge identifier or clustered corridor identifier")
    u: int = Field(..., description="Origin node ID of edge")
    v: int = Field(..., description="Target node ID of edge")
    key: int = Field(default=0, description="Multigraph edge key")
    road_name: Optional[str] = Field(default=None, description="Human-readable road name")
    road_reference: Optional[str] = Field(default=None, description="Official road reference (e.g. NH-544)")
    highway_type: str = Field(..., description="Functional road class (trunk, primary, secondary, etc.)")
    delay_seconds: float = Field(..., gt=0.0, description="Absolute delay contribution in seconds")
    delay_minutes: float = Field(..., gt=0.0, description="Delay contribution in minutes")
    observed_speed_kmh: float = Field(..., gt=0.0, description="Estimated traversal speed in km/h")
    free_flow_speed_kmh: float = Field(..., gt=0.0, description="Reference baseline speed in km/h")
    speed_ratio: float = Field(..., ge=0.0, description="Ratio of speed to free-flow (v_pred / v_free)")
    congestion_level: CongestionLevel = Field(..., description="P5 congestion severity (HIGH, SEVERE)")
    arrival_time: datetime = Field(..., description="Timezone-aware vehicle arrival timestamp at bottleneck")
    display_title: str = Field(..., description="Concise bottleneck label for UI card")
```

#### D. Evaluated Route & Multi-Route Comparison Contracts
```python
from typing import List

class EvaluatedRoute(BaseModel):
    """Comprehensive predictive evaluation of a single route candidate."""
    route_id: str = Field(..., description="Unique route identifier from P6")
    name: str = Field(..., description="Descriptive corridor name (e.g. 'Via NH-544')")
    departure_time: datetime = Field(..., description="Planned departure timestamp (timezone-aware)")
    estimated_arrival_time: datetime = Field(..., description="Simulated trip completion timestamp (timezone-aware)")
    total_distance_m: float = Field(..., ge=0.0, description="Total route distance in meters")
    total_distance_km: float = Field(..., ge=0.0, description="Total distance in kilometres")
    free_flow_time_seconds: float = Field(..., ge=0.0, description="Theoretical zero-traffic journey duration in seconds")
    predicted_travel_time_seconds: float = Field(..., ge=0.0, description="Forward time-integrated journey duration in seconds")
    predicted_travel_time_minutes: float = Field(..., ge=0.0, description="Journey duration in minutes")
    total_delay_seconds: float = Field(..., description="Signed cumulative delay in seconds: predicted_time - free_flow_time")
    total_delay_minutes: float = Field(..., description="Signed cumulative delay in minutes")
    average_speed_kmh: float = Field(..., ge=0.0, description="Average journey velocity under predicted traffic")
    route_congestion_level: CongestionLevel = Field(..., description="Overall Level of Service (LoS)")
    bottlenecks: List[RouteBottleneck] = Field(default_factory=list, description="Top delay-inducing bottlenecks")
    evaluated_segments: List[EvaluatedRouteSegment] = Field(..., description="Segment-by-segment simulation details")
    data_provenance: Dict[str, Any] = Field(..., description="Breakdown of models, assumptions, and fallback usage")

class RouteComparisonResult(BaseModel):
    """Side-by-side comparative evaluation of primary and alternative routes."""
    departure_time: datetime = Field(..., description="Evaluation departure timestamp (timezone-aware)")
    recommended_route_id: str = Field(..., description="Route ID recommended for lowest predicted travel time")
    recommendation_reason: str = Field(..., description="Human-readable decision rationale based strictly on travel time")
    time_savings_seconds: float = Field(..., ge=0.0, description="Time saved in seconds by choosing recommendation over primary route")
    time_savings_minutes: float = Field(..., ge=0.0, description="Time saved in minutes")
    primary_route: EvaluatedRoute = Field(..., description="Evaluated primary corridor")
    alternative_routes: List[EvaluatedRoute] = Field(default_factory=list, description="Evaluated alternative corridors")
    ranked_routes: List[EvaluatedRoute] = Field(..., description="All routes sorted by predicted travel time ascending")
    evaluation_disclaimer: str = Field(
        default="DISCLAIMER: This recommendation optimizes estimated travel time only based on simulated traffic impedance. It does NOT evaluate road safety, accident risk, pavement condition, lighting, or weather hazards, and must not be interpreted as a safety recommendation.",
        description="Explicit non-safety advisory"
    )
```

---

## 4. Forward Spatio-Temporal Simulation Engine

### 4.1 Mathematical Formulation

Let candidate route $R$ consist of $N$ sequential directed road segments $e_0, e_1, \dots, e_{N-1}$.

1. **Departure Initialization:**
   $$t_{\text{arr}}(e_0) = t_{\text{dep}}$$
   Where $t_{\text{dep}}$ must be a timezone-aware `datetime`.

2. **Sequential Dynamic Traversal:**
   For each segment $e_i = (u_i, v_i, k_i)$ with length $L(e_i) > 0\text{ meters}$:
   - Retrieve dynamic velocity estimate at current simulated arrival time:
     $$v(e_i) = \text{Provider.get\_edge\_speed}(u_i, v_i, k_i, \text{edge\_data}, t_{\text{arr}}(e_i), t_{\text{dep}})$$
   - Convert velocity to SI units ($\text{m/s}$):
     $$v_{\text{mps}}(e_i) = v(e_i) \times \left(\frac{1000.0}{3600.0}\right)$$
   - Compute segment traversal duration in seconds:
     $$\tau(e_i) = \frac{L(e_i)}{v_{\text{mps}}(e_i)} \quad [\text{seconds}]$$
   - Propagate arrival time to the subsequent segment:
     $$t_{\text{arr}}(e_{i+1}) = t_{\text{arr}}(e_i) + \text{timedelta}(\text{seconds}=\tau(e_i))$$

3. **Cumulative Metrics & Consistency Invariants:**
   - Total Predicted Travel Time:
     $$T_{\text{pred}}(R, t_{\text{dep}}) = \sum_{i=0}^{N-1} \tau(e_i) \quad [\text{seconds}]$$
   - Total Free-Flow Reference Time:
     $$T_{\text{ff}}(R) = \sum_{i=0}^{N-1} \frac{L(e_i)}{v_{\text{free}}(e_i) \times (1000.0 / 3600.0)} \quad [\text{seconds}]$$
   - Total Route Congestion Delay:
     $$\Delta T(R) = T_{\text{pred}}(R, t_{\text{dep}}) - T_{\text{ff}}(R) = \sum_{i=0}^{N-1} \left( \tau(e_i) - \tau_{\text{ff}}(e_i) \right) \quad [\text{seconds}]$$

---

### 4.2 Timezone & Numerical Validation Policies

1. **Timezone Policy:**
   - Input departure timestamps must be timezone-aware. Naive timestamps without `tzinfo` must be rejected with a descriptive `ValueError`.
   - UTC and `Asia/Kolkata` (UTC+05:30) are officially supported. Segments record timezone-aware ISO8601 timestamps.
2. **Dimension Validation:**
   - $L(e_i)$ must be strictly positive ($> 0.0\text{ m}$) and finite. Non-finite or $\le 0.0$ lengths reject route evaluation.
   - $v_{\text{free}}(e_i)$ must be strictly positive ($> 0.0\text{ km/h}$) and finite (defaulting to standardized OSM hierarchy default $30.0\text{ km/h}$ if missing).
3. **Monotonicity Guarantee:**
   - For all $i < j$, $t_{\text{arr}}(e_i) < t_{\text{arr}}(e_j)$ is strictly guaranteed whenever $\sum_{k=i}^{j-1} L(e_k) > 0$.
4. **Single-Node Route Invariant (`SAME_LOCATION`):**
   - When a route contains 1 node and 0 edges:
     - $t_{\text{arr}} = t_{\text{dep}}$, $T_{\text{pred}} = 0.0\text{ s}$, $T_{\text{ff}} = 0.0\text{ s}$, $\text{Delay} = 0.0\text{ s}$, `bottlenecks = []`.

---

## 5. Speed Provider Layer: Evidence, Capabilities & Fallback Policy

### 5.1 Repository Evidence & Reality Audit

An exhaustive audit of repository assets establishes the following ground truths:
1. **No Measured Tamil Nadu Sensor Telemetry:** The repository contains no loop detector telemetry, probe GPS feeds, or commercial speed data for Tamil Nadu.
2. **No Calibrated IRC / MoRTH Datasets or Calibration Code:** While strings in `src/traffic/ingestion.py` cite `"IRC:106-1990"`, there is **no empirical calibration pipeline, no raw survey dataset, and no regression-fitted parameter table** in the codebase.
3. **P4 DCRNN Model Domain:** The trained PyTorch model `artifacts/models/p4_dcrnn_best.pt` operates strictly on the 207-sensor Los Angeles METR-LA benchmark graph. It has no physical mapping to the 251,867 OpenStreetMap nodes of Coimbatore.

**Mandatory Truthfulness Safeguards:**
- Under NO circumstances will METR-LA predictions be mapped onto Coimbatore road edges.
- Under NO circumstances will diurnal curves be described as "calibrated" or "empirical".
- All heuristic curves are explicitly documented as **assumption-based diurnal estimates**.

---

### 5.2 Provider Interface: `BaseSpeedProvider`

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any

class BaseSpeedProvider(ABC):
    """Abstract interface defining speed evaluation for network edges."""

    @abstractmethod
    def get_edge_speed(
        self,
        u: int,
        v: int,
        key: int,
        edge_data: Dict[str, Any],
        target_arrival_time: datetime,
        departure_time: datetime,
    ) -> SpeedQueryResult:
        """
        Evaluate speed for edge (u, v, key) at vehicle arrival timestamp.

        Args:
            u: Source node ID.
            v: Target node ID.
            key: Edge key in multigraph.
            edge_data: NetworkX edge attribute dictionary.
            target_arrival_time: Simulated arrival time of vehicle at edge start (timezone-aware).
            departure_time: Initial trip departure timestamp (timezone-aware).

        Returns:
            SpeedQueryResult containing finite positive speed, horizon status, and explicit provenance.
        """
        pass
```

---

### 5.3 Concrete Provider Implementations

#### Provider 1: `HeuristicDiurnalSpeedProvider` (Primary for Coimbatore Pilot Corridor)
- **Methodology:** Implements an assumption-based engineering heuristic modeled after the diurnal curve in `src/traffic/ingestion.py`. It adjusts edge free-flow speeds using time-of-day multipliers:
  - **Local Time Calculation:** Converts arrival timestamp to Indian Standard Time (IST, UTC+05:30).
  - **Diurnal Multiplier Curve $M(t, \text{is\_weekend}, \text{highway\_type})$:**
    - *Morning Peak (08:00 – 10:30 IST):* Multiplier $M \in [0.55, 0.70]$ on urban arterials (`primary`, `secondary`); $M \in [0.75, 0.85]$ on highways (`trunk`, `motorway`).
    - *Evening Peak (17:30 – 20:30 IST):* Multiplier $M \in [0.50, 0.65]$ on urban arterials; $M \in [0.70, 0.80]$ on highways.
    - *Midday Inter-Peak (11:00 – 16:30 IST):* Multiplier $M \in [0.80, 0.90]$.
    - *Night Off-Peak (23:00 – 05:00 IST):* Multiplier $M \in [0.95, 1.00]$ (near free-flow).
    - *Weekend Adjustment:* Peak impedance attenuated by $25\%$ ($M$ closer to $1.0$).
- **Output Speed:**
  $$v_{\text{pred}}(e, t) = \max\left(10.0, v_{\text{free}}(e) \times M(t)\right) \quad [\text{km/h}]$$
- **Explicit Lineage:** `provenance_type = ASSUMPTION_BASED_DIURNAL`, `confidence_tier = "HEURISTIC_ESTIMATE"`, `forecast_horizon_status = NOT_APPLICABLE`.

#### Provider 2: `DCRNNSpeedProvider` (P4 Model on Sensorized Benchmark Subgraphs Only)
- **Operational Scope:** Strictly restricted to graphs with authentic sensor-to-node mappings (such as the METR-LA benchmark). Prohibited from running on Coimbatore OSM nodes.
- **Horizon Policy (Explicit 60-Minute Limit):**
  - The P4 model forecasts 12 discrete 5-minute steps ($5, 10, \dots, 60$ minutes).
  - If simulated arrival delta $\Delta t = t_{\text{arr}} - t_{\text{dep}} \le 60\text{ minutes}$:
    - Step index: $h = \min(12, \max(1, \lceil \Delta t / 5\text{ min} \rceil))$.
    - `forecast_horizon_minutes = h * 5`.
    - `forecast_horizon_status = WITHIN_HORIZON`.
    - `is_fallback = False`.
  - If simulated arrival delta $\Delta t > 60\text{ minutes}$:
    - **No Unsupported Holding or Blending:** The repository contains no empirical validation for holding step 12 steady or blending multi-step forecasts.
    - The provider **must explicitly transition to the fallback provider** (`StaticFreeFlowSpeedProvider` or `HeuristicDiurnalSpeedProvider`).
    - At segment level: `forecast_horizon_status = BEYOND_MODEL_HORIZON`, `forecast_horizon_minutes = None`, `is_fallback = True`, `model_source = "P4_DCRNN_EXCEEDED_FALLBACK"`.

#### Provider 3: `StaticFreeFlowSpeedProvider` (Baseline Fallback)
- **Methodology:** Returns edge `free_flow_speed_kmh` without dynamic modification.
- **Lineage:** `provenance_type = FREE_FLOW_FALLBACK`, `confidence_tier = "BASELINE_UNMONITORED"`, `is_fallback = True`.

#### Provider 4: `MockSpeedProvider` (Deterministic Testing Fixture)
- **Methodology:** Injects pre-configured static or time-indexed speeds per edge key for unit testing.
- **Lineage:** `provenance_type = SYNTHETIC_TEST`, `confidence_tier = "SYNTHETIC_TEST_ONLY"`.

---

### 5.4 Speed Guard & Numerical Degradation Policies

1. **Zero Speed (Stopped Traffic) Policy:**
   - If a provider reports $v = 0.0\text{ km/h}$ (stopped traffic / gridlock):
     - Traversal calculation enforces an explicit minimum physical crawl floor $v_{\text{crawl}} = 3.0\text{ km/h}$ ($0.833\text{ m/s}$) strictly to avoid division-by-zero errors.
     - The segment's `SegmentCongestionResult` is explicitly assigned `CongestionStatus.STOPPED` and `CongestionLevel.SEVERE`.
     - The segment is elevated to a high-priority bottleneck.
2. **Negative, NaN, or Non-Finite Speeds:**
   - Traversal automatically substitutes $v_{\text{free}}(e)$ (or standardized OSM hierarchy fallback $30.0\text{ km/h}$).
   - Provenance is marked `is_fallback = True` with status `INVALID_SPEED` or `MISSING_SPEED`.
3. **Speeds Exceeding Free-Flow:**
   - Traversal accepts valid speeds up to $1.2 \times v_{\text{free}}$ (yielding negative delay / speedup), assigned `CongestionStatus.FASTER_THAN_FREE_FLOW`.

---

## 6. Congestion Scoring & Bottleneck Attribution

### 6.1 Route-Level Level of Service (LoS)

The route-level congestion category aggregates segment conditions using length-weighted congestion index:

$$\overline{CI}(R) = \frac{\sum_{i=0}^{N-1} L(e_i) \times CI(e_i)}{\sum_{i=0}^{N-1} L(e_i)}$$

Where $CI(e_i) = 1.0 - \min\left(1.0, \frac{v_{\text{pred}}(e_i)}{v_{\text{free}}(e_i)}\right) \in [0.0, 1.0]$.

- $\overline{CI} < 0.20$: **LOW** (Free flow across corridor)
- $0.20 \le \overline{CI} < 0.40$: **MODERATE** (Minor queuing / slow crawl sections)
- $0.40 \le \overline{CI} < 0.60$: **HIGH** (Significant congestion throughout major segments)
- $\overline{CI} \ge 0.60$: **SEVERE** (Heavy corridor-wide delay / gridlock)

---

### 6.2 Bottleneck Extraction & Non-Double-Counting Rules

1. **Delay Definitions:**
   - **Segment Signed Delay:** $\Delta \tau(e) = \tau_{\text{pred}}(e) - \tau_{\text{free\_flow}}(e)$. Note that $\Delta \tau(e)$ may be negative when predicted speed exceeds reference free-flow speed (speedup).
   - **Net Signed Route Delay:** 
     $$\Delta T_{\text{net}}(R) = T_{\text{pred}}(R) - T_{\text{free\_flow}}(R) = \sum_{e \in R} \Delta \tau(e)$$
   - **Segment Positive Delay Contribution:** $\Delta \tau^{+}(e) = \max(0.0, \Delta \tau(e))$.
   - **Gross Positive Route Delay:**
     $$\Delta T_{\text{pos}}(R) = \sum_{e \in R} \Delta \tau^{+}(e) = \sum_{e \in R} \max(0.0, \Delta \tau(e))$$
   - **Gross Negative Delay (Speedup):**
     $$\Delta T_{\text{neg}}(R) = \sum_{e \in R} \max(0.0, -\Delta \tau(e))$$
   - **Relationship:** $\Delta T_{\text{net}}(R) = \Delta T_{\text{pos}}(R) - \Delta T_{\text{neg}}(R) \le \Delta T_{\text{pos}}(R)$. When any segment exhibits negative delay ($\Delta \tau(e) < 0$), net route delay is strictly less than gross positive delay ($\Delta T_{\text{net}}(R) < \Delta T_{\text{pos}}(R)$). Therefore, total positive bottleneck delay must **not** be bounded by net signed route delay.

2. **Qualification Criteria:**
   A segment $e$ qualifies as an actionable bottleneck if:
   - Traffic is stopped (`CongestionStatus.STOPPED` or $v = 0.0\text{ km/h}$), **OR**
   - `congestion_level` is `HIGH` or `SEVERE` ($CI(e) \ge 0.40$), **AND**
   - Positive delay contribution $\Delta \tau(e) \ge 30.0\text{ seconds}$ ($0.5\text{ minutes}$).

3. **Corridor Clustering & Deduplication:**
   Consecutive segments in the route that qualify as bottlenecks and share the same road name (or same road reference if road name is missing) are aggregated into a single continuous corridor bottleneck:
   $$\Delta \tau_{\text{bottleneck}} = \sum_{e \in \text{group}} \Delta \tau^{+}(e)$$
   If road name and reference are both missing, consecutive unnamed segments are not aggregated into ambiguous multi-link clusters; instead, each unnamed segment is attributed to its own localized bottleneck near its origin node.

4. **Mathematically Valid Non-Double-Counting Invariant:**
   Let each bottleneck $B_k$ be formed from a subset of qualifying segments $S_k \subseteq R$.
   - **Disjoint Attribution:** The subsets $S_k$ are mutually disjoint:
     $$S_j \cap S_k = \emptyset \quad \forall j \neq k$$
     Each segment's positive delay contribution is attributed at most once:
     $$\sum_{k: e \in S_k} 1 \le 1 \quad \forall e \in R$$
   - **Bounded by Gross Positive Delay:**
     $$\sum_{B_k \in \text{bottlenecks}} \Delta \tau(B_k) \le \Delta T_{\text{pos}}(R)$$
     The total attributed bottleneck delay is strictly bounded by the route's gross positive delay $\Delta T_{\text{pos}}(R)$. It may exceed net signed route delay $\Delta T_{\text{net}}(R)$ if other segments provide negative delay (speedup).

5. **Display Title Formatting:**
   - If road name and reference exist: `"{name} ({road_reference}): +{delay_min:.1f}m delay"` (e.g. *"Avinashi Road (NH-544): +4.2m delay"*)
   - If only road name exists: `"{name}: +{delay_min:.1f}m delay"`
   - If only road reference exists: `"{road_reference}: +{delay_min:.1f}m delay"`
   - If unnamed: `"{highway_type.replace('_', ' ').title()} corridor near Node {u}: +{delay_min:.1f}m delay"`

---

## 7. Predictive Multi-Route Comparison & Recommendation Policy

### 7.1 Configurable Decision Parameters

```python
class PredictiveEvaluatorConfig(BaseModel):
    """Configurable evaluation and recommendation thresholds."""
    min_time_savings_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Minimum seconds an alternative must save over the primary route to trigger recommendation"
    )
    tie_threshold_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Threshold within which two route travel times are considered tied"
    )
    max_bottlenecks_per_route: int = Field(
        default=5,
        ge=1,
        description="Maximum number of top bottlenecks to surface per route"
    )
    zero_speed_crawl_kmh: float = Field(
        default=3.0,
        gt=0.0,
        description="Crawl speed floor for stopped traffic to avoid zero division"
    )
```

---

### 7.2 Recommendation & Tie-Breaking Rules

1. **Sorting Key:**
   All routes are ranked ascending by `predicted_travel_time_seconds`:
   $$\text{Rank 1} = \arg\min_{R} T_{\text{pred}}(R, t_{\text{dep}})$$
2. **Recommendation Logic:**
   - If $\text{Rank 1}$ route is the primary route: `recommended_route_id = primary_route.route_id`.
   - If $\text{Rank 1}$ route is an alternative route and saves $\ge \text{min\_time\_savings\_seconds}$ ($60\text{ s}$ default) over the primary route:
     - `recommended_route_id = alternative_route.route_id`.
     - `time_savings_seconds = primary.predicted_travel_time_seconds - alternative.predicted_travel_time_seconds`.
     - `recommendation_reason = "Alternative route '{name}' has lower estimated travel time (saves {savings_min:.1f} mins) due to reduced congestion."`
   - If an alternative is faster but saves $< \text{min\_time\_savings\_seconds}$:
     - Recommendation remains with primary route to avoid recommending complex detours for trivial gains.
     - The time difference is still transparently reported.
3. **Tie-Breaking ($\Delta T \le \text{tie\_threshold\_seconds}$):**
   - If two routes have travel times within $30.0\text{ seconds}$:
     - Prefer shorter physical distance (`total_distance_m` ascending).
     - If distances are equal within $50\text{ meters}$, prefer primary route.
4. **Mandatory Non-Safety Advisory:**
   All comparison responses must include the explicit disclaimer:
   > *"DISCLAIMER: This recommendation optimizes estimated travel time only based on simulated traffic impedance. It does NOT evaluate road safety, accident risk, pavement condition, lighting, or weather hazards, and must not be interpreted as a safety recommendation."*

---

## 8. Incremental Implementation Plan

Milestone P7 will be implemented in four strictly partitioned, test-driven phases:

```
[ P7.1: Simulation Engine & Speed Provider Contracts ]
                          │
                          ▼
[ P7.2: Route Congestion Scoring & Bottlenecks ]
                          │
                          ▼
[ P7.3: Predictive Multi-Route Comparison Facade ]
                          │
                          ▼
[ P7.4: Pilot Integration & Regression Closure ]
```

---

### Phase P7.1 — Spatio-Temporal Propagation Engine & Speed Providers
- **Scope Restriction:** Limited strictly to forward time propagation, speed lookup abstraction, and numerical/timezone invariants. Excludes bottleneck extraction (P7.2) and route comparison (P7.3).
- **Exact Files Created / Modified:**
  - Create: `src/traffic/predictive_evaluation.py`
  - Create: `tests/p7/test_time_propagation.py`
- **Classes & Functions to Implement:**
  - `SpeedProvenanceType`, `ForecastHorizonStatus`, `SpeedQueryResult` schemas.
  - `BaseSpeedProvider` interface.
  - `StaticFreeFlowSpeedProvider` implementation.
  - `HeuristicDiurnalSpeedProvider` implementation (IST peak/off-peak curve).
  - `MockSpeedProvider` implementation (deterministic testing).
  - `forward_propagate_route(route, departure_time, speed_provider)` engine function.
- **Focused Unit Tests (`tests/p7/test_time_propagation.py`):**
  1. Monotonic timestamp advancement across multi-segment routes.
  2. Timezone validation: rejects naive timestamps; handles UTC and IST offsets.
  3. Single-node route behavior (0 distance, 0 time, same arrival time).
  4. Heuristic diurnal speed variation across 08:30 IST (peak), 14:00 IST (midday), and 02:00 IST (night).
  5. Fallback behavior when edge speeds are missing or non-finite.
  6. Zero-speed crawl floor ($3.0\text{ km/h}$) preventing infinite traversal time.
- **Measurable Exit Criteria:**
  - 100% passing tests in `tests/p7/test_time_propagation.py`.
  - Zero modifications to P0–P6 source files.
  - Full 153-test baseline preserved.

---

### Phase P7.2 — Route Congestion Scoring & Bottleneck Attribution
- **Scope Restriction:** Implement segment-level delay integration, route-level LoS, and corridor bottleneck extraction.
- **Exact Files Created / Modified:**
  - Modify: `src/traffic/predictive_evaluation.py`
  - Create: `tests/p7/test_bottlenecks.py`
- **Classes & Functions to Implement:**
  - `EvaluatedRouteSegment` and `RouteBottleneck` schemas.
  - Integration with P5's `DelayCalculator` and `CongestionClassifier`.
  - Corridor bottleneck clustering, ranking, and naming logic.
  - Non-double-counting verification invariant.
- **Focused Unit Tests (`tests/p7/test_bottlenecks.py`):**
  1. Net signed delay sum matches total route delay ($\Delta T_{\text{net}} = \sum \Delta \tau_i$), and gross positive delay decomposes correctly ($\Delta T_{\text{net}} = \Delta T_{\text{pos}} - \Delta T_{\text{neg}}$).
  2. Bottleneck delay bounded by gross positive delay ($\sum \Delta \tau(b) \le \Delta T_{\text{pos}}$), and each segment's positive delay is attributed at most once.
  3. Continuous corridor deduplication (consecutive segments with same road name or reference aggregated).
  4. Stopped traffic flagged as `STOPPED` and elevated to critical bottleneck.
  5. Missing road names and display title formatting (handling named, referenced, and unnamed corridors).
  6. Routes with no positive-delay bottlenecks (free-flow, sub-threshold delay, and zero-distance routes).
- **Measurable Exit Criteria:**
  - Congestion levels strictly match P5 classification formulas.
  - 100% passing tests in `tests/p7/test_bottlenecks.py`.

---

### Phase P7.3 — Predictive Multi-Route Comparison Facade
- **Scope Restriction:** Implement multi-route evaluation, ranking, configurable time savings threshold, and recommendation reasoning.
- **Exact Files Created / Modified:**
  - Modify: `src/traffic/predictive_evaluation.py`
  - Create: `tests/p7/test_route_comparison.py`
- **Classes & Functions to Implement:**
  - `PredictiveEvaluatorConfig`, `EvaluatedRoute`, and `RouteComparisonResult` schemas.
  - `PredictiveRouteEvaluator.evaluate_route_plan(plan_result, departure_time, provider, config)`.
  - Configurable minimum savings evaluation ($60\text{ s}$ default).
  - Tie-breaking logic and non-safety disclaimer inclusion.
- **Focused Unit Tests (`tests/p7/test_route_comparison.py`):**
  1. Multi-corridor synthetic graph: Highway with peak delay vs. free-flowing arterial.
  2. Alternative recommended only when savings $\ge \text{min\_time\_savings\_seconds}$.
  3. Tie-breaking on distance and primary preference within $30\text{ s}$.
  4. Presence and text validation of non-safety disclaimer.
- **Measurable Exit Criteria:**
  - 100% passing tests in `tests/p7/test_route_comparison.py`.

---

### Phase P7.4 — Integration, Pilot Corridor Verification & Milestone Closure
- **Scope Restriction:** Real-world pilot corridor verification and full test suite regression closure.
- **Exact Files Created / Modified:**
  - Create: `tests/p7/test_pilot_predictive_integration.py`
  - Create: `docs/P7_verification_report.md`
- **Focused Integration Tests:**
  1. Evaluation of Coimbatore Railway Station $\to$ Avinashi corridor at 08:30 IST vs. 02:00 IST.
  2. Verification that morning peak exhibits higher delay and lower average speed on urban segments.
  3. Multi-alternative evaluation on pilot network corridors.
- **Measurable Exit Criteria:**
  - Full test suite passes: all 153 existing tests + all new P7 tests (~25 new tests, target: $\ge 178$ total).
  - Zero file modifications outside approved P7 scope.
  - Preserved cryptographic checksums for pilot network graph and model checkpoints.

---

## 9. Consistency Review & Explicit Limitations

- **Consistency with P0:** Directly implements Workflow Step 5 (Dynamic Route Travel Time Integration), Formulation 2.3 (Forward Spatio-Temporal Simulation), and Section 6 (Graceful & Honest Degradation Policy).
- **Consistency with P5:** Reuses `CongestionClassifier`, `DelayCalculator`, and `SegmentCongestionResult` schemas without redefining classification thresholds.
- **Consistency with P6:** Consumes `RoutePlanResult`, `Route`, and `RouteSegment` read-only. Does not mutate the underlying graph or alter P6 shortest-path topological search.
- **Explicit Limitations Documented:**
  1. *Assumption-Based Diurnal Estimator:* The diurnal speed curve for Coimbatore is an uncalibrated engineering assumption based on synthetic peak-hour schedules, not empirical probe measurements.
  2. *DCRNN Model Domain:* The P4 model weights represent the METR-LA benchmark network and cannot be evaluated on Coimbatore nodes.
  3. *Forecast Horizon Capping:* Multi-step deep learning forecasting is valid only up to 60 minutes. Trips exceeding 60 minutes transition to explicit fallback.
  4. *No Safety Claims:* Predictive route evaluation optimizes estimated travel time only and provides zero safety guarantees.
