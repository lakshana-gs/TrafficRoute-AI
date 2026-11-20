# P5 — Congestion & Delay Intelligence Technical Specification

**Milestone:** P5 — Congestion & Delay Intelligence  
**Package:** `src.traffic.congestion`  
**Scope:** Data-Source-Independent Road Segment Traffic State, Classification, and Delay Engine  
**Status:** COMPLETE (All 22/22 P5 Unit Tests Passed; 79/79 Full Project Tests Passed)  

---

## 1. Executive Summary & Objective

Milestone **P5** establishes a rigorous, mathematically deterministic, data-source-independent intelligence calculation layer for segment-level traffic speed observations and model forecasts. 

Prior milestones established:
- **P1:** Road network extraction, topology modeling, routing, and OSM attribute standardization for Tamil Nadu (Coimbatore-Avinashi Road).
- **P2:** Real-world traffic probe ingestion, cleaning, temporal bucketing, and spatial snapping pipeline.
- **P3 & P4/P4.1:** Historical baseline speed prediction and DCRNN spatio-temporal deep learning benchmark on authentic METR-LA loop detector data.

**P5 bridges raw speed values to actionable congestion intelligence** by computing:
1. Heuristic speed-ratio-based categorical congestion tiers (`low`, `moderate`, `high`, `severe`, `unknown`).
2. Exact travel time estimations and signed delays against baseline free-flow conditions.
3. Strict canonical result contracts preserving genuine lineage, geographic jurisdiction, and operational status without fabricating unobserved attributes.

---

## 2. Non-Goals & Architectural Boundaries

To ensure strict engineering integrity, P5 adheres to the following non-goals:
- **Zero Fabricated Mappings:** P5 does **not** assume that METR-LA California freeway sensor IDs map to Coimbatore OpenStreetMap road network edges. No fake spatial crosswalks or synthetic associations are introduced.
- **Zero Fake Live Traffic:** P5 does not invent live real-time traffic observations, simulated vehicle telemetry, or mock provider feeds.
- **No Model Retraining / Modification:** P4.1 deep learning checkpoints, datasets, scaler parameters, and evaluation logs remain completely untouched and immutable.
- **No External/Paid API Dependencies:** All calculations are self-contained, offline-capable, and deterministic without live TomTom, Google Maps, or external network calls.
- **No Premature P6 Implementation:** Dynamic graph weight updates, predictive routing Dijkstra penalization, and frontend UI components are strictly reserved for P6 and later milestones.

---

## 3. Mathematical Formulations & Physical Units

All calculations in P5 operate on standardized, unscaled physical units.

### 3.1 Standard Physical Units
- **Speed ($v$):** Kilometres per hour ($\text{km/h}$)
- **Length ($L$):** Kilometres ($\text{km}$)
- **Travel Time ($T$):** Minutes ($\text{min}$)
- **Delay ($\Delta T$):** Signed minutes ($\text{min}$)

### 3.2 Congestion Ratio Formulation
Given predicted or observed segment speed $v_{\text{pred}}$ and reference free-flow speed $v_{\text{free}}$:
$$r = \frac{v_{\text{pred}}}{v_{\text{free}}}$$

### 3.3 Free-Flow Baseline Travel Time
$$T_{\text{free\_flow}} = 60.0 \times \frac{L}{v_{\text{free}}}$$
Where $L > 0$ and $v_{\text{free}} > 0$. $T_{\text{free\_flow}}$ is strictly positive and finite.

### 3.4 Estimated Segment Travel Time
For valid non-zero speeds $v_{\text{pred}} > 0$:
$$T_{\text{estimated}} = 60.0 \times \frac{L}{v_{\text{pred}}}$$
*(For $v_{\text{pred}} = 0.0$, finite travel time is mathematically undefined; division by zero is prevented, and travel time is assigned `None` with status `STOPPED`)*.

### 3.5 Estimated Delay & Signed Semantics
$$\Delta T_{\text{delay}} = T_{\text{estimated}} - T_{\text{free\_flow}}$$

**Signed Delay Semantics:**
- **Positive Delay ($\Delta T > 0$):** Traffic impedance/delay. The segment takes longer than free-flow travel time.
- **Zero Delay ($\Delta T = 0$):** Exactly free-flow condition ($v_{\text{pred}} = v_{\text{free}}$).
- **Negative Delay ($\Delta T < 0$):** Speedup / time savings. The predicted speed exceeds the legal/reference free-flow speed ($v_{\text{pred}} > v_{\text{free}}$), resulting in travel time below baseline. Signed values are strictly preserved rather than artificially clamped to zero, avoiding distortion of raw velocity observations.
- **Undefined Delay ($\Delta T = \text{None}$):** When $v_{\text{pred}} = 0.0$, missing, or invalid.

> [!WARNING]
> **Domain Clarification:** Estimated delay is a model-derived or speed-derived estimate against theoretical free-flow speed. It must **never** be presented to users as measured empirical queue waiting time or observed signal delay.

---

## 4. Congestion Classification Tiers & Threshold Configuration

### 4.1 Default Threshold Boundaries
Congestion level is classified using the following documented heuristic ratio thresholds:

| Speed Ratio ($r = v_{\text{pred}} / v_{\text{free}}$) | Congestion Level | Operational Meaning | Status Flag |
| :---: | :---: | :--- | :--- |
| $r > 1.00$ | `low` | Speed exceeds free-flow baseline | `FASTER_THAN_FREE_FLOW` |
| $0.80 \le r \le 1.00$ | `low` | Free-flow or near free-flow conditions | `NORMAL` |
| $0.60 \le r < 0.80$ | `moderate` | Minor to moderate speed impedance | `NORMAL` |
| $0.40 \le r < 0.60$ | `high` | Significant queuing and heavy traffic | `NORMAL` |
| $0.0 < r < 0.40$ | `severe` | Severe breakdown / crawling traffic | `NORMAL` |
| $r == 0.0$ | `severe` | Traffic stopped ($v = 0.0\text{ km/h}$) | `STOPPED` |
| Missing / Negative / Non-finite | `unknown` | Invalid or uncomputable observation | `MISSING_SPEED` / `INVALID_SPEED` |

### 4.2 Configuration Mechanism (`CongestionThresholdConfig`)
Thresholds are fully configurable through the frozen dataclass `CongestionThresholdConfig`:
```python
@dataclass(frozen=True)
class CongestionThresholdConfig:
    low_min_ratio: float = 0.80
    moderate_min_ratio: float = 0.60
    high_min_ratio: float = 0.40
```
**Strict Boundary Validation:**
- Values must be finite floats within $(0.0, 1.0]$.
- Must satisfy: $0.0 < \text{high\_min\_ratio} < \text{moderate\_min\_ratio} < \text{low\_min\_ratio} \le 1.0$.
- Inverted, negative, or out-of-range configurations raise `ValueError` on initialization.

---

## 5. Defensive Input Handling & Edge Cases

| Scenario | Input Condition | Classification Level | Status Flag | Travel Time | Delay | Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal Traffic** | $0 < v \le v_{\text{free}}$ | `low` / `moderate` / `high` / `severe` | `NORMAL` | Computed | Computed | Standard deterministic calculation. |
| **Speeding / Fast Flow** | $v > v_{\text{free}}$ | `low` | `FASTER_THAN_FREE_FLOW` | Computed | Negative ($\Delta T < 0$) | Preserves raw signed time savings. |
| **Stopped Traffic** | $v == 0.0$ | `severe` | `STOPPED` | `None` | `None` | Prevents ZeroDivisionError; finite travel time is undefined. |
| **Missing Prediction** | $v \text{ is None}$ or $\text{NaN}$ | `unknown` | `MISSING_SPEED` | `None` | `None` | Never classifies missing data as valid severe gridlock. |
| **Negative Speed** | $v < 0.0$ | `unknown` | `INVALID_SPEED` | `None` | `None` | Physical impossibility; rejected. |
| **Infinite Speed** | $v = \pm\infty$ | `unknown` | `INVALID_SPEED` | `None` | `None` | Floating-point non-finite rejection. |
| **Invalid Free-Flow** | $v_{\text{free}} \le 0$ or non-finite | `unknown` | `INVALID_FREE_FLOW_SPEED` | `None` | `None` | Division by zero or negative baseline prevented. |
| **Invalid Length** | $L \le 0$ or non-finite | `unknown` | `INVALID_SEGMENT_LENGTH` | `None` | `None` | Negative or zero physical road segment rejected. |

---

## 6. Canonical Output Contract (`SegmentCongestionResult`)

Calculations emit the immutable, validated Pydantic model `SegmentCongestionResult`:

```python
class SegmentCongestionResult(BaseModel):
    segment_id: str
    forecast_timestamp: Optional[datetime] = None
    forecast_horizon_minutes: Optional[int] = None
    predicted_speed_kmh: Optional[float] = None
    free_flow_speed_kmh: Optional[float] = None
    speed_ratio: Optional[float] = None
    congestion_level: CongestionLevel
    congestion_status: CongestionStatus
    segment_length_km: Optional[float] = None
    estimated_travel_time_minutes: Optional[float] = None
    free_flow_travel_time_minutes: Optional[float] = None
    estimated_delay_minutes: Optional[float] = None
    model_source: str = "UNSPECIFIED"
    source_geography: str = "UNSPECIFIED"
    is_valid: bool = True
    status_message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Provenance Guarantee:
- **No Fabricated Dimensions:** If segment length ($L$) or free-flow speed ($v_{\text{free}}$) are invalid ($\le 0.0$, non-finite, or missing), `segment_length_km`, `free_flow_speed_kmh`, and `free_flow_travel_time_minutes` remain `None`. The engine never substitutes hardcoded fallbacks (such as 1.0 km or 30.0 km/h) into the result contract.
- **No Fabricated Timestamps:** If timestamp is unprovided, it remains `None`.
- **No Fabricated Horizons:** Horizon remains `None` unless genuinely evaluating a multi-step forecast (e.g. 5, 15, 30, 60 min).
- **Explicit Geographic Separation:** `source_geography` must reflect the true origin:
  - `"Los Angeles County, CA (METR-LA Freeway Loop Detectors)"` for P4 benchmark predictions.
  - `"Tamil Nadu (Coimbatore-Avinashi Road)"` for P1 OSM road network edges.

---

## 7. Integration Adapter for P1 Road Network Edges

`CongestionIntelligenceEngine` provides `evaluate_road_network_edge()` which seamlessly accepts P1 `RoadNetworkGraph` edge dictionaries without modifying the underlying binary graph:
- Automatically converts OSM length in meters (`length_m`) to kilometres (`length_km = length_m / 1000.0`).
- Reads the validated `free_flow_speed_kmh` derived in P1 from OSM `maxspeed` or road classification hierarchy.
- Preserves OSM attributes (`highway_type`, `name`, `road_reference`, `speed_source`) in the result metadata.

---

## 8. Verification & Test Execution Results

All 26 dedicated unit tests were executed in `tests/p5/` alongside the full regression suite:

```powershell
.\.venv\Scripts\pytest -v tests/p5
```
**Test Results: 26 passed in 0.17s**

### Test Coverage Highlights:
1. `test_default_congestion_categories`: Verifies canonical speed ratios for low, moderate, high, and severe tiers.
2. `test_exact_threshold_boundaries`: Confirms exact threshold boundary classification at 0.80, 0.7999, 0.60, 0.5999, 0.40, and 0.3999.
3. `test_speed_exceeding_free_flow`: Confirms low congestion with `FASTER_THAN_FREE_FLOW` status.
4. `test_exact_zero_predicted_speed`: Confirms severe classification with `STOPPED` status and no ZeroDivisionError.
5. `test_missing_and_nan_predicted_speed`: Confirms None and NaN result in `UNKNOWN` and `MISSING_SPEED`.
6. `test_invalid_predicted_speed`: Confirms negative and infinite speeds are rejected safely.
7. `test_invalid_free_flow_speed`: Confirms non-positive and non-finite free-flow speeds are rejected safely.
8. `test_custom_configurable_thresholds`: Verifies dynamic threshold injection and boundary updates.
9. `test_invalid_threshold_configurations_raise_error`: Confirms ValueError on inverted or out-of-range configs.
10. `test_hand_computable_travel_times_and_delays`: Confirms hand-calculated travel times (10 km at 60, 30, 15, and 45 km/h).
11. `test_speed_exceeding_free_flow_preserves_signed_negative_delay`: Confirms signed negative delay (time savings).
12. `test_zero_predicted_speed_handling_no_division_by_zero`: Confirms travel time is `None` with `STOPPED` flag for 0 km/h.
13. `test_missing_and_nan_speed_handling`: Confirms robust handling of missing inputs.
14. `test_invalid_predicted_speed_handling`: Confirms invalid speed inputs do not compute bogus travel times.
15. `test_invalid_segment_length_handling`: Confirms non-positive lengths are caught.
16. `test_invalid_free_flow_speed_handling`: Confirms non-positive free-flow speeds are caught.
17. `test_output_contract_fields_and_types`: Verifies full canonical schema validation and typing.
18. `test_optional_unavailable_fields_remain_explicitly_none`: Verifies zero metadata fabrication.
19. `test_deterministic_repeated_calculations`: Confirms exact idempotency across repeated runs.
20. `test_batch_evaluation_processing`: Verifies multi-segment batch processing.
21. `test_road_network_edge_adapter_integration`: Verifies integration with P1 `RoadNetworkGraph` edge attributes.
22. `test_geographic_and_dataset_integrity_isolation`: Verifies separation between METR-LA and Tamil Nadu scopes.
23. `test_invalid_segment_length_does_not_fabricate_dimensions`: Verifies invalid lengths result in `None` dimensions without fabricating 1.0 km.
24. `test_invalid_free_flow_speed_does_not_fabricate_dimensions`: Verifies invalid free-flow speeds result in `None` dimensions without fabricating 30.0 km/h.
25. `test_both_dimensions_invalid_does_not_fabricate_dimensions`: Verifies concurrent dimension failures leave all metrics `None`.
26. `test_road_network_edge_adapter_missing_dimensions_fails_safely`: Verifies P1 edge adapter fails safely with `None` dimensions when attributes are absent.

### Full Regression Suite:
```powershell
.\.venv\Scripts\pytest -v
```
**Total: 83 passed in 58.12s** (P0: 5, P1: 17, P2: 15, P3: 13, P4: 20, P5: 26).

---

## 9. Known Limitations & Roadmap for P6

- **Static vs Dynamic:** P5 performs point-in-time calculation given speed inputs; it does not dynamically propagate traffic congestion across time steps or graph paths.
- **Routing Engine Integration:** P5 provides the travel time and delay calculation primitives. P6 will integrate these into the NetworkX / Dijkstra routing engine as dynamic edge weight cost penalties.
- **Corridor Calibration:** Free-flow speeds currently rely on OSM `maxspeed` or Indian highway hierarchy defaults. Empirical calibration using local Tamil Nadu probe data will refine these baselines when local data becomes available.
- **No P6 Functionality Implemented:** P6 has not been started.
