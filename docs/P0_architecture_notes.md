## 1. System Vision & Core User Workflow

### 1.1 Product Principle & User-First Mission
**TrafficRoute AI is an end-user predictive transportation product, NOT a traffic-data visualization or academic dataset showcase.**

The system exists to solve a concrete, everyday driver and commuter problem: given an Origin and Destination in Tamil Nadu, provide actionable, predictive route intelligence that helps the user make smarter travel decisions (which route to take, when to leave, how much delay to expect, and where bottlenecks will occur).

#### Non-Goals & Anti-Patterns (Strictly Prohibited):
* ❌ **NO Raw Dataset Dashboards:** No generic charts plotting historical sensor curves for arbitrary sensor IDs.
* ❌ **NO Meaningless Traffic Tables:** No tables listing raw sensor telemetry, internal node IDs, or unaggregated CSV dumps.
* ❌ **NO Decorative Analytics:** No vanity metrics that do not directly inform the user's route choice or travel time.
* ❌ **NO Fabricated Live Claims:** No pretending to have live physical sensors where only historical or calibrated baseline profiles exist.

Every visual element, metric, and API endpoint must directly answer a specific user travel question.

---

### 1.2 Core User Questions & Architectural Capabilities

| User Question | Mathematical / System Resolution | Actionable UI Output |
| :--- | :--- | :--- |
| **"How long will my trip probably take?"** | Forward spatio-temporal speed integration: $T(R, t_{\text{dep}}) = \sum \frac{L(e)}{\hat{v}(e, t_{\text{arr}}(e))}$ | Predicted Total Travel Time (e.g., *48 mins*) |
| **"How much delay should I expect?"** | Congestion delay calculation: $\Delta T = T(R, t_{\text{dep}}) - T_{\text{free}}(R)$ | Expected Delay vs Free-Flow (e.g., *+14 mins delay*) |
| **"Which route has lower predicted travel time?"** | Multi-route candidate evaluation: $\Delta T_{R_1, R_2} = T(R_1) - T(R_2)$ | Side-by-side route comparison card with delta |
| **"Where is congestion expected on my route?"** | Segment Congestion Index ($CI \ge 0.50$) mapped to geometry | Color-coded route polyline highlights on the map |
| **"Is traffic expected to worsen before/during my trip?"** | Multi-horizon speed trend: $\frac{\partial \hat{v}(e, t)}{\partial t}$ | Trend badge (*"Traffic building up on NH-544"*) |
| **"What alternative routes are available?"** | $k$-Shortest path generation (Yen's / A* / Contraction Hierarchies) | Distinct route options (e.g., *Via NH-544* vs *Via Annur*) |
| **"What road segments are responsible for the delay?"** | Edge bottleneck attribution: $\text{top-}k \left( \Delta \tau(e) \right)$ | Named bottleneck callouts (*"Chitra Junction: +8m delay"*) |
| **"What departure time would reduce travel time?"** | Temporal departure sweep: $\arg\min_{t \in [t_0, t_0+2h]} T(R, t)$ | Departure optimizer (*"Leaving at 08:45 saves 12 mins"*) |

---

### 1.3 End-to-End User Workflow
The system executes a streamlined, end-to-end pipeline focused entirely on the user's journey:

```
[ User Input: Origin + Destination + Planned Departure Time ]
                           │
                           ▼
[ 1. Geocoding & Coordinate Resolution (Nominatim / Photon) ]
                           │
                           ▼
[ 2. Road Network Map Matching (Nearest Node / Edge Snapping on OSM Graph) ]
                           │
                           ▼
[ 3. Dynamic Route Generation (k-Alternative Candidate Paths via Graph Engine) ]
                           │
                           ▼
[ 4. Spatio-Temporal Speed Prediction (Edge Velocity Forecasts: \hat{v}(e, t)) ]
                           │
                           ▼
[ 5. Dynamic Route Travel Time Integration & Congestion Intelligence ]
                           │
                           ▼
[ 6. Route Comparison, Bottleneck Detection & Explainability Engine ]
                           │
                           ▼
[ 7. Actionable Map-Based Visual Presentation (React UI) ]
```

---

## 2. Mathematical Formulation

### 2.1 Road Network Graph Representation
The physical road network is modeled as a directed spatial multigraph:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$$

Where:
* $\mathcal{V} = \{v_1, v_2, \dots, v_n\}$ is the set of vertices (intersections, dead-ends, highway interchanges), where each vertex has geographic coordinates $(\text{lat}_v, \text{lon}_v)$.
* $\mathcal{E} = \{e_1, e_2, \dots, e_m\} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of directed edges (road segments).
* Each edge $e = (u, v) \in \mathcal{E}$ possesses static structural attributes:
  * $L(e) \in \mathbb{R}^+$: Geodesic segment length in meters.
  * $C(e) \in \{\text{motorway}, \text{trunk}, \text{primary}, \text{secondary}, \text{tertiary}, \text{residential}, \dots\}$: Functional highway classification.
  * $v_{\text{free}}(e) \in \mathbb{R}^+$: Free-flow / baseline speed limit (km/h).
  * $\text{lanes}(e) \in \mathbb{N}^+$: Number of lanes.
  * $\text{geom}(e)$: Ordered polyline sequence of WGS84 coordinate pairs.

---

### 2.2 Dynamic Traffic Speed Prediction Formulation
Let $\mathbf{X}_t \in \mathbb{R}^{|\mathcal{E}| \times F}$ represent the traffic state matrix across all road segments at time step $t$, where features $F$ may include historical average speed, congestion ratio, time-of-day encoding, day-of-week encoding, and road structural properties.

The primary ML objective is to learn a predictive mapping function $f_\theta$:

$$\hat{\mathbf{V}}_{t+1 : t+H} = f_\theta\left( \mathbf{X}_{t-P+1 : t}, \mathcal{G} \right)$$

Where:
* $P$ is the number of historical observation time steps (e.g., past 60 minutes in 5-minute or 15-minute intervals).
* $H$ is the prediction horizon (e.g., forecasting speeds at $t + 15\text{ min}, t + 30\text{ min}, t + 45\text{ min}, t + 60\text{ min}$).
* $\hat{v}(e, t_k) \in \mathbb{R}^+$ denotes the predicted vehicle speed on edge $e$ at future time $t_k$.

---

### 2.3 Dynamic Route Travel Time Estimation
Given an origin $O \in \mathcal{V}$ and destination $D \in \mathcal{V}$, a candidate route $R$ consists of an ordered sequence of connected edges:

$$R = \left( e_1 = (u_1, u_2), e_2 = (u_2, u_3), \dots, e_k = (u_k, u_{k+1}) \right)$$

Because vehicle speed varies dynamically as the vehicle traverses the network over time, travel time cannot be computed using a static snapshot. Instead, it must be evaluated via **forward spatio-temporal simulation**:

1. **Departure Time at Origin:** $t_{\text{arr}}(e_1) = t_{\text{dep}}$
2. **Sequential Edge Traversal:** For each edge $e_i \in R$ ($i = 1, \dots, k$):
   $$\tau(e_i) = \frac{L(e_i)}{\hat{v}\left(e_i, t_{\text{arr}}(e_i)\right)}$$
   $$t_{\text{arr}}(e_{i+1}) = t_{\text{arr}}(e_i) + \tau(e_i)$$
3. **Total Route Estimated Travel Time (ETT):**
   $$T(R, t_{\text{dep}}) = \sum_{i=1}^k \tau(e_i) = \sum_{i=1}^k \frac{L(e_i)}{\hat{v}\left(e_i, t_{\text{arr}}(e_i)\right)}$$

---

### 2.4 Congestion Delay & Level of Service (LoS)
To generate actionable insights and explanations for the user, each edge and route is evaluated against free-flow baselines:

1. **Delay on Segment:**
   $$\Delta \tau(e) = \tau(e) - \frac{L(e)}{v_{\text{free}}(e)}$$

2. **Congestion Index ($CI$):**
   $$CI(e, t) = 1.0 - \min\left(1.0, \frac{\hat{v}(e, t)}{v_{\text{free}}(e)}\right) \in [0.0, 1.0]$$
   * $0.00 \le CI < 0.20$: **Free Flow** (Green)
   * $0.20 \le CI < 0.50$: **Moderate Congestion** (Yellow)
   * $0.50 \le CI < 0.75$: **Heavy Congestion / Bottleneck** (Orange)
   * $0.75 \le CI \le 1.00$: **Severe Gridlock** (Red)

3. **Total Route Delay:**
   $$\text{Delay}(R) = T(R, t_{\text{dep}}) - T_{\text{free}}(R)$$

---

## 3. Modular Decoupling Architecture

To ensure the system is maintainable, scientifically sound, and adaptable across data environments, the architecture strictly decouples the **Routing Engine**, **Graph Topology**, **Speed Provider Layer**, and **Presentation Layer**.

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                     │
│    - React Map UI (Leaflet / MapLibre)                      │
│    - Route comparison cards, bottleneck callouts, ETT       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────▼──────────────────────────────┐
│                     FastAPI Backend                         │
│    - /api/v1/geocode (Origin, Destination -> Lat/Lon)       │
│    - /api/v1/routes (k-paths with predicted ETT & stats)    │
│    - /api/v1/explain (Congestion reasons, bottleneck nodes) │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│     Graph & Routing Engine   │       │     Speed Provider Layer     │
│ - OSM Network Graph (OSMnx)  │       │ (Pluggable Speed Interface)  │
│ - Node snapping & k-shortest │◄──────┤ - Baseline Heuristic Profile │
│   paths (Yen's / Dijkstra)   │ Dynamic│ - ML Spatio-Temporal Model  │
│ - Dynamic edge cost evaluation│ Weights│ - External Traffic API Feed │
└──────────────────────────────┘       └──────────────────────────────┘
```

### Pluggable Speed Provider Interface
By defining an abstract Python interface `BaseSpeedProvider`:

```python
class BaseSpeedProvider(ABC):
    @abstractmethod
    def get_predicted_speeds(
        self, 
        edge_ids: list[tuple[int, int, int]], 
        target_time: datetime
    ) -> dict[tuple[int, int, int], float]:
        """
        Returns mapping of (u, v, key) -> predicted_speed_kmh
        """
        pass
```

The system can seamlessly operate under three operational modes:
1. **Mode A (Empirical ML Predictor):** Uses trained GNN/LightGBM weights on corridors where continuous training data is available.
2. **Mode B (Calibrated Diurnal Baseline):** Uses time-of-day, day-of-week, and road-classification speed reduction functions calibrated to published Indian urban traffic studies (IRC / MoRTH standards).
3. **Mode C (Live / Commercial API Provider):** Ingests live or historical flow vectors from TomTom / HERE / Mappls if API credentials are provided.

---

## 4. Geographic Scoping Strategy

| Tier | Geographic Scope | Road Network Source | Speed Provider Strategy |
| :--- | :--- | :--- | :--- |
| **Pilot Corridor / Core Demonstration** | **Coimbatore Urban & Coimbatore $\leftrightarrow$ Avinashi (NH-544)** | High-precision OSM subgraph (~5,000 nodes) | High-fidelity calibrated spatio-temporal profiles + empirical validation |
| **Regional Urban Focus** | **Chennai Metropolitan & OMR IT Corridor** | OSM subgraph (~15,000 nodes) | Calibration informed by IIT Madras SPT dataset + arterial profiles |
| **State-Wide Network** | **Tamil Nadu (All 38 Districts)** | Complete Geofabrik Tamil Nadu OSM extract | Macroscopic highway hierarchy speed baseline + dynamic routing |

---

## 5. Summary of Architectural Feasibility
* **Graph Connectivity & Topography:** 100% Feasible via OpenStreetMap (GREEN).
* **Multi-Route Identification:** 100% Feasible via Yen's Algorithm / A* / OSRM (GREEN).
* **Dynamic Travel Time Integration:** 100% Feasible mathematically (GREEN).
* **Pluggable Decoupling:** Ensures zero dependency on fake/synthetic data while maintaining full system functionality across all milestones.

---

## 6. Graceful & Honest Degradation Policy

To maintain absolute transparency and real-world credibility with the end-user:

1. **Explicit Data Confidence Indicators:**
   Every route response from the backend will include a `data_confidence` metadata block:
   ```json
   {
     "route_id": "R1_NH544",
     "total_distance_km": 42.6,
     "estimated_travel_time_mins": 48.2,
     "delay_mins": 11.4,
     "data_provenance": {
       "coverage_level": "PILOT_CORRIDOR_HIGH_FIDELITY",
       "speed_model": "CALIBRATED_DIURNAL_SPATIO_TEMPORAL",
       "is_live_sensor": false,
       "confidence_score": 0.88,
       "explanation": "Predicted using Coimbatore-Avinashi calibrated peak-hour speed model."
     }
   }
   ```
2. **Degradation Tiers in the User Interface:**
   * **Tier A (Empirical / Calibrated Pilot Corridors):** Full predictive intelligence, dynamic departure optimizer, and named bottleneck delays.
   * **Tier B (State-Wide Rural Roads / Uncalibrated Links):** Road-hierarchy free-flow routing with static delay factors, explicitly displaying a badge: *"Static Highway Profile (Predictive model not active for this rural segment)"*.
3. **Zero Synthetic Falsification:**
   The UI will never generate animated fake "live sensor dots" or synthetic traffic jams. Missing data will always be represented as unmonitored or baseline free-flow with explicit user disclosure.

