# TrafficRoute AI — P0 Dataset & Problem Feasibility

**Project:** TrafficRoute AI (Tamil Nadu AI-Based Predictive Traffic & Route Intelligence System)  
**Milestone:** P0 — Dataset & Problem Feasibility  
**Status:** Completed & Validated  
**Date:** September 2026  

---

## 1. Project Objective & Core Product Principles

### 1.1 Product Mission: Actionable End-User Route Intelligence
**TrafficRoute AI is designed as a genuinely useful, professional end-user transportation product, NOT a traffic-data visualization project or academic dataset showcase.**

The final system solves a concrete, real-world user problem: A driver, commuter, or fleet operator inputs an origin and destination in Tamil Nadu and receives actionable predictive route intelligence to optimize their journey.

#### The system is engineered to answer core user questions:
1. **"How long will my trip probably take?"** $\to$ Dynamic estimated travel time based on future segment speeds.
2. **"How much delay should I expect?"** $\to$ Quantified delay against free-flow baseline (*"+12 mins delay due to peak traffic"*).
3. **"Which route has lower predicted travel time?"** $\to$ Direct comparison of candidate routes with expected time delta.
4. **"Where is congestion expected on my route?"** $\to$ Segment-by-segment congestion mapping along the chosen path.
5. **"Is traffic expected to worsen before/during my journey?"** $\to$ Predictive trend analysis across departure time windows.
6. **"What alternative routes are available?"** $\to$ Topologically distinct alternative paths.
7. **"What road segments are responsible for the predicted delay?"** $\to$ Named bottleneck attribution (*e.g., Hope College junction, KMCH / Chitra junction on NH-544*).
8. **"What departure time would reduce predicted travel time?"** $\to$ Actionable departure recommendation when temporal data permits.

#### Anti-Patterns & Strict Non-Goals:
* ❌ **NO Raw Sensor Dashboards:** No plotting unaggregated speed curves for abstract sensor IDs.
* ❌ **NO Generic Decorative Charts:** Visualizations exist ONLY when they help the user understand or act on their route.
* ❌ **NO Fabricated Capabilities or Live Data:** Honest degradation when high-resolution prediction is unmonitored; no synthetic fake data.

The primary Machine Learning target is **future road-segment traffic speed (km/h)**.

---

## 2. Required Data Categories

To build this end-to-end predictive routing system honestly and robustly without relying on fabricated data, the project requires four distinct data categories:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           REQUIRED DATA CATEGORIES                      │
├──────────────────────────┬──────────────────────────────────────────────┤
│ 1. Road Network Data     │ Road geometries, node intersections, length,  │
│                          │ hierarchy, one-way tags, connectivity graph. │
├──────────────────────────┼──────────────────────────────────────────────┤
│ 2. Traffic Speed Data    │ Empirical time-series vehicle speeds, probe  │
│                          │ velocities, or calibrated speed profiles.    │
├──────────────────────────┼──────────────────────────────────────────────┤
│ 3. Geocoding / POI Data  │ Place name resolution (e.g. Coimbatore       │
│                          │ Railway Station -> 11.0016° N, 76.9667° E).  │
├──────────────────────────┼──────────────────────────────────────────────┤
│ 4. Ground-Truth / Bench- │ Validation datasets for algorithm accuracy  │
│    marking Data          │ (e.g., PEMS-BAY, IITM-SPT, TomTom/Google).   │
└──────────────────────────┴──────────────────────────────────────────────┘
```

---

## 3. Road-Network Feasibility (OpenStreetMap & Geofabrik)

### Feasibility Status: **GREEN (Fully Feasible)**

The physical road network for the entire state of Tamil Nadu can be acquired and modeled using **OpenStreetMap (OSM)** through Geofabrik regional extracts and the `OSMnx` Python ecosystem.

### Key Findings:
1. **State-Wide Extraction:** Geofabrik provides daily updated regional extracts for Tamil Nadu (`tamil-nadu-latest.osm.pbf`, compressed size: ~180 MB, raw XML: ~2.5 GB).
2. **Topological Completeness:** The dataset covers all 38 districts of Tamil Nadu, spanning National Highways (e.g., NH-544, NH-44, NH-45), State Highways (SH), Major District Roads (MDR), urban arterials, and local streets.
3. **Graph Ingestion:** Using `osmnx` and `networkx`, the OSM vector data can be ingested directly into a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:
   * **Nodes ($\mathcal{V}$):** Intersections, junctions, roundabouts, highway ramps with exact WGS84 coordinates.
   * **Edges ($\mathcal{E}$):** Road segments possessing unique OSM IDs, precise `LineString` geometries, and accurate segment lengths (in meters).
4. **Attribute Availability & Quality:**
   * *Road Geometry:* **Excellent** (continuous sub-meter vector polylines).
   * *Road Name & Classification:* **Excellent** (`name`, `ref`, `highway=trunk|primary|secondary|tertiary|residential`).
   * *Directionality:* **Excellent** (`oneway=yes`, `oneway=-1`, dual-carriageway splits).
   * *Speed Limits (`maxspeed`):* **Moderate / Incomplete**. Tagged on major National Highways (e.g., 80/100 km/h), but frequently omitted on secondary/urban roads. This requires a standard heuristic speed-limit lookup table mapped to the `highway` classification hierarchy for default free-flow speeds.

---

## 4. Traffic-Data Feasibility

### Feasibility Status: **YELLOW (Feasible with Defined Scope & Multi-Tier Strategy)**

A major objective of P0 was to rigorously determine whether continuous historical traffic speed datasets exist publicly for all roads across Tamil Nadu.

### Key Findings & Realities:
1. **Tamil Nadu-Wide Open Sensor Network Reality:**
   * Unlike California's Caltrans PeMS system (which has tens of thousands of permanent inductive loop detectors streaming 5-minute speed/flow data), **there is NO publicly open, state-wide 24/7 road-sensor speed dataset covering all of Tamil Nadu**.
   * Open government portals (`data.gov.in`, `data.tn.gov.in`, `opencity.in`) provide static administrative transport statistics (accident records, vehicle registrations, depot revenues), but **zero** dynamic time-series segment-speed telemetry.
2. **Local Academic & Trajectory Datasets:**
   * **SPT (Seerattra Pokkuvarathu Tharavu) Dataset (IIT Madras / TU Dresden):** High-precision UAV drone vehicle trajectories over a 560m section on OMR, Chennai. Provides ground-truth empirical speed distributions and acceleration profiles in disordered Indian traffic, but cannot cover state-wide inter-city corridors.
   * **IITM-HeTra (IIT Madras):** CCTV video dataset for vehicle detection/classification in Chennai. Useful for vision-based queue counting at junctions, but not a time-series GIS road network speed dataset.
   * **CHETNA-Road (IIT Bombay / LSCE):** Daily gridded emissions data (500m resolution) across 15 Indian cities. Too coarse for 15-minute predictive routing.
   * **Uber Movement (Historical Archives):** Provided hourly aggregated speeds on OSM segments in selected metros (Bangalore, Mumbai, New Delhi; limited Chennai corridor literature). Discontinued in June 2023, with no coverage for Coimbatore/Avinashi.
3. **Open Spatio-Temporal Algorithm Benchmarks (PEMS-BAY / METR-LA):**
   * Standard benchmark datasets provide continuous 5-minute segment speeds on real road graphs. They are vital for verifying that our Spatio-Temporal Graph Neural Network (STGCN / DCRNN / Graph WaveNet) code and loss functions converge correctly before deployment.

---

## 5. Dataset Comparison Table

| Dataset | Provider / Organization | Geographic Coverage | Time Coverage | Temporal Resolution | Spatial Resolution | Speed Available | Traffic Volume | Coordinates | Road Identity | License | Usefulness in Project | Major Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenStreetMap (OSM) TN Extract** | Geofabrik / OSM Community | Tamil Nadu (All 38 Districts) | Continuous (Daily snapshot) | Static | Road segment (sub-meter) | Static limits only | No | Yes (WGS84) | Yes (OSM Way & Node ID) | Open Database License (ODbL) 1.0 | **Critical**: Base road network graph, geometry, and topology | No dynamic traffic speeds; incomplete maxspeed tags |
| **SPT Chennai Trajectory Dataset** | IIT Madras & TU Dresden | Chennai (560m OMR corridor) | Peak & off-peak survey sessions | Sub-second (video frame rate) | Microscopic vehicle coordinates | **Yes** (Instantaneous velocity per vehicle) | Yes (Class counts) | Yes (DGPS georegistered) | Partial (Fixed 560m corridor) | Academic research citation | **High for Micro Dynamics**: Authentic empirical speed distributions by vehicle class | Highly localized (560m only); cannot train city-wide macro graphs |
| **IITM-HeTra Dataset** | IIT Madras (RBC-DSAI) | Chennai (Selected junctions) | Multi-hour video captures | Video frame rate (25-30 fps) | Camera field of view | Indirect (via object tracking) | Yes (Vehicle detection/labels) | No (Pixel bounding boxes) | No (Camera IDs only) | Academic / Research | **Moderate**: Junction vehicle counting and queue density | Video-only; not mapped to GIS road network edges |
| **CHETNA-Road Dataset** | IIT Bombay / LSCE France | 15 Indian Cities (incl. Chennai) | 2019–2021 (Daily) | Daily | 500m $\times$ 500m grid raster | Indirect (Derived COPERT proxy) | Indirect (Daily emission flux) | Yes (Grid centroids) | No (Grid cells, not OSM edges) | CC-BY 4.0 Open Academic | **Low**: Macro urban congestion hotspot context | Daily time step and 500m grid too coarse for routing |
| **Uber Movement (Historical Archives)** | Uber Technologies Inc. | Selected Indian Metros (Discontinued 2023) | 2017–2020 | Hourly aggregates | OSM road segments / H3 hexes | **Yes** (Mean & std segment speeds) | No | Yes (OSM links) | Yes (OSM Way IDs) | Historical Open Data | **Moderate**: Calibrates diurnal peak-hour speed drop curves | Discontinued; no updates; no Coimbatore-Avinashi coverage |
| **PEMS-BAY & METR-LA Benchmarks** | Caltrans PeMS / Los Angeles MTA | California, USA (207 & 325 sensors) | 4–6 months continuous | 5-minute continuous | Sensor / highway segment | **Yes** (Continuous 5-min speed matrix) | Yes (Flow & Occupancy) | Yes (WGS84 Lat/Lon) | Yes (Sensor ID + Adjacency Matrix) | Public Domain / Open Research | **High for Verification**: Gold standard to validate STGCN / DCRNN models | Non-Indian geography; cannot be claimed as Tamil Nadu traffic |
| **TomTom Traffic Stats & Flow API** | TomTom International BV | Pan-India & Complete Tamil Nadu | 2015–present (Live & Historical) | 5-minute to 1-hour bins | Road segment link level | **Yes** (Harmonic avg, current, free-flow) | Yes (Probe density index) | Yes (WGS84 / OpenLR) | Yes (Standard segment IDs) | Commercial / Freemium (2.5k req/day) | **High**: Commercial-grade empirical validation for TN | SaaS licensing restrictions on bulk offline training extraction |
| **HERE Traffic Analytics & Flow** | HERE Technologies | Pan-India & Tamil Nadu | 2021–present | 15-minute / Hourly | Road link level | **Yes** (Actual speed, jam factor) | Yes (Probe counts & confidence) | Yes (WGS84 links) | Yes (HERE Link IDs) | Commercial / Freemium (~250k req/mo) | **High**: Highway speed and delay validation | Commercial license; bulk historical downloads require enterprise tier |
| **Mappls (MapmyIndia) Traffic API** | CE Info Systems Ltd | Pan-India & Tamil Nadu | Live & Predictive historical models | Query-time dynamic | Route and segment level | **Yes** (Live & predictive route speeds) | Indirect (Congestion level) | Yes (WGS84) | Yes (Mappls segment IDs) | Commercial Proprietary / Trial credits | **High**: Localized Indian route and delay benchmark | Closed API; outputs route ETAs rather than open CSV tensors |
| **Google Maps Routes API** | Google LLC | Global & Complete Tamil Nadu | Live & Predictive departure_time | Query-time dynamic | Route / Step level | Indirect (Duration in traffic) | No | Yes (WGS84) | No (Internal routing graph) | Commercial / $200 monthly credit | **High for Benchmarking**: Industry ground-truth validation | ToS strictly forbids scraping/storing for training custom ML models |
| **Tamil Nadu Open Data Portals** | Govt of TN / OpenCity.in | Tamil Nadu & Chennai | Annual / Monthly reports | Monthly / Annual | District / Municipal level | **No** | Indirect (Registrations, crashes) | Coarse (District level) | No | Government Open Data License (GODL) | **Low**: Transport policy and demographic context | No continuous sensor-level speed telemetry |

---

## 6. Historical Traffic API & Data Service Investigation

We investigated four leading commercial and developer platforms to assess their technical viability, India/Tamil Nadu coverage, developer tier accessibility, and licensing terms:

### 1. TomTom Platform (Traffic Stats & Flow API)
* **Coverage:** Complete Tamil Nadu coverage, including NH-544 (Coimbatore–Avinashi), NH-44, Chennai OMR/GST, Madurai, Trichy, Salem, and urban arterials.
* **Capabilities:** 
  * *Traffic Flow API:* Real-time speed, free-flow speed, and travel time per segment.
  * *Traffic Stats (MOVE Portal / API):* Historical speed profiles, travel times, and congestion percentiles by hour of day and day of week.
* **Developer Access:** Freemium tier offers **2,500 free non-tile API calls/day** with no credit card required.
* **Licensing Constraint:** Terms of Service permit real-time query consumption and dashboard visualization, but prohibit bulk scraping to build an independent, competing offline commercial foundation dataset.

### 2. HERE Technologies (Traffic Analytics & Flow API)
* **Coverage:** India and Tamil Nadu highway network supported with probe-based historical records from January 2021 to present.
* **Capabilities:** Segment-level speed, jam factor (0–10 scale), and historical speed distribution matrices.
* **Developer Access:** Freemium plan provides ~250,000 transactions/month for standard location APIs.

### 3. Mappls (MapmyIndia) Routing & Predictive ETA API
* **Coverage:** Deepest local road mapping in India, capturing local flyovers, bypasses, and state corridors.
* **Capabilities:** Offers `predictive_eta` with a `date_time` parameter to project travel durations using historical traffic patterns.
* **Developer Access:** Mappls Console offers evaluation credits for registered developers.

### 4. Google Maps Platform (Routes / Distance Matrix API)
* **Coverage:** Ubiquitous across all Tamil Nadu roads.
* **Capabilities:** High-accuracy predictive travel duration via the `departure_time` parameter.
* **Licensing Constraint:** Section 3.2.3 of Google Maps Platform ToS strictly prohibits storing, scraping, or using outputs to train independent ML models. It is strictly suitable only for live ground-truth comparison and benchmarking.

---

## 7. Origin-Destination Routing Feasibility

### Feasibility Status: **GREEN (Fully Feasible)**

Can we support Origin $\to$ Destination routing (e.g., *Coimbatore Railway Station $\to$ Avinashi*)? **Yes, 100% feasible.**

### Implementation Mechanics:
1. **Geocoding:** Nominatim (OSM's open geocoding engine) or Photon resolves input text queries (*"Coimbatore Railway Station"*, *"Avinashi, Tamil Nadu"*) into exact coordinates $(11.0016^\circ\text{N}, 76.9667^\circ\text{E})$ and $(11.1932^\circ\text{N}, 77.2690^\circ\text{E})$.
2. **Nearest Node Snapping:** Spatial KD-Tree or BallTree snaps the resolved coordinates to the nearest valid entry/exit vertices $u_{\text{orig}}, v_{\text{dest}} \in \mathcal{V}$ on the Tamil Nadu OSM road graph.
3. **Multi-Route Generation:** 
   * Shortest path by physical distance (Dijkstra / A* using edge length $L(e)$ as weight).
   * Fast free-flow route ($L(e) / v_{\text{free}}(e)$).
   * $k$-Alternative candidate paths via Yen's Algorithm or Contraction Hierarchies (e.g., NH-544 direct route vs. via Annur/Somanur bypass).
4. **Dynamic Edge Cost Evaluation:** The routing engine evaluates edge traversal costs using predicted dynamic speeds $\hat{v}(e, t_{\text{arr}})$ rather than static distances.

---

## 8. ML Prediction Feasibility

### Feasibility Status: **GREEN (Feasible with Decoupled Architecture)**

Can we train an ML model to predict future road-segment speed? **Yes.**

### ML Problem Formulation:
* **Input:** Past $P$ time-step observations of edge features $\mathbf{X}_{t-P+1:t}$ (historical speed ratios, time of day $[0..23]$, day of week $[0..6]$, holiday indicator, road classification, lane count, segment length) and road network graph topology $\mathcal{G}$.
* **Target:** Future speed matrix $\hat{\mathbf{V}}_{t+1:t+H}$ across horizons $H \in \{15\text{ min}, 30\text{ min}, 45\text{ min}, 60\text{ min}\}$.
* **Model Families to Evaluate in Later Milestones (P3/P4):**
  1. *Baseline Predictors:* Historical Average (HA), Autoregressive Moving Average (ARIMA), Gradient Boosted Trees (LightGBM / XGBoost) with spatio-temporal lag features.
  2. *Spatio-Temporal Graph Neural Networks (ST-GNNs):* Spatio-Temporal Graph Convolutional Networks (STGCN), Diffusion Convolutional Recurrent Neural Networks (DCRNN), Graph WaveNet.

### Travel Time Integration ($T = \sum L_i / \hat{v}_i$):
* Travel time estimation is mathematically exact: forward simulation accumulates traversal duration across sequential route segments, updating the simulated arrival time at each successive edge to query the appropriate temporal speed bucket.

---

## 9. Geographic Coverage Limitations

To maintain scientific integrity and honesty, TrafficRoute AI will clearly define its geographic scope across three tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GEOGRAPHIC COVERAGE TIERS                       │
├────────────────────────────────┬───────────────────────────────────────┤
│ Tier 1: Core Pilot Corridor    │ Coimbatore Urban & Coimbatore ↔      │
│         (Highest Fidelity)     │ Avinashi (NH-544 corridor). Full      │
│                                │ micro-segment graph, calibrated       │
│                                │ diurnal curves, multi-route engine.  │
├────────────────────────────────┼───────────────────────────────────────┤
│ Tier 2: Regional Urban Focus   │ Chennai Metropolitan Area & OMR       │
│                                │ IT Corridor. Calibrated using IIT    │
│                                │ Madras empirical research baselines.  │
├────────────────────────────────┼───────────────────────────────────────┤
│ Tier 3: State-Wide Network     │ Complete Tamil Nadu (38 Districts).   │
│         (Macroscopic Routing)  │ Full OSM topological graph routing    │
│                                │ with road-hierarchy speed baselines.  │
└────────────────────────────────┴───────────────────────────────────────┘
```

> [!IMPORTANT]
> We will **never** claim to possess 24/7 live physical hardware sensor feeds on every rural village road in Tamil Nadu. All speeds will be transparently attributed to their source: empirical pilot corridors, calibrated diurnal baselines, or integrated API feeds.

---

## 10. Data Licensing & Ethical Considerations

1. **OpenStreetMap Data:** Licensed under **ODbL 1.0**. Requires proper attribution (`© OpenStreetMap contributors`) in both backend responses and React UI footer. Allows unrestricted commercial, academic, and derivative use.
2. **Academic Datasets (IIT Madras SPT / IITM-HeTra):** Used strictly under academic research citation terms.
3. **Benchmark Datasets (PEMS-BAY / METR-LA):** Public domain / open research licenses.
4. **Commercial Traffic APIs (TomTom / HERE / Mappls / Google Maps):** Used strictly in compliance with developer terms (query-time computation, rate-limit compliance, no unauthorized mass scraping for offline distillation).

---

## 11. Recommended Data Strategy

We recommend a **Modular Three-Tier Hybrid Data Strategy**:

```
                              ┌────────────────────────────────────────┐
                              │       OpenStreetMap Base Graph         │
                              │     (100% Real TN Road Network)        │
                              └───────────────────┬────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌──────────────────────────────────┐              ┌──────────────────────────────────┐
        │     Empirical Pilot Corridors    │              │   Calibrated Macroscopic Model   │
        │ - Coimbatore ↔ Avinashi (NH-544) │              │ - All 38 Tamil Nadu Districts    │
        │ - Chennai OMR Arterial           │              │ - Diurnal Peak/Off-Peak Functions│
        │ - Validated via SPT & Benchmarks │              │ - Parameterized by Road Class    │
        └──────────────────────────────────┘              └──────────────────────────────────┘
```

1. **Foundation Graph:** Ingest 100% real OpenStreetMap vector road network for Tamil Nadu using `OSMnx`.
2. **Spatio-Temporal Model Calibration:**
   * Benchmark and verify GNN/ML architectures on standard open spatio-temporal datasets (PEMS-BAY / METR-LA).
   * Calibrate Indian traffic speed-flow curves using empirical findings from IIT Madras (SPT dataset) and IRC (Indian Roads Congress) standards for urban/highway peak congestion drops.
3. **Pluggable Speed Provider:** Decouple the speed prediction layer so the system can switch dynamically between:
   * (a) Calibrated ML model predictions,
   * (b) Statistical diurnal baseline curves, and
   * (c) Live TomTom/Mappls API telemetry when API credentials are provided.

---

## 12. Risks and Mitigation Strategies

| Risk | Severity | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Absence of open state-wide sensor speed feeds** | High | Transparently adopt a multi-tier geographic scope; use calibrated diurnal models parameterized by road class; do not fabricate fake telemetry. |
| **Missing `maxspeed` tags in OSM** | Medium | Build a comprehensive road-classification fallback speed lookup table based on Indian highway standards (Motorway: 100, Trunk/NH: 80, Primary: 60, Secondary: 40, Residential: 25 km/h). |
| **Commercial API rate limits / costs** | Medium | Architect a strict caching and rate-limiting proxy; keep the core routing and ML engine completely independent of paid third-party APIs. |
| **Graph scale & memory constraints (TN-wide)** | Medium | Partition the graph into subgraphs (e.g., Coimbatore metropolitan graph, Chennai urban graph, State Highway backbone); use NetworkX MultiDiGraph with Contraction Hierarchies. |

---

## 13. P0 Conclusion & Requirement Classification

Each core capability of the proposed TrafficRoute AI system is classified below:

| Project Requirement | Classification | Justification & Evidence |
| :--- | :--- | :--- |
| **1. Tamil Nadu Road Network Graph** | **GREEN** | 100% supported by OpenStreetMap Geofabrik Tamil Nadu extract; includes all nodes, edges, geometry, lengths, and highway hierarchy. |
| **2. Origin-Destination Geocoding** | **GREEN** | Supported by Nominatim / Photon open geocoding for all Tamil Nadu cities, towns, and landmarks. |
| **3. Graph Routing Engine (k-Paths)** | **GREEN** | Dijkstra, A*, Yen's k-shortest paths, and Contraction Hierarchies fully supported via NetworkX/OSMnx. |
| **4. Dynamic Travel Time Estimation** | **GREEN** | Mathematically rigorous forward edge-speed accumulation ($T = \sum L_i / \hat{v}_i$). |
| **5. Algorithmic ML Architecture Verification**| **GREEN** | Verifiable using open spatio-temporal benchmarks (PEMS-BAY / METR-LA) with STGCN/DCRNN/LightGBM. |
| **6. Pilot Corridor Empirical Speed Modeling** | **GREEN** | Feasible for Coimbatore $\leftrightarrow$ Avinashi (NH-544) and Chennai OMR using calibrated diurnal and IIT Madras research profiles. |
| **7. Commercial API Integration (TomTom/HERE)**| **YELLOW** | Feasible via free developer tiers (2,500 req/day); requires API key configuration and rate-limit handling. |
| **8. State-Wide 24/7 Real-Time Sensor Feeds** | **RED** | **Currently not supported as open public data.** Mitigated by using calibrated macroscopic diurnal speed profiles for state-wide routing. |

---

## 14. Exact Requirements for Milestone P1

With the successful completion and validation of P0, the project is ready to proceed to **Milestone P1 (Tamil Nadu Road Network)**.

### P1 Prerequisites & Tasks:
1. **Network Extraction:**
   * Extract the Tamil Nadu road network from OpenStreetMap (`OSMnx` / Geofabrik extract).
   * Define spatial bounding boxes for the primary pilot area: **Coimbatore Urban & Coimbatore $\leftrightarrow$ Avinashi Corridor (NH-544)**.
2. **Graph Construction & Cleaning:**
   * Build clean, directed NetworkX graph representations ($G = (V, E)$).
   * Clean disconnected components, simplify complex intersections, and preserve true segment lengths and geometries.
3. **Attribute Standardization:**
   * Standardize edge properties: `length` (meters), `geometry` (WGS84 polylines), `highway` (hierarchy string), `name`, `oneway` (boolean), `lanes` (integer), and default `free_flow_speed` (km/h).
4. **Serialization & Validation:**
   * Save graph artifacts in standard formats (`.graphml`, `.geojson`, `.gpkg`).
   * Provide unit tests verifying node connectivity, edge traversal, and shortest-path computation between sample Tamil Nadu coordinates.
