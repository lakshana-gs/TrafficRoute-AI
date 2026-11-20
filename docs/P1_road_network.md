# TrafficRoute AI — P1 Road Network Milestone Report

**Project:** TrafficRoute AI (Tamil Nadu AI-Based Predictive Traffic & Route Intelligence System)  
**Milestone:** P1 — Tamil Nadu Road Network  
**Status:** **PASS (Validated & Verified)**  
**Date:** September 2026  

---

## 1. Objective

The objective of Milestone P1 was to construct a real-world, topologically consistent, directed spatial road network for Tamil Nadu—specifically focusing on the **Coimbatore Metropolitan Area and the Coimbatore $\leftrightarrow$ Avinashi / NH-544 Corridor**—that serves as the permanent physical foundation for the TrafficRoute AI predictive routing system.

This road network is designed from first principles to answer real driver and commuter queries:
> *"Take me from Coimbatore Railway Station to Avinashi."*

Every architectural decision in P1 was evaluated against the following core requirements:
1. **Coordinate Mapping:** Arbitrary geographic locations in Tamil Nadu can be mapped and indexed in sub-millisecond time.
2. **Accurate Snapping:** Locations not directly on road centerlines are snapped to valid road nodes and edges within a documented distance threshold.
3. **Multi-Route Generation:** The engine finds topologically distinct, meaningful alternative routes (e.g., NH-544 vs. Annur/SH-80 vs. Somanur/Sulur).
4. **Map-Ready Geometry:** Every route outputs standardized GeoJSON polylines, bounding boxes, turn-by-turn steps, and road classifications for frontend map rendering.
5. **Dynamic Extensibility:** Every edge in the network is structured to accept dynamic predicted speeds ($\hat{v}(e, t)$) and travel times in future milestones (P5/P6) without modifying the routing engine.

---

## 2. Data Source & Provenance

The road network was extracted directly from **OpenStreetMap (OSM)**, the authoritative global open geospatial database.

### Network Provenance Record:
* **Dataset Name:** `Coimbatore_Avinashi_Pilot_Road_Network`
* **Source Organization:** OpenStreetMap Foundation & Global OSM Contributors
* **Source URL:** [https://www.openstreetmap.org](https://www.openstreetmap.org)
* **License:** Open Database License (ODbL) 1.0 ([https://opendatacommons.org/licenses/odbl/1-0/](https://opendatacommons.org/licenses/odbl/1-0/))
* **Extraction Timestamp:** September 2026
* **OSM Filter:** Motorable highway ways (`motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `unclassified`, `residential`, `living_street`, `service`, and all corresponding `*_link` ramps), strictly excluding non-drivable footways, cycleways, tracks, and pedestrian paths.
* **Extraction Tooling:** Custom Overpass API querying with mirror fallback architecture (`overpass-api.de`, `overpass.kumi.systems`, `overpass.nchc.org.tw`, `overpass.openstreetmap.fr`).
* **Metadata Artifact:** [`data/metadata/coimbatore_avinashi_provenance.json`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/data/metadata/coimbatore_avinashi_provenance.json)

---

## 3. Geographic Scope & Bounding Box

The pilot network encompasses the complete Coimbatore Metropolitan Area, satellite towns, and the high-density National Highway 544 (Salem–Kochi Highway) corridor connecting Coimbatore to Avinashi:

```
                          North: 11.2300° N (Annur / Avinashi Bypass)
                                         ▲
                                         │
        West: 76.9000° E ◄───────────────┼───────────────► East: 77.3000° E
   (Coimbatore West / RS Puram /         │                (Avinashi / Tiruppur Border)
       Railway Station)                  ▼
                          South: 10.9400° N (Sulur / Singanallur)
```

* **Spatial Coverage Area:** $\sim 32.2\text{ km (lat)} \times 43.6\text{ km (lon)} \approx 1,400\text{ km}^2$
* **Total Vertices (Intersections & Junctions):** **251,867 nodes**
* **Total Directed Road Edges:** **549,919 segments**

---

## 4. Graph Design & Topological Structure

The road network is modeled as a directed spatial multigraph:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$$

* **Graph Class:** `networkx.MultiDiGraph` wrapped by `RoadNetworkGraph` ([`src/network/graph.py`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/src/network/graph.py)).
* **Directionality Enforcement:**
  * One-way roads (`oneway=yes`, `oneway=1`, `highway=motorway`, `junction=roundabout`) generate a single directed edge $(u, v)$.
  * Reverse one-way roads (`oneway=-1`) generate $(v, u)$.
  * Standard two-way streets generate bidirectional paired edges $(u, v)$ and $(v, u)$.
* **Connectivity:** The graph was pruned to its largest connected component, ensuring 100% routing reachability between any pair of reachable nodes.

---

## 5. Node & Edge Schema

### Node Schema ($\mathcal{V}$):
| Attribute | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `node_id` | `int` | Authoritative OSM Node ID | `245891024` |
| `latitude` | `float` | WGS84 Latitude in degrees | `10.999023` |
| `longitude`| `float` | WGS84 Longitude in degrees | `76.963015` |
| `x` / `y` | `float` | Projected / coordinate aliases for GIS compatibility | `76.963015`, `10.999023` |

### Edge Schema ($\mathcal{E}$):
| Attribute | Type | Description | Handling when Missing |
| :--- | :--- | :--- | :--- |
| `edge_id` | `str` | Unique composite ID (`{u}_{v}_{key}`) | Deterministically generated |
| `u`, `v`, `key` | `int`, `int`, `int` | Source node, target node, multigraph index | Preserved |
| `geometry` | `LineString` | Shapely WGS84 vector polyline | Straight line $(u \to v)$ fallback |
| `length_m` | `float` | Geodesic segment length in meters | Calculated via Haversine |
| `highway_type` | `str` | Standardized road classification | Mapped to `unclassified` |
| `name` | `str` or `None` | Official road name | `None` / `null` |
| `road_reference` | `str` or `None` | Highway code (e.g., `NH-544`, `SH-80`) | `None` / `null` |
| `oneway` | `bool` | True if vehicle travel is unidirectional | `False` (Two-way default) |
| `maxspeed` | `float` or `None`| Explicit legal speed limit from OSM (km/h) | `None` (Preserved transparently) |
| `free_flow_speed_kmh` | `float` | Default baseline speed from hierarchy | Derived from engineering lookup |
| `free_flow_time_s` | `float` | Segment travel time at free-flow ($L / v$) | Computed: $L(e) / v_{\text{free}}(e)$ |
| `lanes` | `int` or `None` | Number of traffic lanes | `None` |
| `bridge` / `tunnel` | `bool` or `None`| Structural infrastructure indicators | `None` |
| `source` | `str` | Data provenance tracker | `"OpenStreetMap"` |

---

## 6. Road Classification Hierarchy & Engineering Baselines

To ensure the system never relies on fake speed limits, we explicitly separate **OSM-provided explicit speed limits** from **derived engineering baseline speeds**:

| Hierarchy Tier | Highway Tag | Description | Derived Free-Flow Speed | Share in Network |
| :---: | :--- | :--- | :---: | :---: |
| **Tier 1** | `motorway`, `motorway_link` | Controlled-access expressways | 100.0 km/h (60 km/h link) | < 0.1% |
| **Tier 2** | `trunk`, `trunk_link` | National Highways (e.g. NH-544) | 80.0 km/h (50 km/h link) | 1.06% |
| **Tier 3** | `primary`, `primary_link` | State Highways / Major arterials (e.g. Avinashi Rd) | 60.0 km/h (40 km/h link) | 2.19% |
| **Tier 4** | `secondary`, `secondary_link` | Major District Roads (MDR) / connectors | 45.0 km/h (35 km/h link) | 3.67% |
| **Tier 5** | `tertiary`, `tertiary_link` | Other District Roads / town collectors | 35.0 km/h (25 km/h link) | 11.75% |
| **Tier 6** | `unclassified` | Local interconnecting links | 30.0 km/h | 17.49% |
| **Tier 7** | `residential` | Urban and neighborhood access streets | 25.0 km/h | 63.85% |
| **Tier 8** | `service`, `living_street` | Alleyways, internal access lanes | 15.0 km/h | < 0.1% |

---

## 7. Location Snapping Architecture

Implemented in [`src/network/snapping.py`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/src/network/snapping.py):

* **Spatial Index:** High-speed `scipy.spatial.KDTree` indexing 251,867 vertices in spherical radian coordinates ($O(\log N)$ query complexity).
* **Two-Stage Snapping:**
  1. *Candidate Node Retrieval:* Rapid KD-Tree lookup finds the $k=10$ nearest graph nodes.
  2. *Perpendicular Edge Projection:* Inspects adjacent road segment `LineString` geometries, projecting the user coordinate onto the nearest physical road centerline to eliminate junction bias.
* **Maximum Distance Threshold:** Default $1,500\text{ meters}$. Coordinates exceeding this radius are flagged as `is_valid=False` with descriptive warnings.
* **Measured Latency:** **0.48 ms per query** (p95: 0.63 ms).

---

## 8. Routing Engine & Dynamic Cost Architecture

Implemented in [`src/network/routing.py`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/src/network/routing.py):

* **Core Algorithm:** Bi-directional Dijkstra / A* with an admissible great-circle heuristic.
* **Pluggable Cost Interface:**
  ```python
  def get_edge_weight(u, v, edge_data, metric=CostMetric.FREE_FLOW_TIME, departure_time=None, speed_provider=None) -> float:
      length_m = edge_data["length_m"]
      if metric == CostMetric.DISTANCE:
          return length_m
      elif metric == CostMetric.PREDICTED_TIME and speed_provider is not None:
          # Future P5/P6 Hook: queries dynamic ML speed prediction
          speed_kmh = speed_provider.get_speed(u, v, edge_data, departure_time)
      else:
          speed_kmh = edge_data["free_flow_speed_kmh"]
      
      speed_mps = max(speed_kmh * (1000.0 / 3600.0), 0.1)
      return length_m / speed_mps
  ```
* **Separation of Concerns:** Zero synthetic traffic is generated. In P1, baseline free-flow time ($L / v_{\text{free}}$) is used, perfectly structured for dynamic speed replacement in future milestones.

---

## 9. Alternative Route Strategy & Diversity Enforcement

Implemented in [`src/network/alternatives.py`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/src/network/alternatives.py):

* **Corridor Discovery Heuristic:** Employs an Iterative Edge-Penalty Graph Search. For each accepted route, a multiplicative cost penalty ($1 + \alpha$) is applied to traversed segments, forcing subsequent searches to explore distinct spatial corridors (e.g. NH-544 direct vs Annur SH-80 bypass vs Somanur southern route).
* **Strict Route Diversity Criterion:**
  $$\text{Overlap}(R_A, R_B) = \frac{\sum_{e \in R_A \cap R_B} L(e)}{\min(L(R_A), L(R_B))} \le 0.80$$
  Candidate routes sharing $> 80\%$ of their physical distance with an existing option are rejected, preventing minor neighborhood lane detours from masquerading as meaningful alternative routes.

---

## 10. Map-Ready Route Geometry & GeoJSON

Implemented in [`src/network/geometry.py`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/src/network/geometry.py):

Every computed route is converted into a standardized GeoJSON `FeatureCollection` containing:
1. **Route LineString Feature:** High-resolution polyline coordinates (`[[lon1, lat1], [lon2, lat2], ...]`) preserving true curve geometry.
2. **Origin & Destination Point Features:** Marking exact input coordinates, snapped road coordinates, and snap distance.
3. **Turn-by-Turn Leg Steps:** Grouped by consecutive road name transitions with step distance (km) and estimated duration (mins).
4. **Bounding Box:** `[min_lon, min_lat, max_lon, max_lat]` for immediate frontend map camera centering.

---

## 11. Validation Results Summary

Detailed in [`docs/P1_validation_report.md`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/docs/P1_validation_report.md):

* **Topological Consistency:** 100% of nodes reside in a single connected component.
* **Geometric Accuracy:** Zero empty geometries, zero zero-length edges, all coordinates strictly bounded in Tamil Nadu.
* **Edge Directionality:** 11,951 one-way segments (2.2%) correctly enforcing highway ramps and divided carriageways.
* **Segment Lengths:** Min $0.06\text{ m}$, Median $34.45\text{ m}$, Max $1,021.61\text{ m}$, Mean $44.66\text{ m}$.

---

## 12. Real Integration Test: Coimbatore $\to$ Avinashi

Tested on real coordinates without hardcoding ([`tests/test_integration.py`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/tests/test_integration.py)):

* **Origin:** Coimbatore Railway Station ($10.9990^\circ\text{N}, 76.9630^\circ\text{E}$) $\to$ Snapped to Station Approach Rd ($43.9\text{ m}$).
* **Destination:** Avinashi Central ($11.1930^\circ\text{N}, 77.2690^\circ\text{E}$) $\to$ Snapped to Avinashi Main Rd ($10.3\text{ m}$).
* **Generated Routes:**
  1. **Route 1 (Primary — Via NH-544 / Avinashi Road):**
     * Total Distance: **43.67 km**
     * Free-Flow Duration: **36.5 minutes**
     * Road Segments: 657 edges
  2. **Route 2 (Alternative — Via Annur / SH-80 Corridor):**
     * Total Distance: **48.12 km**
     * Free-Flow Duration: **44.2 minutes**
     * Corridor Overlap with Route 1: $< 35\%$
* **Verdict:** **PASSED (100% Real-Pipeline Execution)**.

---

## 13. Performance Benchmarks

Measured on a standard development machine across 251,867 nodes and 549,919 edges:

| Operation | Measured Performance | Target SLA | Status |
| :--- | :--- | :--- | :--- |
| **Optimized Binary Graph Loading** | **6.05 seconds** | < 15.0 s | **PASSED** |
| **Spatial KD-Tree Index Construction** | **1.13 seconds** | < 3.0 s | **PASSED** |
| **Memory Footprint in RAM** | **1,242 MB** | < 2,000 MB | **PASSED** |
| **Location Snapping Latency** | **0.48 ms** | < 5.0 ms | **PASSED** |
| **Single Route Calculation Latency** | **849.75 ms** | < 2,000 ms | **PASSED** |
| **3-Alternative Route Generation Latency**| **2,961 ms** | < 5,000 ms | **PASSED** |
| **Map-Ready GeoJSON Serialization** | **7.23 ms** | < 50.0 ms | **PASSED** |

---

## 14. Artifact Inventory

```
TrafficRoute-AI/
├── data/
│   ├── raw/
│   │   └── coimbatore_avinashi_osm.json          # 29 MB raw Overpass OSM vector JSON
│   ├── processed/
│   │   ├── coimbatore_avinashi_network.bin       # Optimized binary road graph cache (6s load)
│   │   ├── coimbatore_avinashi_network.graphml   # 459 MB standard GraphML open graph
│   │   └── coimbatore_avinashi_edges.geojson     # 221 MB GeoJSON edge layer
│   └── metadata/
│       └── coimbatore_avinashi_provenance.json   # Structured provenance metadata
├── src/
│   └── network/
│       ├── __init__.py
│       ├── graph.py                              # RoadNetworkGraph & schema engine
│       ├── provenance.py                         # Provenance data model
│       ├── extraction.py                         # Overpass query & parsing pipeline
│       ├── snapping.py                           # KD-Tree spatial indexer & edge snapper
│       ├── routing.py                            # Extensible Dijkstra/A* routing engine
│       ├── alternatives.py                       # Alternative route generator & diversity filter
│       ├── geometry.py                           # Map-ready GeoJSON builder
│       └── validation.py                         # Comprehensive graph validator
├── tests/
│   ├── test_graph.py                             # Graph schema & serialization tests
│   ├── test_snapping.py                          # Spatial snapping & distance tests
│   ├── test_routing.py                           # Shortest distance & fastest time tests
│   ├── test_alternatives.py                      # Alternative routes & overlap metric tests
│   ├── test_geometry.py                          # GeoJSON & bounding box tests
│   ├── test_validation.py                        # Validator suite tests
│   └── test_integration.py                       # Coimbatore -> Avinashi full pipeline test
└── docs/
    ├── P0_dataset_feasibility.md
    ├── P0_data_sources.md
    ├── P0_architecture_notes.md
    ├── P1_road_network.md                        # This Milestone Report
    └── P1_validation_report.md                   # Full Topological Validation Report
```

---

## 15. Limitations & Expansion Strategy

1. **Memory Scaling for State-Wide Expansion:**
   * In-memory storage for Coimbatore–Avinashi (550k edges) consumes $\sim 1.2\text{ GB}$ RAM.
   * Expanding to the entirety of Tamil Nadu ($\sim 5\text{M}$ edges) would require $\sim 10\text{–}12\text{ GB}$ RAM if loaded in a single monolithic NetworkX object.
   * **Scalable Solution for P6/P8:** Partition the state into regional metropolitan subgraphs (Coimbatore, Chennai, Madurai, Salem, Trichy) interconnected via a pruned State Highway/National Highway backbone graph, or integrate high-performance spatial engines (e.g. Rust/C++ OSRM or Valhalla tile caches) as the geographic footprint expands.

---

## 16. Requirements for Milestone P2 (Traffic Data Pipeline)

With the physical road network established, validated, and tested, the project is ready for **Milestone P2 (Traffic Data Pipeline)**:
1. **Traffic Speed Ingestion Interface:** Build data loaders to map speed time-series observations to edge identifiers (`edge_id`, `(u, v, key)`).
2. **Empirical Baseline Construction:** Parameterize diurnal time-of-day / day-of-week speed reduction functions for each road classification based on Indian traffic studies.
3. **Corridor Calibration Pipeline:** Calibrate speed curves for the Coimbatore–Avinashi (NH-544) corridor against empirical research distributions (IIT Madras SPT dynamics and standard peak-hour congestion factors).
4. **Historical Profile Storage:** Store aggregated speed tensors indexed by `(edge_id, hour_of_day, day_of_week)` for sub-millisecond retrieval by the routing engine.

---

## 17. Milestone Decision

```
============================================================
P1 STATUS: PASS
============================================================
All 14 success criteria verified:
[x] Real OSM road network obtained (Overpass OSM API)
[x] Network is reproducible (provenance & raw cache)
[x] Directed graph constructed (251,867 nodes, 549,919 edges)
[x] Road attributes preserved (highway, name, ref, lanes, oneway, maxspeed)
[x] Graph validated (0 zero-length edges, 0 invalid geometries)
[x] Arbitrary coordinates snapped (0.48 ms KD-Tree indexer)
[x] Origin -> Destination routing works (Dijkstra / A*)
[x] Multiple meaningful routes generated (with diversity enforcement)
[x] Route geometry is map-ready (GeoJSON FeatureCollection with turn steps)
[x] Coimbatore -> Avinashi tested through real pipeline (43.67 km, 36.5 mins)
[x] Automated tests pass (13/13 pytest passed in 13.3s)
[x] Provenance is documented (JSON metadata with ODbL license)
[x] Performance is measured (6.05s load, 0.48ms snap, 849ms routing)
[x] Architecture ready for future predicted-speed integration
============================================================
```
