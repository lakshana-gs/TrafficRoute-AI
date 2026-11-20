# TrafficRoute AI — P1 Road Network Validation Report

**Network Dataset:** Coimbatore Metropolitan Area & Coimbatore $\leftrightarrow$ Avinashi (NH-544) Corridor  
**Total Vertices (Nodes):** 251,867  
**Total Directed Edges:** 549,919  
**Overall Validation Status:** **PASSED (GREEN)**

---

## 1. Graph Summary & Connectivity

| Metric | Measured Value | Requirement / Status |
| :--- | :--- | :--- |
| **Total Graph Nodes** | 251,867 | > 5,000 (PASSED) |
| **Total Directed Edges** | 549,919 | > 10,000 (PASSED) |
| **Weakly Connected Components (WCC)** | 1 | Main component dominance: 100.0% |
| **Strongly Connected Components (SCC)** | 1 | Largest SCC nodes: 251,867 |
| **One-Way Directed Edges** | 11,951 (2.2%) | Dual carriageway & ramp compliance |
| **Two-Way Directed Edges** | 537,968 | Bidirectional paired edges |

---

## 2. Geometric Integrity & Coordinate Bounds

| Metric | Measured Value | Standard / Result |
| :--- | :--- | :--- |
| **Coordinate Bounds (WGS84)** | Lat: [10.8575, 11.2977], Lon: [76.8399, 77.3284] | Exact Tamil Nadu pilot bbox |
| **Invalid Coordinate Nodes** | 0 | 0 (PASSED) |
| **Invalid / Empty Geometries** | 0 | 0 (PASSED) |
| **Zero-Length Road Edges** | 0 | 0 (PASSED) |
| **Min Segment Length** | 0.06 m | Positive non-zero |
| **Median Segment Length** | 34.45 m | Standard urban segment length |
| **Max Segment Length** | 1021.61 m | Inter-city expressway section |
| **Mean Segment Length** | 44.66 m | Consistent road graph density |

---

## 3. Road Classification Hierarchy Breakdown

| Highway Classification | Edge Count | Percentage | Default Free-Flow Speed (km/h) |
| :--- | :--- | :--- | :--- |
| `residential` | 351,100 | 63.85% | Derived from Tamil Nadu Highway Standards |
| `unclassified` | 96,162 | 17.49% | Derived from Tamil Nadu Highway Standards |
| `tertiary` | 64,176 | 11.67% | Derived from Tamil Nadu Highway Standards |
| `secondary` | 20,077 | 3.65% | Derived from Tamil Nadu Highway Standards |
| `primary` | 11,385 | 2.07% | Derived from Tamil Nadu Highway Standards |
| `trunk` | 5,471 | 0.99% | Derived from Tamil Nadu Highway Standards |
| `primary_link` | 643 | 0.12% | Derived from Tamil Nadu Highway Standards |
| `tertiary_link` | 444 | 0.08% | Derived from Tamil Nadu Highway Standards |
| `trunk_link` | 359 | 0.07% | Derived from Tamil Nadu Highway Standards |
| `secondary_link` | 102 | 0.02% | Derived from Tamil Nadu Highway Standards |

---

## 4. Attribute Completeness Analysis

| Attribute | Present Count | Percentage | Policy / Handling Strategy |
| :--- | :--- | :--- | :--- |
| **Road Names (`name`)** | 51,627 | 9.39% | Preserved from OSM; `null` on minor unnamed links |
| **Road References (`ref`)** | 32,050 | 5.83% | Preserved on National/State Highways (e.g. NH-544) |
| **OSM Explicit Speed Limits (`maxspeed`)** | 9,234 | 1.68% | Fallback to Road Hierarchy Speed Lookup (Transparent) |
| **Lanes Information (`lanes`)** | 17,255 | 3.14% | Preserved where available; `null` otherwise |

---

## 5. End-to-End Routing Connectivity Smoke Tests

### Test Case: Coimbatore Railway Station -> Avinashi (Core NH-544)
* **Status:** **PASSED (GREEN)**
* **Total Distance:** `43.67 km`
* **Free-Flow Duration:** `36.5 mins`
* **Traversed Road Segments:** `657 edges`
* **Origin Snap Distance:** `43.92 m`
* **Destination Snap Distance:** `10.32 m`

### Test Case: Gandhipuram Bus Stand -> Coimbatore Airport (KMCH / Avinashi Rd)
* **Status:** **PASSED (GREEN)**
* **Total Distance:** `10.46 km`
* **Free-Flow Duration:** `13.3 mins`
* **Traversed Road Segments:** `232 edges`
* **Origin Snap Distance:** `18.39 m`
* **Destination Snap Distance:** `401.27 m`

### Test Case: RS Puram -> Annur (Northern SH Corridor)
* **Status:** **PASSED (GREEN)**
* **Total Distance:** `38.83 km`
* **Free-Flow Duration:** `36.9 mins`
* **Traversed Road Segments:** `846 edges`
* **Origin Snap Distance:** `0.36 m`
* **Destination Snap Distance:** `165.44 m`

---

## 6. Validation Verdict
The road network graph fulfills all topological, geometric, routing, and attribute integrity criteria required for Milestone P1.
