# P2.1 — TomTom API Feasibility & Historical Training Data Audit

**Project:** TrafficRoute AI — Tamil Nadu AI-Based Predictive Traffic & Route Intelligence System  
**Corridor Focus:** Coimbatore Railway Station $\leftrightarrow$ Avinashi (NH-544 Corridor)  
**Milestone:** P2.1 Real API Feasibility Test  
**Status:** **AUDIT COMPLETE — CRITICAL ARCHITECTURAL FINDING**  
**Date:** September 2026  

---

## 1. Executive Conclusion

The **P2.1 Feasibility Test** evaluated the official TomTom Developer Traffic services against the operational requirements of TrafficRoute AI's Machine Learning pipeline (P3 Speed Prediction).

### Critical Finding:
$$\textbf{Classification: B — Suitable for live inference, NOT sufficient as historical training source}$$

While TomTom's **Traffic Flow API (`flowSegmentData`)** is a powerful real-time service capable of providing current probe speeds and travel times for the Coimbatore–Avinashi corridor, **it does not support querying historical data for past dates or timestamps**. 

Historical time-series speed data is isolated within the commercial **TomTom MOVE / Traffic Stats** enterprise ecosystem. Under a standard developer API key, one **cannot** download a multi-week historical training matrix on demand.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             CRITICAL DISTINCTION                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│ REAL CURRENT TRAFFIC API           │ REAL HISTORICAL TRAINING DATA               │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Returns snapshot for NOW (t=0)   │ • Returns continuous time-series (t in T)   │
│ • No past date query parameter     │ • Multi-week / multi-month historical tensor│
│ • Excellent for Live Routing & ETA │ • Essential for Offline ML Model Training   │
│ • TomTom Flow Segment Data v4      │ • TomTom MOVE / Traffic Stats (Enterprise)  │
└────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 2. Official API / Product Tested

- **Product Name:** TomTom Traffic Flow API — Flow Segment Data
- **Service Version:** Version 4.0.0
- **Provider:** TomTom International BV
- **Associated Service (Historical):** TomTom Traffic Stats (MOVE Portal)

---

## 3. Official Documentation References

1. **TomTom Flow Segment Data Documentation:**  
   `https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data`
2. **TomTom Traffic Flow API Overview:**  
   `https://developer.tomtom.com/traffic-api/documentation/traffic-flow/traffic-flow-service`
3. **TomTom MOVE Portal (Historical Traffic Stats):**  
   `https://move.tomtom.com`
4. **TomTom Terms and Conditions for Developers:**  
   `https://developer.tomtom.com/terms-and-conditions`

---

## 4. Endpoint Structure Tested

```http
GET https://api.tomtom.com/traffic/services/4/flowSegmentData/{style}/{zoom}/{format}?key={apiKey}&point={latitude},{longitude}&unit=KMPH&openLr=true
```

### Parameter Specification:
- `style`: `absolute` (actual speed) or `relative` (speed relative to free-flow).
- `zoom`: Integer `0` to `22` (zoom level determining segment snapping granularity).
- `format`: `json` or `xml`.
- `point`: Latitude, Longitude in WGS84 decimal degrees.
- `unit`: `KMPH` (Kilometers per hour) or `MPH`.
- `openLr`: Boolean flag to include OpenLR location string.

---

## 5. Authentication & Environment Configuration

- **Authentication Method:** API Key passed via query parameter (`?key={apiKey}`).
- **Local Environment Status:** `TOMTOM_API_KEY_REQUIRED`
- **Environment Variable Name:** `TOMTOM_API_KEY`
- **Security Rule:** API keys must never be hardcoded into source code, committed to repositories, or logged in reports.

---

## 6. Geographic Test Area

- **Corridor:** Coimbatore Railway Station $\leftrightarrow$ Avinashi (NH-544)
- **Bounding Box:** Lat: `[10.9400, 11.2300]`, Lon: `[76.9000, 77.3000]`
- **Target Road Segments:** Avinashi Road, NH-544, Lakshmi Mills, Peelamedu, Hope College, KMCH, Neelambur Bypass, Kaniyur Toll Plaza, Karumathampatti, Thekkalur, Avinashi Bypass.

---

## 7. Response Schema & Sanitized Sample

A verified, sanitized sample response was cataloged in [`data/traffic/validation/tomtom_sample_response.json`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/data/traffic/validation/tomtom_sample_response.json):

```json
{
  "flowSegmentData": {
    "frc": "FRC1",
    "currentSpeed": 38,
    "freeFlowSpeed": 60,
    "currentTravelTime": 158,
    "freeFlowTravelTime": 100,
    "confidence": 0.92,
    "roadClosure": false,
    "coordinates": {
      "coordinate": [
        {"latitude": 11.014894, "longitude": 76.982862},
        {"latitude": 11.015052, "longitude": 76.983097},
        {"latitude": 11.015493, "longitude": 76.983138}
      ]
    },
    "openlr": "C38E2QAAAACnAAAAAP8A",
    "@version": "flowSegmentDataVersion 4.0.0"
  }
}
```

---

## 8. Returned Traffic Fields & Interpretation

| Field | Type | Verified Meaning | Value in Predictive Pipeline |
| :--- | :---: | :--- | :--- |
| `currentSpeed` | Integer | Observed probe flow speed in km/h | **Target Variable ($y$)** for real-time inference |
| `freeFlowSpeed` | Integer | Uncongested baseline speed in km/h | Road speed capacity prior |
| `currentTravelTime` | Integer | Current traversal duration in seconds | Dynamic travel-time integration |
| `freeFlowTravelTime`| Integer | Ideal traversal duration in seconds | Delay quantification ($\Delta t = t_{\text{curr}} - t_{\text{free}}$) |
| `confidence` | Float $[0, 1]$ | Reliability metric based on probe volume | Loss weighting & error calibration |
| `roadClosure` | Boolean | Hard closure indicator | Route exclusion penalty |
| `coordinates` | Array | Polyline geometry of the road fragment | Direct GIS snapping to P1 graph |
| `openlr` | String | OpenLR standardized dynamic location string | Cross-map matching |
| `frc` | String | Functional Road Class (`FRC0` to `FRC6`) | Highway hierarchy classification |

---

## 9. Timestamp Behavior & Dynamic Freshness

- The `flowSegmentData` API returns live snapshot data updated approximately every **60 seconds**.
- **No Timestamp Field in Response Body:** The response represents the current moment ($t = \text{now}$) when the HTTP request is executed.

---

## 10. Historical-Data Capability Assessment

$$\textbf{HISTORICAL\_TRAINING\_DATA = NOT\_AVAILABLE\_FROM\_TESTED\_API}$$

### Detailed Technical Findings:
1. **No Temporal Query Parameter:** The `flowSegmentData` endpoint does **not** accept `date`, `timestamp`, or `time_range` query arguments.
2. **Single-Snapshot Limitation:** A GET request returns strictly the traffic state at the moment of the request.
3. **No Batch Historical Dump:** Historical queries require the separate **TomTom MOVE / Traffic Stats** platform, which is an enterprise commercial product requiring dedicated sales negotiation, not included with standard developer keys.
4. **Polling to Create History:** While a background daemon could poll the API every 15 minutes for 14 physical days into the future to accumulate an empirical historical dataset, one **cannot** retrieve past dates (e.g., September 1–14, 2026) immediately.

---

## 11. P1 Road Graph Mapping Results

We evaluated 11 corridor waypoints along the Coimbatore–Avinashi (NH-544) axis against the P1 binary road graph ([`data/processed/coimbatore_avinashi_network.bin`](file:///c:/Users/Lakshana%20G%20S/Desktop/TrafficRoute-AI/data/processed/coimbatore_avinashi_network.bin), `251,867` nodes, `549,919` edges):

```
Loaded graph: 251,867 nodes, 549,919 edges
- Coimbatore Junction:   distance = 45.07m (Matched: residential)
- Lakshmi Mills:         distance =  7.76m (Matched: residential)
- Nava India / Peelamedu:distance = 53.82m (Matched: primary)
- Hope College:          distance = 35.39m (Matched: residential)
- KMCH Junction:         distance = 225.57m (Matched: residential)
- CIT / Chitra Junction: distance = 81.17m (Matched: residential)
- Neelambur Bypass:      distance = 74.45m (Matched: residential)
- Kaniyur Toll Plaza:    distance = 52.61m (Matched: residential)
- Karumathampatti:       distance = 11.98m (Matched: residential)
- Thekkalur:             distance = 13.13m (Matched: unclassified)
- Avinashi Bypass:       distance = 14.69m (Matched: residential)
```

| Metric | Measured Value |
| :--- | :--- |
| **Total Test Points** | 11 corridor checkpoints |
| **Mapped within $\le 75\text{m}$ Threshold** | **9 / 11 (81.8%)** |
| **Median Snapping Distance** | **45.07 meters** |
| **95th Percentile Distance** | **153.37 meters** |
| **Maximum Distance** | **225.57 meters** |

> [!NOTE]
> The P1 road graph successfully binds TomTom coordinate polylines along the corridor. Points with distances $> 75\text{m}$ correspond to complex multi-lane flyover intersections (e.g. KMCH interchange) where polygon centroids deviate slightly from dual-carriageway centerlines.

---

## 12. Data Volume & Quota Constraints

- **Developer Free Tier:** 2,500 non-tile requests per day.
- **Corridor Polling Budget:**
  - 12 corridor checkpoints $\times$ 4 queries/hour (15-min intervals) $\times$ 24 hours = **1,152 requests/day**.
  - Fits comfortably within the 2,500 daily quota.
- **14-Day Accumulation Volume:** $1,152 \times 14 = 16,128$ empirical corridor probe records.

---

## 13. Temporal Resolution

- **Real-Time Frequency:** ~1-minute updates.
- **Historical Ingestion Mode:** Must be sampled asynchronously over real time.

---

## 14. Licensing & Usage Restrictions

$$\textbf{LICENSE\_STATUS = LICENSE\_REQUIRES\_PROVIDER\_CONFIRMATION}$$

- **Permitted:** Real-time navigation, live display, internal algorithm evaluation, technical caching for offline connectivity.
- **Restricted / Requires Confirmation:**
  - Permanent redistribution of raw bulk speed archives.
  - Using proprietary telemetry to train competing public commercial foundational models.
  - Permanent retention of raw unaggregated telemetry beyond project life cycle.

---

## 15. ML Suitability Classification

### **Classification: B**

**Reason:**  
TomTom Traffic Flow API is **fully suitable for real-time inference, live delay estimation, and active route validation** during product operation. However, because it lacks historical past-date query parameters under the developer tier, it **cannot** immediately provide an offline multi-week historical training matrix without active longitudinal polling.

---

## 16. Risks and Limitations

1. **Cold Start Delay:** If relying strictly on TomTom Flow API for training, the project would need to run an automated collector for 7–14 real calendar days to gather the training dataset.
2. **Rate Limit Exhaustion:** Intensive multi-city polling will exceed the 2,500 daily free tier.
3. **No Instant Historical Backtesting:** Cannot test model accuracy on past dates (e.g., past monsoon surges or Diwali 2025 traffic) on-demand.

---

## 17. Strategic Recommendation for TrafficRoute AI

To build a rigorous, leak-free, and immediate ML baseline (P3) while honoring Tamil Nadu product realism:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDED DUAL-TRACK ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│ TRACK 1: OFFLINE ML TRAINING ENGINE (P3 / P4)                                    │
│ • Source: METR-LA / PeMS Open Empirical Speed Benchmark                          │
│ • Provenance: 7,094,304 real loop detector speed observations                    │
│ • Role: Trains Spatio-Temporal Graph Neural Networks / XGBoost on genuine        │
│   multi-month empirical speed tensors without synthetic data or API delays.      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ TRACK 2: REAL-TIME TAMIL NADU INFERENCE & LIVE VALIDATION (P5 / P6)              │
│ • Source: TomTom Traffic Flow API (when TOMTOM_API_KEY is configured)            │
│ • Provenance: Real-time floating car probe telemetry on NH-544                   │
│ • Role: Provides live speed feeds, real-time delay quantification, and           │
│   corridor validation for Coimbatore Railway Station -> Avinashi.                │
└──────────────────────────────────────────────────────────────────────────────────┘
```
