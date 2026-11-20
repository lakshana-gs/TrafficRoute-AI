# TrafficRoute AI — P0 Data Sources Investigation

This document provides an exhaustive inventory and technical evaluation of all potential road-network, traffic-speed, trajectory, and commercial API data sources investigated for the **TrafficRoute AI** project during the **P0 (Dataset & Problem Feasibility)** milestone.

---

## 1. Road Network Data Sources

### 1.1 OpenStreetMap (OSM) — Geofabrik Tamil Nadu Regional Extract
* **Provider / Organization:** OpenStreetMap Foundation / Geofabrik GmbH
* **Source URL:** [https://download.geofabrik.de/asia/india.html](https://download.geofabrik.de/asia/india.html)
* **Geographic Coverage:** Complete State of Tamil Nadu (all 38 districts including Chennai, Coimbatore, Madurai, Tiruchirappalli, Salem, Tiruppur, Erode, etc.)
* **Temporal Currency:** Updated daily snapshot (historical archives available via OSM Full History Dump)
* **Spatial Resolution:** Sub-meter vector road geometries (WGS84 EPSG:4326)
* **File Format:** Protocolbuffer Binary Format (`.osm.pbf`), raw XML (`.osm`), or GeoJSON/Shapefile
* **Approximate Size:** ~180 MB (compressed `.osm.pbf`), expands to ~2.5 GB raw graph data
* **Licensing:** Open Database License (ODbL) 1.0 (free for commercial and academic use with proper attribution)
* **Download / Access Method:** Direct HTTP/FTP download from Geofabrik or targeted spatial query via `OSMnx` / Overpass API
* **Available Attributes:**
  * Road Geometry: Precise `LineString` coordinate tuples
  * Road Identity: Unique `osmid` (OSM Way ID), `name`, `name:ta` (Tamil), `name:en`, `ref` (e.g., NH-544, SH-18)
  * Functional Hierarchy: `highway` tag (`motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `residential`, `unclassified`, `service`, `link` variants)
  * Segment Length: Geodesic length along the polyline in meters
  * Topological Connectivity: Node IDs (`u`, `v`), intersections, roundabouts, turn restrictions
  * Speed Limits (`maxspeed`): Present on major National Highways (e.g., `80 km/h`, `100 km/h`); frequently missing or unpopulated on urban arterial and secondary streets
  * Directionality: `oneway=yes`, `oneway=-1`, dual-carriageway splits
  * Physical Attributes: `lanes`, `surface`, `bridge`, `tunnel`
* **Potential Usefulness:** **Critical / Core Foundation.** Serves as the authoritative, topologically consistent graph representation of Tamil Nadu's physical road network.
* **Major Limitations:** Does **not** contain dynamic time-series traffic speed or vehicle flow observations. Maxspeed tags are sparse on non-highway roads, necessitating a fallback heuristic speed lookup by road class.

---

## 2. Public Indian Traffic & Trajectory Datasets

### 2.1 SPT (Seerattra Pokkuvarathu Tharavu) Trajectory Dataset
* **Provider / Organization:** Indian Institute of Technology (IIT) Madras (Transportation Engineering Division, RBC-DSAI) & Technische Universität Dresden
* **Source URL / Reference:** [https://chennaitrafficdata.com](https://chennaitrafficdata.com) / *Rajput et al., Transp. Res. Part C, 2026*
* **Geographic Coverage:** 560-meter continuous arterial road section on Rajiv Gandhi Salai (OMR / IT Corridor), Chennai, Tamil Nadu
* **Time Coverage:** Selected daylight peak and off-peak observation periods
* **Temporal Resolution:** Sub-second (video frame rate trajectory extraction)
* **Spatial Resolution:** Microscopic vehicle-level coordinates ($x, y$ local and DGPS WGS84)
* **Available Features:** Unique vehicle ID, vehicle category (Two-Wheeler, Three-Wheeler/Auto-rickshaw, Car, Light Commercial Vehicle [LCV], Heavy Commercial Vehicle [HCV]), longitudinal position, lateral position, instantaneous velocity, instantaneous acceleration, headway, yaw rate
* **Speed Available:** Yes (direct empirical velocity time-series per vehicle)
* **Timestamps Available:** Yes (sub-second millisecond timestamps)
* **Road / Segment Identity:** Fixed 560m single road section
* **Coordinates:** Yes (georegistered with Ground Control Points)
* **File Format:** CSV / Plain text
* **Approximate Size:** 50 MB – 500 MB
* **Licensing:** Academic / Research use with mandatory citation
* **Potential Usefulness:** Invaluable for empirical speed distribution modeling, vehicle-class heterogeneous congestion dynamics, and micro-behavior validation in disordered Indian traffic.
* **Major Limitations:** Highly localized spatial footprint (single 560m corridor). Cannot be used directly to train a macroscopic state-wide or city-wide graph neural network across thousands of interconnected road links.

---

### 2.2 IITM-HeTra Heterogeneous Traffic Dataset
* **Provider / Organization:** IIT Madras (Robert Bosch Centre for Data Science and AI)
* **Source URL / Reference:** Kaggle / IIT Madras Research Repository
* **Geographic Coverage:** Selected signalized intersections and midblock road sections in Chennai
* **Time Coverage:** Multi-hour daylight video recordings
* **Temporal Resolution:** Video frame rate (25–30 fps)
* **Spatial Resolution:** Pixel bounding boxes from fixed CCTV surveillance camera angles
* **Available Features:** Bounding box coordinates (`[xmin, ymin, xmax, ymax]`), vehicle classification labels (2W, 3W, Car, Bus, Truck), tracking IDs
* **Speed Available:** Only indirectly through pixel-to-meter camera calibration tracking
* **Timestamps Available:** Video frame timestamps
* **Road / Segment Identity:** Camera identifier only (not connected to a global GIS road graph)
* **Coordinates:** Pixel coordinates; no embedded global GPS attributes
* **File Format:** Video files (.mp4) with JSON/XML annotations
* **Approximate Size:** Several Gigabytes
* **Licensing:** Academic research license
* **Potential Usefulness:** Excellent for training computer vision models to count vehicles and compute junction queue density from live camera feeds.
* **Major Limitations:** Not a time-series GIS road network speed dataset. Cannot directly serve as edge weights for graph routing without extensive multi-camera fusion and road-snapping infrastructure.

---

### 2.3 CHETNA-Road (City-wise Emissions & Traffic Tracking)
* **Provider / Organization:** IIT Bombay, LSCE France, Université Paris-Saclay
* **Source URL / Reference:** Figshare / Scientific Data Publications (2023–2025)
* **Geographic Coverage:** 15 major Indian metropolitan areas including Chennai
* **Time Coverage:** Multi-year daily estimates (2019–2021)
* **Temporal Resolution:** Daily
* **Spatial Resolution:** 500m $\times$ 500m spatial grid raster
* **Available Features:** Daily gridded emissions of $\text{CO}_2$, $\text{NO}_x$, $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{CO}$, VOCs, derived from underlying floating car GPS speed-emission COPERT factors
* **Speed Available:** Underlying GPS speed was used during generation, but published deliverables are gridded emission inventories
* **Timestamps Available:** Daily time index
* **Road / Segment Identity:** No (gridded spatial raster, not snapped to individual OSM road links)
* **Coordinates:** Bounding grid centroid coordinates (Latitude/Longitude)
* **File Format:** NetCDF / GeoTIFF / CSV
* **Approximate Size:** ~100 MB – 1 GB
* **Licensing:** Open Academic (Creative Commons CC-BY 4.0)
* **Potential Usefulness:** Provides macro-level spatial congestion and environmental intensity baselines across urban Chennai.
* **Major Limitations:** Daily temporal aggregation and 500m grid cell geometry make it unusable for sub-hourly or segment-level predictive routing.

---

### 2.4 Uber Movement Speeds & Travel Times (Historical Archives)
* **Provider / Organization:** Uber Technologies Inc.
* **Source URL / Status:** Platform shut down in June 2023; historical data archived in research repositories
* **Geographic Coverage:** Selected Indian metropolitan areas (Bangalore, Mumbai, New Delhi, Hyderabad; limited corridor archives in published Chennai studies)
* **Time Coverage:** Historical records spanning 2017–2020
* **Temporal Resolution:** Hourly aggregated time bins (including peak/off-peak indicators)
* **Spatial Resolution:** Snapped to OSM way IDs and Uber H3 hexagonal spatial bins
* **Available Features:** OSM Way ID, mean travel speed (km/h), standard deviation of speed, 50th/85th percentile speeds, time bin, hour of day, day of week
* **Speed Available:** Yes (mean and percentile speeds)
* **Timestamps Available:** Yes (hourly epoch timestamps)
* **Road / Segment Identity:** Yes (mapped to OpenStreetMap way IDs)
* **Coordinates:** Snapped to OSM link geometries
* **File Format:** CSV
* **Licensing:** Historical Public Data (no longer updated or supported by Uber)
* **Potential Usefulness:** Highly useful for validating diurnal speed drop curves (peak morning/evening ratios) on Indian urban corridors.
* **Major Limitations:** Service is permanently discontinued. No ongoing updates, no real-time telemetry, and no coverage for Tier-2 Tamil Nadu cities/corridors such as Coimbatore, Tiruppur, Avinashi, Salem, or Madurai.

---

### 2.5 Government Open Data Portals (data.gov.in, data.tn.gov.in, OpenCity.in)
* **Provider / Organization:** Ministry of Electronics and IT (MeitY), Government of Tamil Nadu, OpenCity Urban Data Initiative
* **Geographic Coverage:** Tamil Nadu (State-wide, District-level, Chennai Metropolitan Area)
* **Available Features:** Monthly/Annual road crash statistics, vehicle registration totals by RTO, public transit depot performance, static GTFS bus routes (partial Chennai MTC)
* **Speed Available:** **No** (no continuous sensor-level or probe-level speed feeds)
* **Temporal Resolution:** Monthly / Annual administrative aggregates
* **Licensing:** Government Open Data License - India (GODL)
* **Potential Usefulness:** Useful for demographic, safety, and high-level transport demand context.
* **Major Limitations:** Completely lacks continuous, fine-grained road-segment speed time-series data required for predictive traffic ML.

---

## 3. Standard Spatio-Temporal Benchmark Datasets (For Algorithmic Verification)

### 3.1 PEMS-BAY & METR-LA
* **Provider / Organization:** Caltrans Performance Measurement System (PeMS) / Los Angeles Metropolitan Transportation Authority
* **Source URL / Reference:** [DCRNN (Li et al., ICLR 2018)](https://github.com/liyaguang/DCRNN) / [STGCN (Yu et al., IJCAI 2018)](https://github.com/VeritasYin/STGCN_IJCAI-18)
* **Geographic Coverage:** Los Angeles County (207 sensors) / San Francisco Bay Area (325 sensors)
* **Time Coverage:** 4 to 6 months of continuous recording
* **Temporal Resolution:** Continuous 5-minute intervals
* **Spatial Resolution:** Point-based inductive loop detector sensors on freeway networks
* **Available Features:** Timestamp, Sensor ID, average traffic speed (mph), flow, occupancy, predefined graph adjacency matrix ($W \in \mathbb{R}^{N \times N}$) computed via road network shortest distance
* **Speed Available:** Yes (continuous ground-truth speed tensor $X \in \mathbb{R}^{N \times T \times F}$)
* **Timestamps Available:** Yes (continuous 5-minute timestamps)
* **File Format:** HDF5 (`.h5`), NumPy (`.npz`), CSV
* **Approximate Size:** ~50 MB – 200 MB
* **Licensing:** Public Domain / Open Academic Research
* **Potential Usefulness:** **Gold-Standard Algorithmic Benchmark.** Crucial for verifying and validating the mathematical correctness of Spatio-Temporal Graph Neural Networks (STGCN, DCRNN, Graph WaveNet) before applying them to customized Tamil Nadu road networks.
* **Major Limitations:** Data is from California highways. It cannot be passed off as Tamil Nadu traffic data.

---

## 4. Commercial & Developer Traffic APIs

### 4.1 TomTom Developer Platform (Traffic Stats & Flow API)
* **Provider / Organization:** TomTom International BV
* **Source URL:** [https://developer.tomtom.com](https://developer.tomtom.com) / [https://move.tomtom.com](https://move.tomtom.com)
* **Geographic Coverage:** Pan-India and comprehensive Tamil Nadu coverage (National Highways, State Highways, Major District Roads, Urban Arterials)
* **Services Offered:**
  * **Traffic Flow API (Real-Time):** Returns current speed, free-flow speed, and travel time on specified coordinates or bounding boxes (updated every 1–5 minutes).
  * **Traffic Stats (Historical Analytics):** Provides historical segment-level average speeds, travel times, and density percentiles over customizable date ranges (e.g., past 1–12 months) via the MOVE portal / REST API.
  * **Traffic Incidents API:** Real-time road closures, construction, and congestion events.
* **Developer Access & Pricing:**
  * Free developer tier: 2,500 non-tile requests/day (no credit card required for basic evaluation).
  * 30-day free trial on TomTom MOVE portal for Traffic Stats reports.
  * Enterprise contracts for large-scale data downloads.
* **Licensing Restrictions:** Standard SaaS developer terms prohibit bulk scraping, caching the entire state-wide network permanently, or using the data to train competing commercial foundation mapping models without specific enterprise licensing.
* **Feasibility Status:** **YELLOW** (Technically excellent and viable for integration/verification; requires managed API credits and adherence to API rate limits and terms).

---

### 4.2 HERE Technologies (HERE Traffic Analytics & Flow API)
* **Provider / Organization:** HERE Global B.V.
* **Source URL:** [https://developer.here.com](https://developer.here.com)
* **Geographic Coverage:** Comprehensive India and Tamil Nadu highway network
* **Services Offered:**
  * **HERE Traffic Analytics:** Multi-year historical traffic speed and probe count archive (available from 2021 onwards for India).
  * **Real-Time Traffic Flow API:** Provides real-time speed, jam factor (0–10 scale), and confidence scores.
  * **HERE Routing API v8:** Calculates routes with live traffic and historical speed profiles.
* **Developer Access & Pricing:**
  * Freemium tier: Up to ~250,000 free transactions/month for core location APIs.
  * Historical Traffic Analytics is an enterprise product requiring commercial agreement for raw bulk data extraction.
* **Licensing Restrictions:** Usage limited to direct API consumption; mass raw dataset extraction for offline model training is subject to commercial terms.
* **Feasibility Status:** **YELLOW**.

---

### 4.3 Mappls (MapmyIndia) Traffic & Predictive Routing API
* **Provider / Organization:** CE Info Systems Ltd (MapmyIndia)
* **Source URL:** [https://about.mappls.com/api/](https://about.mappls.com/api/)
* **Geographic Coverage:** Exhaustive Pan-India and Tamil Nadu coverage (including deep interior state roads, bypasses, flyovers, and local town streets)
* **Services Offered:**
  * `route_traffic`: Route generation factoring in live congestion and road closures.
  * `predictive_eta`: Computes route duration and speed profiles given a target future `date_time` parameter based on historical traffic patterns.
  * Driving Range Polygon API: Computes reachable isochrones accounting for traffic.
* **Developer Access & Pricing:** Developer portal with tiered plans; trial credits available for registered developers.
* **Licensing Restrictions:** Proprietary API. Raw speed matrices are not exported as open CSV files; responses are delivered as computed routes and ETAs.
* **Feasibility Status:** **YELLOW**.

---

### 4.4 Google Maps Platform (Routes API & Distance Matrix API)
* **Provider / Organization:** Google LLC
* **Source URL:** [https://developers.google.com/maps](https://developers.google.com/maps)
* **Geographic Coverage:** Complete Tamil Nadu coverage
* **Services Offered:** Route calculation with predictive traffic modeling using the `departure_time` parameter (returns `duration_in_traffic`).
* **Developer Access & Pricing:** Pay-as-you-go with $200 recurring monthly free credit.
* **Licensing Restrictions:** **Strict Terms of Service.** Section 3.2.3 explicitly forbids caching, scraping, bulk storing, or using Google Maps outputs to train machine learning models or build competing datasets.
* **Potential Usefulness:** Excellent as an independent ground-truth evaluation benchmark to verify that our system's predicted travel times are realistic and aligned with real-world conditions.
* **Feasibility Status:** **YELLOW** (Valid for live query comparison/benchmarking; strictly forbidden for training data collection).

---

## 5. Summary Matrix of Data Feasibility

| Data Category | Primary Source | Legal Status | Spatial Coverage | Dynamic Speed Available | Project Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Road Network Graph** | OpenStreetMap (Geofabrik TN Extract) | Open (ODbL 1.0) | Complete Tamil Nadu (38 Districts) | Static limits only | **GREEN** |
| **Micro-Traffic Dynamics** | SPT Chennai (IIT Madras) | Academic Open | 560m OMR Corridor, Chennai | Yes (High-res vehicle velocity) | **YELLOW** (Corridor only) |
| **Algorithm Benchmark** | PEMS-BAY / METR-LA | Public Domain | California Highway Sensor Network | Yes (5-min continuous time series) | **GREEN** (Benchmarking only) |
| **Commercial Speeds** | TomTom / HERE / Mappls APIs | Proprietary / Freemium | Comprehensive Tamil Nadu | Yes (Real-time & Historical stats) | **YELLOW** (API access required) |
| **State-wide Sensor Speeds** | State Govt / NHAI continuous feeds | Not publicly released | Non-existent as open feeds | None available open | **RED** |
