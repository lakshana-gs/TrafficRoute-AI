import json
import time
import requests
from pathlib import Path

# Targeted BBox for Coimbatore Urban + NH-544 Avinashi Corridor
# South: 10.94, North: 11.23, West: 76.90, East: 77.30
query = """[out:json][timeout:90];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"](10.94,76.90,11.23,77.30);
);
out body;
>;
out skel qt;
"""

headers = {
    "User-Agent": "TrafficRouteAI-PredictiveSystem/1.0 (TamilNadu Route Research; contact@trafficroute.ai)",
    "Accept": "application/json",
}

endpoints = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

Path("data/raw").mkdir(parents=True, exist_ok=True)
output_file = Path("data/raw/coimbatore_avinashi_osm.json")

success = False
for ep in endpoints:
    try:
        print(f"Querying Overpass endpoint: {ep} ...")
        t0 = time.time()
        resp = requests.post(ep, data={"data": query}, headers=headers, timeout=90)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            elements = data.get("elements", [])
            ways = [e for e in elements if e.get("type") == "way"]
            nodes = [e for e in elements if e.get("type") == "node"]
            print(f"SUCCESS from {ep} in {elapsed:.2f}s! Total elements: {len(elements)} (Ways: {len(ways)}, Nodes: {len(nodes)})")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"Saved raw OSM data to {output_file}")
            success = True
            break
        else:
            print(f"HTTP {resp.status_code} from {ep}: {resp.text[:200]}")
    except Exception as exc:
        print(f"Failed connecting to {ep}: {exc}")

if not success:
    print("Could not fetch from any Overpass endpoint.")
