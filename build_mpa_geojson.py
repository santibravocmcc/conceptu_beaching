# =============================================================================
# BUILD COMPACT MPA GEOJSON
# =============================================================================
# One-time preprocessing: clip the 97 MB WDPA marine geopackage to the study
# basin, simplify geometry, keep only the fields the dashboard needs, round
# coordinates, and write a small GeoJSON the app can embed in the maps.
# Re-run only if the source geopackage changes.
# =============================================================================

import json
import geopandas as gpd

SRC = "protected_area/WDPA_WDOECM_Jun2026_marine.gpkg"
LAYER = "wdpa_wdoecm_poly_jun2026_marine"
OUT = "protected_area/mpa_mediterranean.geojson"

BBOX = (-6.0, 30.0, 39.0, 46.0)   # lon_min, lat_min, lon_max, lat_max
SIMPLIFY_TOL = 0.008              # ~0.9 km, fine at basin scale
COORD_DECIMALS = 4                # ~11 m precision


def clean(value, default="Not Reported"):
    if value is None:
        return default
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "not reported"}:
        return default
    return s


def round_coords(obj, nd):
    if isinstance(obj, (int, float)):
        return round(obj, nd)
    if isinstance(obj, list):
        return [round_coords(x, nd) for x in obj]
    return obj


def main():
    gdf = gpd.read_file(SRC, layer=LAYER, bbox=BBOX)
    print(f"clipped features: {len(gdf)}")

    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    features = []
    for _, row in gdf.iterrows():
        site_id = clean(row.get("SITE_ID"), "")
        site_pid = clean(row.get("SITE_PID"), "")
        url = ""
        if site_id:
            url = f"https://www.protectedplanet.net/{site_id}"
            if site_pid:
                url += f"?site_pid={site_pid}"

        geom = json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"]
        geom["coordinates"] = round_coords(geom["coordinates"], COORD_DECIMALS)

        features.append({
            "type": "Feature",
            "properties": {
                "name": clean(row.get("NAME")),
                "desig": clean(row.get("DESIG_ENG")),
                "desig_type": clean(row.get("DESIG_TYPE")),
                "iucn": clean(row.get("IUCN_CAT")),
                "url": url,
            },
            "geometry": geom,
        })

    fc = {"type": "FeatureCollection", "features": features}
    with open(OUT, "w") as f:
        json.dump(fc, f, separators=(",", ":"))

    import os
    print(f"wrote {OUT}: {len(features)} features, {os.path.getsize(OUT) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
