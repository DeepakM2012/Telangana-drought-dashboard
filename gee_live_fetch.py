"""
gee_live_fetch.py

Live Google Earth Engine data fetch for the Telangana Drought Dashboard's
"Live Prediction" tab. Fetches the most recent available month's Rainfall,
Temperature, Soil Moisture and NDVI for a single district, so the user
doesn't have to type them in manually.

Requires:
    pip install earthengine-api
One-time auth (already done if gee_data_pipeline.py worked for you):
    python -c "import ee; ee.Authenticate()"

Usage from the dashboard:
    import gee_live_fetch
    result = gee_live_fetch.fetch_live_indicators(
        geojson_path=DISTRICT_GEOJSON_PATH,
        district_name="Karimnagar",
    )
    # result = {"Rainfall": .., "Temperature": .., "Soil_Moisture": ..,
    #           "NDVI": .., "Year": .., "Month": .., "Period": "2026-06"}
"""

import json
import re
import datetime
import ee

# Same Cloud project ID used in gee_data_pipeline.py
GEE_PROJECT = "project-8928310d-34bc-4bd5-aef"

# Dataset IDs — NOT constructed as ee.ImageCollection here, because that would run
# at import time, before init_gee() has had a chance to call ee.Initialize(). They're
# built inside the functions below, after init_gee() has already run.
CHIRPS_ID     = "UCSB-CHC/CHIRPS/V3/DAILY_RNL"
ERA5_ID       = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
MODIS_NDVI_ID = "MODIS/061/MOD13Q1"

_ee_initialized = False


def init_gee():
    """Initialize Earth Engine once per process. Safe to call repeatedly."""
    global _ee_initialized
    if _ee_initialized:
        return
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    _ee_initialized = True


def _normalize_name(s):
    """Uppercase and strip everything except letters, so 'Karimnagar', 'KARIMNAGAR',
    and 'karim nagar' all match the same way."""
    return re.sub(r"[^A-Z]", "", s.upper())


def _district_geometry(geojson_path, district_name, name_property="district"):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    target_norm = _normalize_name(district_name)
    available = []
    for feat in gj["features"]:
        raw_name = feat["properties"].get(name_property)
        if raw_name is None:
            continue
        available.append(raw_name)
        if _normalize_name(raw_name) == target_norm:
            return ee.Geometry(feat["geometry"])

    raise ValueError(
        f"District '{district_name}' not found in GeoJSON using property '{name_property}'. "
        f"Names actually present in the file: {sorted(set(available))}"
    )


def _month_bounds(y, m):
    start = ee.Date.fromYMD(y, m, 1)
    end = start.advance(1, "month")
    return start, end


def _reduce_one(image, geometry, scale):
    val = image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, maxPixels=1e9
    )
    result = val.getInfo()
    return next(iter(result.values())) if result else None


def fetch_live_indicators(geojson_path, district_name, name_property="district",
                           max_months_back=6, scale=5000):
    """
    Walks backward from the current month (up to max_months_back months) to find
    the most recent month where CHIRPS, ERA5-Land, and MODIS NDVI all have data —
    satellite products typically lag 1-3 months behind real time (ERA5-Land's
    monthly aggregate product in particular can lag more than CHIRPS/MODIS).

    Returns a dict:
        {"Rainfall": float, "Temperature": float, "Soil_Moisture": float,
         "NDVI": float, "Year": int, "Month": int, "Period": "YYYY-MM"}

    Raises RuntimeError if no complete month is found within max_months_back.
    """
    init_gee()
    geometry = _district_geometry(geojson_path, district_name, name_property)

    today = datetime.date.today()
    y, m = today.year, today.month

    for _ in range(max_months_back + 1):
        start, end = _month_bounds(y, m)

        try:
            rain_img = ee.ImageCollection(CHIRPS_ID).filterDate(start, end).select("precipitation").sum().rename("Rainfall")
            temp_img = (ee.ImageCollection(ERA5_ID).filterDate(start, end).select("temperature_2m").mean()
                            .subtract(273.15).rename("Temperature"))
            sm_img   = (ee.ImageCollection(ERA5_ID).filterDate(start, end).select("volumetric_soil_water_layer_1")
                            .mean().rename("Soil_Moisture"))
            ndvi_img = (ee.ImageCollection(MODIS_NDVI_ID).filterDate(start, end).select("NDVI").mean()
                            .multiply(0.0001).rename("NDVI"))

            rain_val = _reduce_one(rain_img, geometry, scale)
            temp_val = _reduce_one(temp_img, geometry, scale)
            sm_val   = _reduce_one(sm_img, geometry, scale)
            ndvi_val = _reduce_one(ndvi_img, geometry, scale)
        except Exception:
            # One or more datasets have no published image for this month yet
            # (e.g. an empty ImageCollection triggers a server-side computation
            # error rather than just returning null). Treat it the same as
            # missing data and try the previous month.
            rain_val = temp_val = sm_val = ndvi_val = None

        if None not in (rain_val, temp_val, sm_val, ndvi_val):
            return {
                "Rainfall": round(rain_val, 2),
                "Temperature": round(temp_val, 2),
                "Soil_Moisture": round(sm_val, 4),
                "NDVI": round(ndvi_val, 4),
                "Year": y,
                "Month": m,
                "Period": f"{y}-{m:02d}",
            }

        # step back one month and try again
        m -= 1
        if m == 0:
            m = 12
            y -= 1

    raise RuntimeError(
        f"No complete satellite data found for {district_name} in the last "
        f"{max_months_back} months. This can happen right after month-end before "
        f"CHIRPS/MODIS products are published — try again in a few days, or increase "
        f"max_months_back."
    )