# ==========================================================
# GEE DISTRICT CLIMATE DATA BACKFILL PIPELINE
#
# Extracts monthly Rainfall, Temperature, Soil Moisture, and
# NDVI for every Telangana district from Google Earth Engine,
# producing a CSV in the same shape as the input to
# otherfeatures.py (District, Year, Month, Rainfall,
# Temperature, Soil_Moisture, NDVI).
#
# Run once to backfill/refresh your climate master dataset,
# then feed the output through otherfeatures.py (SPI-3, lag
# features, Groundwater_Proxy) to produce a new
# Telangana_Model_Input.csv for train_drought_model.py.
# ==========================================================

import ee
import json
import time
from datetime import date

import pandas as pd

# ==========================================================
# CONFIG — EDIT THESE
# ==========================================================

GEE_PROJECT = "project-8928310d-34bc-4bd5-aef"   # <-- your registered Earth Engine Cloud project ID

DISTRICT_GEOJSON_PATH = r"D:\Drought_Temp\telangana_districts.geojson"

# Property key in the GeoJSON that holds the district name.
# Check your file's properties first — the dashboard script tries several
# candidates (District, district, DISTRICT, dtname, NAME_2, District_N).
# Set the one that actually matches your GeoJSON here.
DISTRICT_NAME_PROPERTY = "district"

START_YEAR = 2013
END_YEAR   = 2025

OUTPUT_CSV = "Telangana_Climate_Master_GEE_2013_2025.csv"

# reduceRegions scale in meters. CHIRPS ~5.5km, ERA5-Land ~11km, MODIS NDVI 250m.
# 5000m is a reasonable middle ground for district-level averages.
SCALE_METERS = 5000

# ==========================================================
# INIT EARTH ENGINE
# ==========================================================

print("=" * 70)
print("INITIALIZING EARTH ENGINE")
print("=" * 70)

ee.Initialize(project=GEE_PROJECT)
print("Earth Engine initialized successfully.\n")

# ==========================================================
# LOAD DISTRICT BOUNDARIES
# ==========================================================

print("Loading district boundaries...")
with open(DISTRICT_GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

for feat in geojson_data["features"]:
    props = feat.setdefault("properties", {})
    if DISTRICT_NAME_PROPERTY not in props:
        raise KeyError(
            f"Property '{DISTRICT_NAME_PROPERTY}' not found in GeoJSON feature properties: "
            f"{list(props.keys())}. Update DISTRICT_NAME_PROPERTY at the top of this script "
            "to match whichever key actually holds the district name."
        )

districts_fc = ee.FeatureCollection(geojson_data)
district_names = sorted({f["properties"][DISTRICT_NAME_PROPERTY] for f in geojson_data["features"]})
print(f"Loaded {len(district_names)} district polygons.\n")

# ==========================================================
# DATASET DEFINITIONS
# ==========================================================

CHIRPS     = ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_RNL")
ERA5       = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
MODIS_NDVI = ee.ImageCollection("MODIS/061/MOD13Q1")


def month_bounds(y, m):
    start = ee.Date.fromYMD(y, m, 1)
    end = start.advance(1, "month")
    return start, end


def rainfall_monthly(y, m):
    start, end = month_bounds(y, m)
    return CHIRPS.filterDate(start, end).select("precipitation").sum().rename("Rainfall")


def temperature_monthly(y, m):
    start, end = month_bounds(y, m)
    return (ERA5.filterDate(start, end)
                 .select("temperature_2m")
                 .mean()
                 .subtract(273.15)          # Kelvin -> Celsius
                 .rename("Temperature"))


def soil_moisture_monthly(y, m):
    start, end = month_bounds(y, m)
    return (ERA5.filterDate(start, end)
                 .select("volumetric_soil_water_layer_1")
                 .mean()
                 .rename("Soil_Moisture"))


def ndvi_monthly(y, m):
    start, end = month_bounds(y, m)
    return (MODIS_NDVI.filterDate(start, end)
                       .select("NDVI")
                       .mean()
                       .multiply(0.0001)    # MODIS NDVI scale factor
                       .rename("NDVI"))


# ==========================================================
# EXTRACTION HELPER
# ==========================================================

def reduce_image_to_districts(image):
    """Runs reduceRegions for one image over all districts, returns {district: mean_value}."""
    stats = image.reduceRegions(
        collection=districts_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_METERS,
    )
    result = stats.getInfo()
    out = {}
    for feat in result["features"]:
        props = feat["properties"]
        dname = props.get(DISTRICT_NAME_PROPERTY)
        out[dname] = props.get("mean")
    return out


# ==========================================================
# EXTRACTION LOOP
# ==========================================================

records = []

print("=" * 70)
print(f"EXTRACTING MONTHLY DATA: {START_YEAR}-01 to {END_YEAR}-12")
print("This calls Earth Engine ~4 times per month (Rainfall, Temperature,")
print("Soil Moisture, NDVI) across all districts — expect this to take a")
print("while for a 13-year backfill. Progress is printed per month.")
print("=" * 70)

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        if date(year, month, 1) > date.today():
            break

        print(f"Processing {year}-{month:02d} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            rain_vals = reduce_image_to_districts(rainfall_monthly(year, month))
            temp_vals = reduce_image_to_districts(temperature_monthly(year, month))
            sm_vals   = reduce_image_to_districts(soil_moisture_monthly(year, month))
            ndvi_vals = reduce_image_to_districts(ndvi_monthly(year, month))
        except Exception as e:
            print(f"FAILED ({e}) — skipping this month")
            continue

        for dname in district_names:
            records.append({
                "District": dname,
                "Year": year,
                "Month": month,
                "Rainfall": rain_vals.get(dname),
                "Temperature": temp_vals.get(dname),
                "Soil_Moisture": sm_vals.get(dname),
                "NDVI": ndvi_vals.get(dname),
            })

        print(f"done ({time.time() - t0:.1f}s)")

# ==========================================================
# SAVE OUTPUT
# ==========================================================

if not records:
    raise RuntimeError(
        "No months were successfully extracted — every request to Earth Engine failed. "
        "Check the FAILED messages above for the actual error (wrong dataset ID, wrong "
        "GEE_PROJECT, or missing Earth Engine registration are the most common causes) "
        "before re-running."
    )

df = pd.DataFrame(records)
df = df.sort_values(["District", "Year", "Month"]).reset_index(drop=True)

print("\nShape:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum())

df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 70)
print(f"SAVED: {OUTPUT_CSV}")
print("=" * 70)
print("\nNext step: point otherfeatures.py's input at this CSV (adjusting")
print("column handling as needed) to compute SPI-3, lag features, and")
print("Groundwater_Proxy, producing a new Telangana_Model_Input.csv for")
print("train_drought_model.py.")