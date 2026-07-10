# ==========================================================
# GEE AUTO-UPDATE PIPELINE
#
# Meant to be run unattended (e.g. via Windows Task Scheduler,
# monthly). Each run:
#   1. Checks the existing climate master CSV for the latest
#      District/Year/Month already present.
#   2. Fetches only the missing month(s) from Earth Engine —
#      not a full backfill.
#   3. Appends the new rows and saves the master CSV.
#   4. Re-runs otherfeatures.py to rebuild Telangana_Model_Input.csv.
#   5. Re-runs train_drought_model.py to retrain the model on
#      the updated data.
#
# Every step logs to auto_update_log.txt (created next to this
# script) so you can check what happened without watching it run.
# ==========================================================

import os
import sys
import subprocess
import datetime
import traceback

import ee
import pandas as pd

# ==========================================================
# CONFIG — should match gee_data_pipeline.py
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)   # Task Scheduler often starts in a different working
                        # directory (e.g. System32) — this forces every
                        # relative path below to resolve next to this script.

GEE_PROJECT = "project-8928310d-34bc-4bd5-aef"   # <-- same project ID as your other GEE scripts

DISTRICT_GEOJSON_PATH = r"D:\Drought_Temp\telangana_districts.geojson"
DISTRICT_NAME_PROPERTY = "district"

MASTER_CSV      = os.path.join(SCRIPT_DIR, "Telangana_Climate_Master_GEE_2013_2025.csv")
OTHERFEATURES_SCRIPT = os.path.join(SCRIPT_DIR, "otherfeatures.py")
TRAIN_SCRIPT          = os.path.join(SCRIPT_DIR, "train_drought_model.py")
LOG_FILE              = os.path.join(SCRIPT_DIR, "auto_update_log.txt")

SCALE_METERS = 5000

# Satellite products (esp. MODIS NDVI, CHIRPS) often aren't published for the
# most recent 1-2 months yet. LAG_MONTHS controls how far back from "today"
# we're willing to consider a month "should be available by now".
LAG_MONTHS = 1


# ==========================================================
# LOGGING
# ==========================================================

def log(msg):
    stamped = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(stamped)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(stamped + "\n")


# ==========================================================
# DETERMINE WHICH MONTHS ARE MISSING
# ==========================================================

def latest_month_in_csv(path):
    """Returns (year, month) of the most recent record in the master CSV."""
    df = pd.read_csv(path)
    df = df.sort_values(["Year", "Month"])
    last = df.iloc[-1]
    return int(last["Year"]), int(last["Month"])


def month_add(y, m, n=1):
    m += n
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    return y, m


def months_to_fetch(last_y, last_m):
    """List of (year, month) tuples strictly after the last recorded month,
    up to (current month - LAG_MONTHS), inclusive."""
    today = datetime.date.today()
    cutoff_y, cutoff_m = month_add(today.year, today.month, -LAG_MONTHS)

    months = []
    y, m = month_add(last_y, last_m, 1)
    while (y, m) <= (cutoff_y, cutoff_m):
        months.append((y, m))
        y, m = month_add(y, m, 1)
    return months


# ==========================================================
# EARTH ENGINE EXTRACTION (same logic as gee_data_pipeline.py)
# ==========================================================

def init_gee():
    ee.Initialize(project=GEE_PROJECT)


def load_districts():
    import json
    with open(DISTRICT_GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        props = feat.setdefault("properties", {})
        if DISTRICT_NAME_PROPERTY not in props:
            raise KeyError(
                f"Property '{DISTRICT_NAME_PROPERTY}' not found in GeoJSON properties: "
                f"{list(props.keys())}"
            )
    fc = ee.FeatureCollection(gj)
    names = sorted({f["properties"][DISTRICT_NAME_PROPERTY] for f in gj["features"]})
    return fc, names


# Dataset IDs — NOT constructed as ee.ImageCollection here, since that would run at
# module-load time, before main() has called init_gee() / ee.Initialize(). Built inside
# the functions below instead, after Earth Engine is actually initialized.
CHIRPS_ID     = "UCSB-CHC/CHIRPS/V3/DAILY_RNL"
ERA5_ID       = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
MODIS_NDVI_ID = "MODIS/061/MOD13Q1"


def month_bounds(y, m):
    start = ee.Date.fromYMD(y, m, 1)
    end = start.advance(1, "month")
    return start, end


def rainfall_monthly(y, m):
    start, end = month_bounds(y, m)
    return ee.ImageCollection(CHIRPS_ID).filterDate(start, end).select("precipitation").sum().rename("Rainfall")


def temperature_monthly(y, m):
    start, end = month_bounds(y, m)
    return (ee.ImageCollection(ERA5_ID).filterDate(start, end).select("temperature_2m").mean()
                .subtract(273.15).rename("Temperature"))


def soil_moisture_monthly(y, m):
    start, end = month_bounds(y, m)
    return (ee.ImageCollection(ERA5_ID).filterDate(start, end).select("volumetric_soil_water_layer_1")
                .mean().rename("Soil_Moisture"))


def ndvi_monthly(y, m):
    start, end = month_bounds(y, m)
    return (ee.ImageCollection(MODIS_NDVI_ID).filterDate(start, end).select("NDVI").mean()
                .multiply(0.0001).rename("NDVI"))


def reduce_image_to_districts(image, districts_fc):
    stats = image.reduceRegions(collection=districts_fc, reducer=ee.Reducer.mean(), scale=SCALE_METERS)
    result = stats.getInfo()
    out = {}
    for feat in result["features"]:
        props = feat["properties"]
        out[props.get(DISTRICT_NAME_PROPERTY)] = props.get("mean")
    return out


def fetch_month(y, m, districts_fc, district_names):
    """Returns a list of row dicts for all districts for one month, or None if
    the underlying satellite data isn't published yet for this month."""
    try:
        rain_vals = reduce_image_to_districts(rainfall_monthly(y, m), districts_fc)
        temp_vals = reduce_image_to_districts(temperature_monthly(y, m), districts_fc)
        sm_vals   = reduce_image_to_districts(soil_moisture_monthly(y, m), districts_fc)
        ndvi_vals = reduce_image_to_districts(ndvi_monthly(y, m), districts_fc)
    except Exception as e:
        log(f"  {y}-{m:02d}: fetch failed ({e}) — likely not published yet, will retry next run")
        return None

    rows = []
    for dname in district_names:
        r, t, s, n = rain_vals.get(dname), temp_vals.get(dname), sm_vals.get(dname), ndvi_vals.get(dname)
        if None in (r, t, s, n):
            log(f"  {y}-{m:02d}: incomplete data for {dname} — will retry next run")
            return None
        rows.append({"District": dname, "Year": y, "Month": m,
                      "Rainfall": r, "Temperature": t, "Soil_Moisture": s, "NDVI": n})
    return rows


# ==========================================================
# SUBPROCESS STEPS
# ==========================================================

def run_script(path, label):
    log(f"Running {label} ({os.path.basename(path)}) ...")
    result = subprocess.run(
        [sys.executable, path],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        input="N\n",   # answers train_drought_model.py's interactive "Y/N" prompt with N
    )
    log(f"{label} exit code: {result.returncode}")
    if result.stdout:
        log(f"{label} stdout (last 2000 chars):\n{result.stdout[-2000:]}")
    if result.returncode != 0:
        log(f"{label} stderr:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"{label} failed — see log above")


# ==========================================================
# MAIN
# ==========================================================

def main():
    log("=" * 70)
    log("GEE AUTO-UPDATE RUN STARTED")
    log("=" * 70)

    if not os.path.exists(MASTER_CSV):
        log(f"ERROR: Master CSV not found at {MASTER_CSV}. "
            f"Run gee_data_pipeline.py once manually first to create it.")
        return

    last_y, last_m = latest_month_in_csv(MASTER_CSV)
    log(f"Latest month currently in master CSV: {last_y}-{last_m:02d}")

    pending = months_to_fetch(last_y, last_m)
    if not pending:
        log("No new months due yet (nothing past the satellite-lag cutoff). Nothing to do.")
        return

    log(f"Months to attempt: {', '.join(f'{y}-{m:02d}' for y, m in pending)}")

    init_gee()
    districts_fc, district_names = load_districts()
    log(f"Loaded {len(district_names)} districts.")

    new_rows = []
    for y, m in pending:
        log(f"Fetching {y}-{m:02d} ...")
        rows = fetch_month(y, m, districts_fc, district_names)
        if rows is None:
            # Stop at the first unavailable month — months should stay contiguous,
            # so don't skip ahead and leave a gap.
            log(f"Stopping at {y}-{m:02d} — will pick up remaining months next scheduled run.")
            break
        new_rows.extend(rows)
        log(f"  {y}-{m:02d}: OK ({len(rows)} district rows)")

    if not new_rows:
        log("No new complete months were available this run. Exiting without changes.")
        return

    # ── Append and save ──
    existing = pd.read_csv(MASTER_CSV)
    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.drop_duplicates(subset=["District", "Year", "Month"], keep="last")
    updated = updated.sort_values(["District", "Year", "Month"]).reset_index(drop=True)
    updated.to_csv(MASTER_CSV, index=False)
    log(f"Master CSV updated: {len(new_rows)} new rows added, {len(updated)} total rows.")

    # ── Rebuild model input ──
    run_script(OTHERFEATURES_SCRIPT, "otherfeatures.py")

    # ── Retrain ──
    run_script(TRAIN_SCRIPT, "train_drought_model.py")

    log("=" * 70)
    log("GEE AUTO-UPDATE RUN COMPLETED SUCCESSFULLY")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("RUN FAILED WITH AN EXCEPTION:")
        log(traceback.format_exc())
        sys.exit(1)