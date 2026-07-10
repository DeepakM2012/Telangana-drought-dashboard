import pandas as pd
import numpy as np

# ==========================================================
# READ MASTER DATASET
# ==========================================================

df = pd.read_csv("Telangana_Climate_Master_GEE_2013_2025.csv")

# ==========================================================
# SORT DATA
# ==========================================================

df = df.sort_values(
    ["District", "Year", "Month"]
).reset_index(drop=True)

# ==========================================================
# CREATE DATE COLUMN
# ==========================================================

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=1
    )
)

# ==========================================================
# CALCULATE SPI-3
# ==========================================================

df["SPI3"] = np.nan

for district in df["District"].unique():

    mask = df["District"] == district

    rainfall = df.loc[mask, "Rainfall"]

    rolling = rainfall.rolling(
        window=3,
        min_periods=3
    ).sum()

    spi = (
        rolling - rolling.mean()
    ) / rolling.std()

    df.loc[mask, "SPI3"] = spi.values

# ==========================================================
# CREATE LAG FEATURES
# ==========================================================

df["Rainfall_lag1"] = df.groupby("District")["Rainfall"].shift(1)

df["SoilMoisture_lag1"] = df.groupby("District")["Soil_Moisture"].shift(1)

df["NDVI_lag1"] = df.groupby("District")["NDVI"].shift(1)

# ==========================================================
# FILL FIRST RECORD OF EACH DISTRICT
# ==========================================================

df["Rainfall_lag1"] = df["Rainfall_lag1"].fillna(df["Rainfall"])

df["SoilMoisture_lag1"] = df["SoilMoisture_lag1"].fillna(df["Soil_Moisture"])

df["NDVI_lag1"] = df["NDVI_lag1"].fillna(df["NDVI"])

# ==========================================================
# NORMALIZE FEATURES (0-1)
# ==========================================================

def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

df["Rainfall_lag1_norm"] = normalize(df["Rainfall_lag1"])

df["SoilMoisture_lag1_norm"] = normalize(df["SoilMoisture_lag1"])

df["NDVI_norm"] = normalize(df["NDVI"])

# ==========================================================
# CREATE GROUNDWATER PROXY
# ==========================================================

df["Groundwater_Proxy"] = (
      0.40 * df["Rainfall_lag1_norm"]
    + 0.40 * df["SoilMoisture_lag1_norm"]
    + 0.20 * df["NDVI_norm"]
)

# ==========================================================
# REMOVE ROWS WHERE SPI3 IS NaN
# (First two months of each district)
# ==========================================================

df = df.dropna(subset=["SPI3"])

# ==========================================================
# FINAL COLUMN ORDER
# ==========================================================

df = df[
    [
        "District",
        "Year",
        "Month",
        "Date",
        "Rainfall",
        "Temperature",
        "Soil_Moisture",
        "NDVI",
        "SPI3",
        "Rainfall_lag1",
        "SoilMoisture_lag1",
        "NDVI_lag1",
        "Rainfall_lag1_norm",
        "SoilMoisture_lag1_norm",
        "NDVI_norm",
        "Groundwater_Proxy"
    ]
]

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    "Telangana_Model_Input.csv",
    index=False
)

print("=" * 70)
print("TELANGANA MODEL INPUT DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print("\nShape :", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nFile Saved As: Telangana_Model_Input.csv")