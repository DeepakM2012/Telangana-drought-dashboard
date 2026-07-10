# ==========================================================
# DISTRICT DROUGHT PREDICTION MODEL
# Based on Original Prediction_Model.py
# ==========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np

import joblib
import shap

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix,

    precision_score,

    recall_score,

    f1_score

)

from xgboost import XGBClassifier

# ==========================================================
# IMPROVEMENT 1
# LOAD DATA
# ==========================================================

print("="*70)
print("LOADING DISTRICT DATASET")
print("="*70)

DATA_PATH = r"D:\Drought_Temp\Telangana_Model_Input.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Input data file not found at: {DATA_PATH}\n"
        "Update DATA_PATH at the top of the script to point to your "
        "Telangana_Model_Input.csv file."
    )

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nColumns")

print(df.columns.tolist())

# ==========================================================
# IMPROVEMENT 2
# CREATE TARGET
# ==========================================================

def create_label(spi):

    if spi <= -1.0:

        return 2

    elif spi <= -0.5:

        return 1

    else:

        return 0


df["Drought_Label"] = df["SPI3"].apply(create_label)

print("\nClass Distribution")

print(df["Drought_Label"].value_counts())

# ==========================================================
# IMPROVEMENT 3
# FEATURES
# ==========================================================

FEATURES = [

    "Rainfall",

    "Temperature",

    "Soil_Moisture",

    "NDVI",

    "Rainfall_lag1",

    "SoilMoisture_lag1",

    "NDVI_lag1",

    "Groundwater_Proxy"

]

TARGET = "Drought_Label"

X = df[FEATURES]

y = df[TARGET]

# ==========================================================
# IMPROVEMENT 4
# TRAIN TEST SPLIT
# ==========================================================

train_df = df[df["Year"] <= 2023]

test_df = df[df["Year"] >= 2024]

X_train = train_df[FEATURES]

y_train = train_df[TARGET]

X_test = test_df[FEATURES]

y_test = test_df[TARGET]

print("\nTraining Shape :", X_train.shape)

print("Testing Shape  :", X_test.shape)

# ==========================================================
# IMPROVEMENT 5
# RANDOM FOREST
# ==========================================================

print("\n")
print("="*70)
print("TRAINING RANDOM FOREST")
print("="*70)

rf = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)

print("Random Forest Training Completed")

# ==========================================================
# IMPROVEMENT 6
# XGBOOST
# ==========================================================

print("\n")
print("="*70)
print("TRAINING XGBOOST")
print("="*70)

xgb = XGBClassifier(

    n_estimators=300,

    max_depth=8,

    learning_rate=0.05,

    objective="multi:softprob",

    num_class=3,

    random_state=42,

    eval_metric="mlogloss"

)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_prob = xgb.predict_proba(X_test)

print("XGBoost Training Completed")

# ==========================================================
# IMPROVEMENT 7
# STACKING
# ==========================================================

print("\n")
print("="*70)
print("TRAINING STACKING MODEL")
print("="*70)

estimators = [

    ("rf", rf),

    ("xgb", xgb)

]

stack = StackingClassifier(

    estimators=estimators,

    final_estimator=LogisticRegression(),

    stack_method="predict_proba",

    n_jobs=-1

)

stack.fit(X_train, y_train)

pred = stack.predict(X_test)

prob = stack.predict_proba(X_test)

print("Stacking Model Completed")

# ==========================================================
# IMPROVEMENT 8
# MODEL COMPARISON
# ==========================================================

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

rf_acc = accuracy_score(y_test, rf_pred)

xgb_acc = accuracy_score(y_test, xgb_pred)

stack_acc = accuracy_score(y_test, pred)

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "Stacking Ensemble"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred),
        accuracy_score(y_test, pred)
    ],
    "Precision": [
        precision_score(y_test, rf_pred, average="weighted"),
        precision_score(y_test, xgb_pred, average="weighted"),
        precision_score(y_test, pred, average="weighted")
    ],
    "Recall": [
        recall_score(y_test, rf_pred, average="weighted"),
        recall_score(y_test, xgb_pred, average="weighted"),
        recall_score(y_test, pred, average="weighted")
    ],
    "F1 Score": [
        f1_score(y_test, rf_pred, average="weighted"),
        f1_score(y_test, xgb_pred, average="weighted"),
        f1_score(y_test, pred, average="weighted")
    ]
})

print(comparison.round(4))

print(comparison)

models = ["Random Forest", "XGBoost", "Stacking Ensemble"]

accuracy = [
    rf_acc * 100,
    xgb_acc * 100,
    stack_acc * 100
]

colors = ["#2B6CB0", "#38A169", "#DD6B20"]

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracy, color=colors)

for bar in bars:
    bar_height = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar_height+0.05,
        f"{bar_height:.2f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")

plt.ylim(min(accuracy)-1, max(accuracy)+1)

plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

plt.savefig("Accuracy_Comparison.png", dpi=300)

plt.close()

# ==========================================================
# IMPROVEMENT 9
# CROSS VALIDATION
# ==========================================================

print("\n")
print("="*70)
print("RUNNING CROSS VALIDATION")
print("="*70)

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

print(type(y))
print(y)

scores = cross_val_score(
    estimator=stack,
    X=X,
    y=y,
    cv=cv,
    scoring="accuracy"
)

cv_result = pd.DataFrame({

    "Fold":[1,2,3,4,5],

    "Accuracy":scores

})

print(cv_result)

print("\nMean Accuracy :", round(scores.mean(),4))

print("Std Accuracy  :", round(scores.std(),4))

# ==========================================================
# IMPROVEMENT 10
# CLASSIFICATION REPORT
# ==========================================================

print("\n")
print("="*70)
print("CLASSIFICATION REPORT")
print("="*70)

print(

classification_report(

    y_test,

    pred,

    target_names=[

        "Low",

        "Moderate",

        "High"

    ]

)

)

print("Precision :", precision_score(y_test,pred,average="weighted"))

print("Recall    :", recall_score(y_test,pred,average="weighted"))

print("F1 Score  :", f1_score(y_test,pred,average="weighted"))

# ==========================================================
# IMPROVEMENT 11
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_test,

    pred

)

plt.figure(figsize=(7,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="YlOrRd",

    xticklabels=[

        "Low",

        "Moderate",

        "High"

    ],

    yticklabels=[

        "Low",

        "Moderate",

        "High"

    ]

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    "Confusion_Matrix.png",

    dpi=300

)

plt.close()

# ==========================================================
# IMPROVEMENT 12
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({

    "Feature":FEATURES,

    "Importance":rf.feature_importances_

})

importance = importance.sort_values(

    "Importance",

    ascending=False

)

print("\n")

print(importance)

importance.to_csv(

    "Feature_Importance.csv",

    index=False

)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"],
    color="#2B6CB0"
)

plt.xlabel("Importance")

plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("Feature_Importance.png", dpi=300)

plt.close()

# ==========================================================
# IMPROVEMENT 13
# SHAP EXPLAINABILITY
# ==========================================================

print("\n")
print("="*70)
print("GENERATING SHAP")
print("="*70)

explainer = shap.TreeExplainer(rf)

sample = X_test.sample(
    n=min(300, len(X_test)),
    random_state=42
)

shap_values = explainer.shap_values(sample)

plt.figure()

shap.summary_plot(

    shap_values,

    sample,

    feature_names=FEATURES,

    show=False

)

plt.tight_layout()

plt.savefig(

    "SHAP_Feature_Importance.png",

    dpi=300

)

plt.close()

print("SHAP Saved Successfully")

# ==========================================================
# IMPROVEMENT 14
# CREATE DISTRICT PREDICTIONS
# ==========================================================

future_df = df.copy()

future_df["Predicted_Class"] = stack.predict(
    future_df[FEATURES]
)

future_df["Confidence"] = stack.predict_proba(
    future_df[FEATURES]
).max(axis=1)

label_map = {

    0:"Low",

    1:"Moderate",

    2:"High"

}

future_df["Predicted_Class"] = future_df["Predicted_Class"].map(label_map)

print("\nPrediction Table Created")

# ==========================================================
# IMPROVEMENT 15
# DISTRICT DROUGHT RISK SCORE
# ==========================================================

print("\n")
print("="*70)
print("GENERATING DROUGHT RISK SCORE")
print("="*70)

from sklearn.preprocessing import MinMaxScaler

risk_features = future_df[

    [

        "Rainfall",

        "Temperature",

        "Soil_Moisture",

        "NDVI",

        "Groundwater_Proxy"

    ]

].copy()

# Fill missing values before scaling
risk_features = risk_features.fillna(risk_features.mean())

# Normalize Features

scaler = MinMaxScaler()

risk_scaled = pd.DataFrame(

    scaler.fit_transform(risk_features),

    columns=risk_features.columns

)

future_df["RiskScore"] = (

      0.30 * (1 - risk_scaled["Rainfall"])

    + 0.25 * (risk_scaled["Temperature"])

    + 0.20 * (1 - risk_scaled["Soil_Moisture"])

    + 0.10 * (1 - risk_scaled["NDVI"])

    + 0.15 * (1 - risk_scaled["Groundwater_Proxy"])

) * 100

future_df["RiskScore"] = future_df["RiskScore"].round(2)

print("Risk Score Generated")

# ==========================================================
# IMPROVEMENT 16
# RISK LEVEL
# ==========================================================

def risk_level(score):

    if score >= 70:

        return "High"

    elif score >= 40:

        return "Moderate"

    else:

        return "Low"

future_df["RiskLevel"] = future_df["RiskScore"].apply(risk_level)

print("Risk Levels Assigned")

# ==========================================================
# IMPROVEMENT 17
# ADVISORY SYSTEM
# ==========================================================

def advisory(level):

    if level == "High":

        return (
            "• Immediate drought preparedness\n"
            "• Promote water conservation\n"
            "• Reduce irrigation losses\n"
            "• Encourage groundwater recharge\n"
            "• Use drought-resistant crops"
        )

    elif level == "Moderate":

        return (
            "• Monitor rainfall regularly\n"
            "• Improve irrigation efficiency\n"
            "• Prepare backup water sources\n"
            "• Monitor crop health"
        )

    else:

        return (
            "• Normal Conditions\n"
            "• Continue monitoring\n"
            "• Maintain efficient water use"
        )

future_df["Advisory"] = future_df["RiskLevel"].apply(advisory)

print("Advisory Generated")

# ==========================================================
# IMPROVEMENT 18
# SAVE MODEL
# ==========================================================

joblib.dump(stack, "Telangana_District_Drought_Model.pkl", compress=3)

joblib.dump(

    FEATURES,

    "District_Model_Features.pkl"

)

print("\nModel Saved Successfully")

print("Feature List Saved Successfully")

comparison.to_csv(
    "District_Model_Comparison.csv",
    index=False
)

# ==========================================================
# IMPROVEMENT 19
# SAVE PREDICTIONS
# ==========================================================

future_df.sort_values(
    ["District","Year","Month"]
).to_csv(
    "District_Drought_Predictions.csv",
    index=False
)

print("\nPrediction CSV Saved")

# ==========================================================
# IMPROVEMENT 20
# SAVE FEATURE IMPORTANCE
# ==========================================================

importance.to_csv(

    "District_Feature_Importance.csv",

    index=False

)

cv_result.to_csv(

    "District_CrossValidation.csv",

    index=False

)

print("Reports Saved")

# ==========================================================
# IMPROVEMENT 21
# INTERACTIVE DISTRICT PREDICTION SYSTEM
# ==========================================================

print("\n")
print("="*70)
print("DISTRICT DROUGHT PREDICTION SYSTEM")
print("="*70)

districts = sorted(future_df["District"].unique())

while True:

    try:
        choice = input("\nDo you want prediction? (Y/N): ").upper()
    except EOFError:
        print("\nNo interactive input available. Skipping prediction console.")
        break

    if choice == "N":

        print("\nSystem Closed Successfully")

        break

    print("\nAvailable Districts:\n")

    print(", ".join(districts))

    district = input("\nEnter District : ").strip().upper()

    year = int(input("Enter Year : "))

    month = int(input("Enter Month (1-12): "))

    result = future_df[

        (future_df["District"].str.upper()==district)

        &

        (future_df["Year"]==year)

        &

        (future_df["Month"]==month)

    ]

    if result.empty:

        print("\nNo Record Found")

        continue

    row = result.iloc[0]

    print("\n")

    print("="*60)

    print("DISTRICT DROUGHT REPORT")

    print("="*60)

    print(f"District            : {row['District']}")

    print(f"Year                : {row['Year']}")

    print(f"Month               : {row['Month']}")

    print("----------------------------------------------")

    print(f"Rainfall            : {row['Rainfall']:.2f}")

    print(f"Temperature         : {row['Temperature']:.2f}")

    print(f"Soil Moisture       : {row['Soil_Moisture']:.3f}")

    print(f"NDVI                : {row['NDVI']:.3f}")

    print(f"Groundwater Proxy   : {row['Groundwater_Proxy']:.3f}")

    print("----------------------------------------------")

    print(f"Prediction          : {row['Predicted_Class']}")

    print(f"Confidence          : {row['Confidence']*100:.2f}%")

    print(f"Risk Score          : {row['RiskScore']:.2f}")

    print(f"Risk Level          : {row['RiskLevel']}")

    print("----------------------------------------------")

    print("ADVISORY")

    print("----------------------------------------------")

    print(row["Advisory"])

    print("="*60)

# ==========================================================
# IMPROVEMENT 22
# EXCEL REPORT
# ==========================================================

summary = future_df[

    [

        "District",

        "Year",

        "Month",

        "Rainfall",

        "Temperature",

        "Soil_Moisture",

        "NDVI",

        "Groundwater_Proxy",

        "Predicted_Class",

        "Confidence",

        "RiskScore",

        "RiskLevel",

        "Advisory"

    ]

]

summary.to_excel(

    "District_Drought_Report.xlsx",

    index=False

)

print("\nExcel Report Saved Successfully")

# ==========================================================
# IMPROVEMENT 23
# CONFIDENCE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(

    future_df["Confidence"],

    bins=20,

    edgecolor="black"

)

plt.xlabel("Prediction Confidence")

plt.ylabel("Frequency")

plt.title("Prediction Confidence Distribution")

plt.tight_layout()

plt.savefig(

    "Prediction_Confidence.png",

    dpi=300

)

plt.close()

# ==========================================================
# IMPROVEMENT 24
# DISTRICT RISK SUMMARY
# ==========================================================

district_summary = (
    future_df.groupby("District")
    .agg(
        Average_Risk=("RiskScore","mean"),
        Max_Risk=("RiskScore","max"),
        High_Risk_Months=("RiskLevel", lambda x:(x=="High").sum())
    )
    .sort_values("Average_Risk", ascending=False)
    .reset_index()
)

district_summary.to_csv(
    "District_Risk_Summary.csv",
    index=False
)

print("\nDistrict Risk Summary Saved")

# ==========================================================
# IMPROVEMENT 25
# TOP HIGH-RISK DISTRICTS
# ==========================================================

top10 = district_summary.sort_values(

    "Average_Risk",

    ascending=False

).head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10["District"],
    top10["Average_Risk"],
    color="#DD6B20"
)

plt.xlabel("Average Risk Score")

plt.title("Top 10 High Risk Districts")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("Top10_HighRisk_Districts.png", dpi=300)

plt.close()

# ==========================================================
# IMPROVEMENT 26
# FINAL SUMMARY
# ==========================================================

print("\n")

print("="*70)

print("DISTRICT DROUGHT PREDICTION COMPLETED SUCCESSFULLY")

print("="*70)

print("\nGenerated Files")

print("--------------------------------------")

print("1. Telangana_District_Drought_Model.pkl")

print("2. District_Model_Features.pkl")

print("3. District_Drought_Predictions.csv")

print("4. District_Drought_Report.xlsx")

print("5. District_Feature_Importance.csv")

print("6. District_CrossValidation.csv")

print("7. District_Risk_Summary.csv")

print("8. SHAP_Feature_Importance.png")

print("9. Confusion_Matrix.png")

print("10. Feature_Importance.png")

print("11. Prediction_Confidence.png")

print("12. Top10_HighRisk_Districts.png")

print("13. District_Model_Comparison.csv")

print("14. Accuracy_Comparison.png")

print("--------------------------------------")

print("="*70)
print("MODEL READY FOR STREAMLIT DASHBOARD")
print("="*70)