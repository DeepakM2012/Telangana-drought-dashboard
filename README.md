# 🌾 Telangana AI-Based Drought Prediction and Risk Assessment System

## 📌 Overview

The **Telangana AI-Based Drought Prediction and Risk Assessment System** is an interactive web application developed using **Streamlit** to monitor, analyze, and predict drought conditions across all **33 districts of Telangana, India**.

The project combines **Machine Learning, Geographic Information Systems (GIS), Remote Sensing, and Google Earth Engine (GEE)** to provide district-level drought predictions, interactive visualizations, and decision-support tools for drought management.

---

## 🎯 Objectives

- Predict drought severity using Machine Learning.
- Monitor drought conditions across Telangana districts.
- Visualize drought data using GIS maps.
- Analyze environmental factors affecting drought.
- Support policymakers with data-driven insights.

---

## 🚀 Features

- 📍 Interactive GIS Map of Telangana Districts
- 🤖 AI-Based Drought Prediction
- 🌦️ Live Google Earth Engine Data Integration
- 📊 Interactive Charts and Visualizations
- 📈 District-wise Risk Analysis
- 📉 Model Performance Comparison
- 🔍 Feature Importance Analysis
- 📋 Confidence Score Estimation
- 📄 Export Reports
- 🌐 User-Friendly Web Dashboard

---

## 🛰️ Data Sources

The project utilizes the following datasets:

- CHIRPS Rainfall Dataset
- ERA5-Land Temperature Dataset
- ERA5-Land Soil Moisture Dataset
- MODIS NDVI Dataset
- Telangana District Boundary GeoJSON

---

## 🤖 Machine Learning Models

The prediction system uses a **Stacking Ensemble Learning** approach.

### Base Models

- Random Forest Classifier
- XGBoost Classifier

### Meta Model

- Logistic Regression

The model predicts district drought severity using environmental indicators and engineered features.

---

## 📊 Input Features

- Rainfall
- Temperature
- Soil Moisture
- NDVI
- SPI-3
- Groundwater Proxy
- Previous Month Rainfall
- Previous Month Soil Moisture
- Previous Month NDVI

---

## 📂 Project Structure

```text
telangana-drought-dashboard/
│
├── telangana_drought_dashboard.py
├── gee_live_fetch.py
├── gee_data_pipeline.py
├── gee_auto_update.py
├── train_drought_model.py
├── otherfeatures.py
│
├── District_Drought_Predictions.csv
├── District_Model_Comparison.csv
├── District_Feature_Importance.csv
├── District_Model_Features.pkl
├── Telangana_District_Drought_Model.pkl
├── Telangana_Model_Input.csv
├── Telangana_Climate_Master_GEE_2013_2025.csv
├── telangana_districts.geojson
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| GIS & Mapping | Folium, Streamlit-Folium, GeoJSON |
| Visualization | Plotly, Matplotlib |
| Cloud Platform | Google Earth Engine |
| Model Storage | Joblib |
| Version Control | Git & GitHub |

---

## 📈 Dashboard Modules

- Dashboard Overview
- District-wise Drought Analysis
- GIS Interactive Map
- Live Prediction
- Model Performance
- Feature Importance
- Risk Assessment
- Report Generation

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/telangana-drought-dashboard.git
```

### Navigate to the Project

```bash
cd telangana-drought-dashboard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run telangana_drought_dashboard.py
```

---

## 🔄 Project Workflow

```text
Google Earth Engine
        │
        ▼
Climate Data Collection
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Model
        │
        ▼
Drought Prediction
        │
        ▼
Interactive Streamlit Dashboard
```

---

## 📊 Model Workflow

1. Collect satellite and climate data.
2. Generate engineered features.
3. Train the stacking ensemble model.
4. Predict district drought severity.
5. Visualize results on an interactive dashboard.
6. Fetch live data using Google Earth Engine.
7. Generate reports and analytics.

---

## 🌍 Applications

- Drought Monitoring
- Agricultural Planning
- Water Resource Management
- Climate Change Studies
- Government Decision Support
- Disaster Risk Management
- Environmental Research

---

## 📌 Future Enhancements

- Real-time Weather API Integration
- Mobile Responsive Dashboard
- SMS & Email Alerts
- Forecast-Based Drought Prediction
- Satellite Image Visualization
- Multi-State Expansion
- Cloud Database Integration

---

## 📷 Dashboard Preview

> Add screenshots of your dashboard here.

Example:

```
images/
├── dashboard.png
├── gis_map.png
├── prediction.png
```

---

## 👨‍💻 Author

**Deepak**

**AI-Based Drought Prediction and Risk Assessment System**

Developed as an academic project using **Machine Learning, GIS, Remote Sensing, and Google Earth Engine**.

---

## 📄 License

This project is intended for educational and research purposes. Feel free to fork, modify, and use it with proper attribution.

---

## ⭐ Acknowledgements

- Google Earth Engine
- Streamlit
- Scikit-learn
- XGBoost
- Plotly
- Folium
- Telangana Open GIS Data
- CHIRPS Rainfall Dataset
- ERA5-Land Climate Dataset
- MODIS NDVI Dataset
