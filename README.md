# 🌾 Crop Yield Prediction — Indian Agriculture

> **DSC-413 · Data Warehousing & Business Intelligence**  
> Indian Institute of Science Education and Research, Thiruvananthapuram (IISERTVM)  
> Academic Year 2025–2026

A machine-learning pipeline that forecasts district-level crop yields across Indian states using historical agricultural records (1997–2014), coupled with an interactive Gradio web UI for real-time scenario modelling.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Gradio UI](#gradio-ui)
- [Installation](#installation)
- [Usage](#usage)
- [Future Roadmap](#future-roadmap)

---

## Overview

India wastes nearly **40% of its harvested produce** annually due to poor demand forecasting and supply-chain mismatches. This project builds a data-driven yield forecasting system that:

- Ingests multi-dimensional agricultural records (state, district, crop, season, area, environmental conditions)
- Applies **Decision Tree**, **Random Forest**, **SVR**, **K-Means**, and **DBSCAN** algorithms
- Compares model performance using R², MAE, and RMSE
- Exposes predictions through a **Gradio interactive UI** accessible to agricultural officers and researchers

| Attribute | Value |
|-----------|-------|
| Dataset Records | 49,999 (49,499 after cleaning) |
| Time Span | 1997–2014 |
| States Covered | 7 Indian states |
| Crop Varieties | 80 |
| Primary Models | Decision Tree · Random Forest · SVR · K-Means · DBSCAN |
| Deployment | Gradio Interactive UI |

---

## Project Structure

```
A-Survey-of-Indian-Crop-Yield/
│
├── 📂 Deliverable_1/               # Phase 1 — EDA & initial analysis
│   ├── deliverable1_report.pdf     # Deliverable 1 report document
│   └── ...                         # Supporting notebooks / scripts
│
├── 📂 Gradio_results/              # Screenshots and outputs from the Gradio UI
│   └── *.jpeg / *.png              # UI demo screenshots
│
├── 📂 results/
│   └── plots/                      # Generated visualisation plots
│       ├── correlation_matrix.png  # Feature correlation heatmap
│       ├── decision_tree_prediction.png
│       ├── random_forest_prediction.png
│       ├── svr_prediction.png
│       ├── kmeans.png
│       └── dbscan.png
│
├── 📄 Crop Prediction dataset.csv  # Raw dataset (49,999 records)
├── 📄 DW_report.pdf                # Final combined project report (Deliverable 2)
├── 📓 model.ipynb                  # Main Jupyter notebook — EDA, training, evaluation
├── 🐍 crop_prediction_1.py         # Gradio application — interactive inference UI
└── 📄 README.md                    # This file
```

---

## Dataset

**Source:** [Kaggle — Crop Yield Prediction Based on Indian Agriculture](https://www.kaggle.com/)  
Mirrors historical records from the Indian Ministry of Agriculture.

| Feature | Type | Description |
|---------|------|-------------|
| `State_Name` | Categorical | Indian state name |
| `District_Name` | Categorical | District within the state |
| `Crop_Year` | Integer | Year of cultivation (1997–2014) |
| `Season` | Categorical | Kharif, Rabi, Whole Year, etc. |
| `Crop` | Categorical | Crop variety (80 types) |
| `Temperature` | Integer | Average temperature (°C) |
| `Humidity` | Integer | Average humidity (%) |
| `Soil_Moisture` | Integer | Soil moisture (%) |
| `Area` | Float | Cultivated area (hectares) |
| `Production` | Float | Total production (tonnes) |
| **`Yield` (target)** | **Float** | **Derived: Production ÷ Area (t/ha)** |

### Preprocessing Steps

1. Type coercion — `Production` and `Area` to numeric; non-numeric → `NaN`
2. Missing value imputation — `Production` NaN filled with column mean
3. String normalisation — trailing whitespace stripped from categorical columns
4. Yield computation — `Yield = Production / Area`; zero-area rows dropped
5. Outlier removal — top 1% of Yield distribution removed (99th percentile)
6. One-hot encoding — categoricals encoded with `pd.get_dummies()` (9 → 210 features)
7. Train/test split — 75% train (37,124) / 25% test (12,375), `random_state=42`

---

## Methodology

### Supervised Regression

| Model | Key Config |
|-------|-----------|
| Decision Tree Regressor | `criterion=mse`, `random_state=42`, no depth limit |
| Random Forest Regressor | `n_estimators=100`, `n_jobs=-1`, `random_state=42` |
| Support Vector Regressor | RBF kernel, normalised feature subset |

### Unsupervised Clustering

| Model | Key Config |
|-------|-----------|
| K-Means | `k=3`, `init=k-means++`, `random_state=42` |
| DBSCAN | `eps=0.5`, `min_samples=5`, standardised features |

---

## Results

### Model Performance (25% hold-out, n = 12,375)

| Model | R² Score | MAE (t/ha) | RMSE (t/ha) |
|-------|----------|-----------|------------|
| Decision Tree Regressor | 0.7368 | 5.04 | 104.48 |
| **Random Forest Regressor** | **0.7688** | 5.08 | **97.93** |
| Support Vector Regressor | ~0.65–0.70 | — | — |

### Top Feature Importances (Random Forest)

| Feature | Importance | Rank |
|---------|-----------|------|
| Crop_Coconut | 0.4554 | 1 |
| Crop_Year | 0.3671 | 2 |
| Area | 0.0524 | 3 |
| District_Name_BILASPUR | 0.0195 | 4 |
| State_Name_Chhattisgarh | 0.0163 | 5 |

### K-Means Cluster Interpretation

| Cluster | Profile | Policy Implication |
|---------|---------|-------------------|
| 0 | Low yield · small area (subsistence farming) | Input subsidies · irrigation |
| 1 | Medium yield · large area (commercial grain belt) | Market linkages |
| 2 | High yield · plantation crops (coconut, sugarcane) | Value-chain development |

---

## Gradio UI

`crop_prediction_1.py` launches an interactive web app with:

- **Cascading dropdowns** — State → District (dynamically filtered)
- **Crop Year slider** — supports extrapolation beyond 2014
- **Environmental sliders** — Temperature, Humidity, Soil Moisture
- **Dual model output** — side-by-side Decision Tree & Random Forest predictions
- **Model agreement indicator** — flags >30% disagreement between models
- **Ensemble average** — auto-computed mean yield across models

```bash
python crop_prediction_1.py
# Opens at http://localhost:7860
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sanjay-balusamy/A-Survey-of-Indian-Crop-Yield.git
cd A-Survey-of-Indian-Crop-Yield

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn gradio jupyter
```

---

## Usage

### Run the Jupyter Notebook (EDA + Training)

```bash
jupyter notebook model.ipynb
```

### Launch the Gradio App

```bash
python crop_prediction_1.py
```

---

## Future Roadmap

| Limitation | Proposed Solution | Priority |
|-----------|------------------|----------|
| One-hot encoding explosion | Target encoding / embeddings for high-cardinality categoricals | 🔴 High |
| SVR scalability (O(n²–n³)) | Replace with LightGBM / XGBoost | 🔴 High |
| No model persistence | Serialise with `joblib` / `pickle` at training | 🔴 High |
| No incremental learning | Online learning with River or incremental RF | 🟡 Medium |
| Single-process server | FastAPI + Gunicorn + Docker | 🟡 Medium |
| Feature drift | Automated validation with Great Expectations | 🟡 Medium |
| Real-time data | Connect to IMD API for live weather feeds | 🟢 Low |
| Data warehouse | Migrate to PostgreSQL + Apache Airflow ETL | 🟢 Low |

---

## 📄 Report

The full project report is available as [`DW_report.pdf`](./DW_report.pdf) in this repository.

---

## 🏫 Institution

**Indian Institute of Science Education and Research, Thiruvananthapuram (IISERTVM)**  
Department of Data Science · DSC-413 Data Warehousing and Business Intelligence
