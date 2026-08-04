# 📈 US County Population Prediction Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

A machine learning pipeline that forecasts US county-level population growth by combining **15 years of Census population estimates** (2010–2024) with **building permits data** as a leading indicator.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Visualizations](#-results--visualizations)
- [Data Sources](#-data-sources)

---

## 🚀 Overview

Building permits are a strong leading indicator of population change. The lifecycle typically follows this timeline:
> `Permit issued → Construction (6-18 months) → Occupancy → Census count`

By modeling this pipeline, we can predict population shifts before they are officially recorded. This project uses a **Pooled Panel LightGBM Regressor** trained on ~47,000 county-year records to predict future population growth rates for all ~3,140 US counties.

### Key Features
* **Automated Data Pipeline**: Programmatically downloads and merges Census Population Estimates and Building Permits Survey (BPS) bulk datasets (No API keys required).
* **Advanced Feature Engineering**: Creates lag features (1-3 years), compound annual growth rates (CAGR), rolling averages, permit density, and "Housing Unit Method" implied growth.
* **Robust Evaluation**: Uses expanding-window time-series cross-validation to prevent temporal data leakage.
* **Recursive Forecasting**: Generates multi-step autoregressive predictions for the years 2025–2027.

---

## 🧠 Methodology

1. **Data Reshaping**: Converts wide-format Census data into long-format panel data, aligning by 5-digit Federal Information Processing Standard (FIPS) codes.
2. **Feature Engineering**: 
   - Extracts lagged permits (1, 2, and 3 years) for both single-family and multi-family units.
   - Calculates momentum metrics (permit density, surge index, YoY growth).
   - Incorporates the **Housing Unit Method (HUM)**: `Implied Population = New Units × Avg Household Size × Occupancy Rate`.
3. **Modeling**: A LightGBM Regressor is trained on the pooled panel data, predicting the *annual growth rate* rather than absolute population to ensure stationarity.
4. **Validation**: Time-series cross-validation (training on years $T_1...T_n$, validating on $T_{n+1}$) is used to simulate real-world forecasting conditions.

---

## 📊 Model Performance

Evaluated on the final holdout test set (2023-2024):

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **MAPE** (Mean Absolute Percentage Error) | **0.23%** | Highly accurate; predictions are off by only 0.23% on average. |
| **MALPE** (Mean Algebraic Percentage Error)| **-0.03%**| Virtually no systematic bias (model does not drastically over/under-predict). |
| **R²** (R-squared) | **0.849** | Explains 85% of the variance in county population growth rates. |
| **MAE** (Mean Absolute Error - Population) | **~402** | Average absolute error of ~402 residents per county. |

---

## 📁 Project Structure

```text
us-county-population-model/
├── config.py                   # Central configuration and hyperparameters
├── main.py                     # Main orchestrator script to run the full pipeline
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_acquisition.py     # Census data downloader
│   ├── data_processing.py      # FIPS alignment, wide-to-long reshaping, merging
│   ├── feature_engineering.py  # Lag creation, CAGR, demographic physics features
│   ├── model.py                # LightGBM training, CV evaluation, and forecasting
│   └── visualize.py            # Chart and choropleth map generation
├── data/                       # Ignored in git (auto-generated)
│   ├── raw/                    # Downloaded bulk CSV/TXT files
│   ├── processed/              # Cleaned and merged panel datasets
│   └── predictions/            # Output forecasts (2025-2027)
└── output/                     # Generated charts and model metrics
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/us-county-population-model.git
   cd us-county-population-model
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

Run the entire end-to-end pipeline with a single command:

```bash
python main.py
```

**The pipeline will automatically:**
1. Download 15 years of population and permit data.
2. Clean, reshape, and engineer features.
3. Train the LightGBM model and run cross-validation.
4. Forecast population for 2025, 2026, and 2027.
5. Generate performance metrics and visualizations in the `output/` folder.

---

## 📈 Results & Visualizations

After running `main.py`, check the `output/` directory for generated visualizations:

* `feature_importance.png`: Shows which features the model relies on most (typically momentum and lagged permits).
* `actual_vs_predicted.png`: Scatter plot validating model accuracy across small, medium, and large counties.
* `county_timeseries.png`: Historical trajectories and future forecasts for selected counties.
* `growth_choropleth.html`: Interactive map of predicted growth rates across the United States.

---

## 📡 Data Sources

This project relies on open public data provided by the U.S. Census Bureau. No API keys are required as the pipeline pulls from bulk data servers.
* **Population**: [Census Population Estimates Program (PEP)](https://www.census.gov/programs-surveys/popest.html)
* **Permits**: [Census Building Permits Survey (BPS)](https://www.census.gov/construction/bps/)