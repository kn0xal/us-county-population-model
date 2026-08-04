# 📈 US County Population Prediction Model

A machine learning project that predicts US county-level population growth by combining **15 years of Census population estimates** (2010–2024) with **building permits data** as a leading indicator.

## 🚀 Overview

Building permits are a strong leading indicator of population change:
`Permit issued → Construction (6-18 months) → Occupancy → Census count`

This project uses a **Pooled Panel LightGBM Regressor** trained on ~47,000 county-year records to predict future population growth rates for all ~3,140 US counties.

### Key Features
* **Automated Data Pipeline**: Downloads and merges Census Population Estimates and Building Permits Survey (BPS) bulk datasets.
* **Advanced Feature Engineering**: Creates lag features (1-3 years), compound annual growth rates (CAGR), rolling averages, permit density, and "Housing Unit Method" implied growth.
* **Robust Evaluation**: Uses expanding-window time-series cross-validation to prevent data leakage.
* **Recursive Forecasting**: Generates multi-step autoregressive predictions for 2025–2027.

## 📊 Model Performance

Evaluated on the 2023-2024 test set:
* **MAPE (Mean Absolute Percentage Error)**: `0.23%` (highly accurate)
* **MALPE (Mean Algebraic Percentage Error)**: `-0.03%` (virtually no bias)
* **R²**: `0.849` (explains 85% of variance in growth rates)

## 💻 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt