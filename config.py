"""
Configuration and constants for the US County Population Prediction Model.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_POP_DIR = os.path.join(RAW_DIR, "population")
RAW_PERMITS_DIR = os.path.join(RAW_DIR, "permits")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Ensure directories exist
for d in [RAW_POP_DIR, RAW_PERMITS_DIR, PROCESSED_DIR, PREDICTIONS_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Population Data URLs ───────────────────────────────────────────────────────
POPULATION_URLS = {
    "2010-2020": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv",
    "2020-2024": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv",
}

# ─── Building Permits Data URLs ─────────────────────────────────────────────────
# Pattern: https://www2.census.gov/econ/bps/County/co{YYYY}a.txt
PERMITS_BASE_URL = "https://www2.census.gov/econ/bps/County/"
PERMITS_YEARS = list(range(2010, 2025))  # 2010 through 2024

# ─── Data Processing Constants ──────────────────────────────────────────────────
# FIPS codes for US states (excluding territories > 56)
MAX_STATE_FIPS = 56

# SUMLEV code for county-level records
COUNTY_SUMLEV = 50

# ─── Feature Engineering Constants ──────────────────────────────────────────────
# Average household size (Census Bureau ACS estimate)
AVG_PERSONS_PER_HOUSEHOLD = 2.53

# Average occupancy rate for new housing
OCCUPANCY_RATE = 0.92

# Lag periods for features
LAG_PERIODS = [1, 2, 3]

# Rolling window sizes
ROLLING_WINDOWS = [3, 5]

# ─── Model Configuration ────────────────────────────────────────────────────────
# LightGBM hyperparameters
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# Years consumed by lag creation (first 3 years are used as lookback)
FIRST_USABLE_YEAR = 2013

# Cross-validation folds (expanding window)
CV_FOLDS = [
    {"train": (2013, 2018), "val": (2019, 2019)},
    {"train": (2013, 2019), "val": (2020, 2020)},
    {"train": (2013, 2020), "val": (2021, 2021)},
    {"train": (2013, 2021), "val": (2022, 2022)},
]

# Final train/test split
FINAL_TRAIN_YEARS = (2013, 2022)
TEST_YEARS = (2023, 2024)

# Full training for forecasting (all available data)
FULL_TRAIN_YEARS = (2013, 2024)

# Forecast horizon
FORECAST_YEARS = [2025, 2026, 2027]

# ─── County Size Tiers (for tiered evaluation) ─────────────────────────────────
SIZE_TIERS = {
    "small": (0, 25_000),
    "medium": (25_000, 100_000),
    "large": (100_000, float("inf")),
}

# ─── Panel Dataset Filename ─────────────────────────────────────────────────────
PANEL_FILENAME = "county_panel.csv"
FEATURES_FILENAME = "county_features.csv"
PREDICTIONS_FILENAME = "county_predictions_2025_2027.csv"
