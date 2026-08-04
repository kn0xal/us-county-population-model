"""
Feature Engineering Module
Creates ML features from the panel dataset including lagged permits,
growth rates, density metrics, and interaction features.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_population_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create population-based features within each county group.
    """
    print("  Creating population features...")

    # Sort for correct lag computation
    df = df.sort_values(["FIPS", "year"]).copy()

    grouped = df.groupby("FIPS")

    # Annual growth rate
    df["pop_growth_rate"] = grouped["population"].pct_change()

    # Lagged population
    for lag in config.LAG_PERIODS:
        df[f"pop_lag{lag}"] = grouped["population"].shift(lag)

    # Lagged growth rates
    for lag in config.LAG_PERIODS[:2]:  # lag 1 and 2
        df[f"pop_growth_lag{lag}"] = grouped["pop_growth_rate"].shift(lag)

    # Compound annual growth rates
    for window in config.ROLLING_WINDOWS:
        col_name = f"pop_cagr_{window}yr"
        pop_lagged = grouped["population"].shift(window)
        df[col_name] = (df["population"] / pop_lagged) ** (1 / window) - 1

    # Population log (for scale normalization)
    df["log_population"] = np.log1p(df["population"])

    return df


def create_permit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create building permit features — the key leading indicators.
    Permits issued today → construction → occupancy → Census count 1-3 years later.
    """
    print("  Creating permit features...")

    grouped = df.groupby("FIPS")

    # Lagged permits (the most important features)
    for lag in config.LAG_PERIODS:
        df[f"permits_lag{lag}"] = grouped["total_permits"].shift(lag)
        df[f"sf_permits_lag{lag}"] = grouped["sf_permits"].shift(lag)
        df[f"mf_permits_lag{lag}"] = grouped["mf_permits"].shift(lag)

    # Rolling sums (housing pipeline)
    for window in config.ROLLING_WINDOWS:
        df[f"permits_rolling_{window}yr"] = (
            grouped["total_permits"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )
        # Shift by 1 so we don't leak current-year data
        df[f"permits_rolling_{window}yr"] = grouped[f"permits_rolling_{window}yr"].shift(1)

    # Permit density (permits per 1,000 residents)
    pop_lag1 = grouped["population"].shift(1)
    df["permit_density"] = (df["total_permits"] / pop_lag1.replace(0, np.nan)) * 1000
    df["permit_density_lag1"] = grouped["permit_density"].shift(1)

    # Year-over-year permit growth
    df["permit_growth_yoy"] = grouped["total_permits"].pct_change()
    df["permit_growth_lag1"] = grouped["permit_growth_yoy"].shift(1)

    # Permit surge index (current / 5-year average)
    rolling_5yr_mean = grouped["total_permits"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()
    )
    df["permit_surge_index"] = df["total_permits"] / rolling_5yr_mean.replace(0, np.nan)
    df["permit_surge_lag1"] = grouped["permit_surge_index"].shift(1)

    # Single-family to multi-family ratio
    df["sf_mf_ratio"] = df["sf_permits"] / df["mf_permits"].replace(0, np.nan)
    df["sf_mf_ratio"] = df["sf_mf_ratio"].clip(upper=100)  # Cap outliers
    df["sf_mf_ratio_lag1"] = grouped["sf_mf_ratio"].shift(1)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction and 'physics-based' features that combine
    population and permit data for domain-informed predictions.
    """
    print("  Creating interaction features...")

    # Implied population addition from permits
    # permits_lag1 * avg_household_size * occupancy_rate
    df["implied_pop_add"] = (
        df["permits_lag1"]
        * config.AVG_PERSONS_PER_HOUSEHOLD
        * config.OCCUPANCY_RATE
    )

    # Implied growth rate from permits alone
    df["implied_growth_rate"] = df["implied_pop_add"] / df["pop_lag1"].replace(0, np.nan)

    # Discrepancy between actual and implied growth
    # Positive = actual growth exceeded what permits implied (migration, etc.)
    # Negative = actual growth was less (demolitions, out-migration, etc.)
    df["permit_growth_discrepancy"] = df["pop_growth_rate"] - df["implied_growth_rate"]
    df["discrepancy_lag1"] = df.groupby("FIPS")["permit_growth_discrepancy"].shift(1)

    # Permit momentum: permits_lag1 / permits_lag2 (acceleration of building)
    df["permit_momentum"] = df["permits_lag1"] / df["permits_lag2"].replace(0, np.nan)
    df["permit_momentum"] = df["permit_momentum"].clip(-10, 10)

    return df


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define the prediction target: population growth rate for year t.
    The model predicts growth rate, then population is reconstructed as:
    Pop_t = Pop_{t-1} * (1 + predicted_growth_rate)
    """
    print("  Defining target variable...")

    # Target is the current year's growth rate
    df["target_growth_rate"] = df["pop_growth_rate"]

    return df


def get_feature_columns() -> list:
    """Return the list of feature column names used for modeling."""
    features = []

    # Population features
    features += ["log_population"]
    features += [f"pop_lag{lag}" for lag in config.LAG_PERIODS]
    features += [f"pop_growth_lag{lag}" for lag in config.LAG_PERIODS[:2]]
    features += [f"pop_cagr_{w}yr" for w in config.ROLLING_WINDOWS]

    # Permit features (lagged)
    features += [f"permits_lag{lag}" for lag in config.LAG_PERIODS]
    features += [f"sf_permits_lag{lag}" for lag in config.LAG_PERIODS]
    features += [f"mf_permits_lag{lag}" for lag in config.LAG_PERIODS]
    features += [f"permits_rolling_{w}yr" for w in config.ROLLING_WINDOWS]
    features += ["permit_density_lag1", "permit_growth_lag1", "permit_surge_lag1"]
    features += ["sf_mf_ratio_lag1"]

    # Interaction features
    features += [
        "implied_pop_add",
        "implied_growth_rate",
        "discrepancy_lag1",
        "permit_momentum",
    ]

    return features


def run_feature_engineering(panel: pd.DataFrame = None) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    print("\n" + "═" * 70)
    print("  US COUNTY POPULATION MODEL — FEATURE ENGINEERING")
    print("═" * 70)

    if panel is None:
        panel_path = os.path.join(config.PROCESSED_DIR, config.PANEL_FILENAME)
        panel = pd.read_csv(panel_path)
        print(f"  Loaded panel from: {panel_path}")

    df = panel.copy()
    df = create_population_features(df)
    df = create_permit_features(df)
    df = create_interaction_features(df)
    df = define_target(df)

    # Summary of features
    feature_cols = get_feature_columns()
    print(f"\n  ✓ Created {len(feature_cols)} features")
    print(f"  ✓ Target: target_growth_rate")

    # Filter to usable years (after lags consume the first 3 years)
    usable = df[df["year"] >= config.FIRST_USABLE_YEAR].copy()
    print(f"  ✓ Usable rows (year >= {config.FIRST_USABLE_YEAR}): {len(usable)}")

    # Report missing values in feature columns
    missing = usable[feature_cols].isnull().sum()
    missing_pct = (missing / len(usable) * 100).round(1)
    has_missing = missing[missing > 0]
    if len(has_missing) > 0:
        print(f"  ⚠ Features with missing values:")
        for col, count in has_missing.items():
            print(f"      {col}: {count} ({missing_pct[col]}%)")

    # Save featured dataset
    output_path = os.path.join(config.PROCESSED_DIR, config.FEATURES_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"\n  💾 Saved featured dataset to: {output_path}")

    return df


if __name__ == "__main__":
    run_feature_engineering()
