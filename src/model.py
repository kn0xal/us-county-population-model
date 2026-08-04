"""
Model Training, Evaluation, and Forecasting Module
Trains a pooled panel LightGBM regressor, evaluates via expanding window CV,
and generates recursive multi-step forecasts for 2025-2027.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import get_feature_columns


TARGET = "target_growth_rate"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, population: np.ndarray = None) -> dict:
    """Compute evaluation metrics for population growth rate predictions."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # Percentage errors (relative to actual growth rate — can be noisy for near-zero growth)
    # Instead, use absolute population error as % of actual population
    metrics = {
        "MAE_growth_rate": float(np.mean(abs_errors)),
        "RMSE_growth_rate": float(np.sqrt(np.mean(errors ** 2))),
        "MedAE_growth_rate": float(np.median(abs_errors)),
        "Mean_Error_growth_rate": float(np.mean(errors)),  # Bias
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }

    if population is not None:
        # Convert growth rate errors to population errors
        pop_pred = population * (1 + y_pred)
        pop_actual = population * (1 + y_true)
        pop_errors = pop_pred - pop_actual
        abs_pop_errors = np.abs(pop_errors)

        # MAPE (population-level)
        metrics["MAPE"] = float(np.mean(abs_pop_errors / np.maximum(pop_actual, 1)) * 100)
        # MALPE (signed — bias detection)
        metrics["MALPE"] = float(np.mean(pop_errors / np.maximum(pop_actual, 1)) * 100)
        # MedAPE
        metrics["MedAPE"] = float(np.median(abs_pop_errors / np.maximum(pop_actual, 1)) * 100)
        # MAE in population units
        metrics["MAE_population"] = float(np.mean(abs_pop_errors))

    return metrics


def tiered_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    population: np.ndarray,
) -> dict:
    """Compute metrics by county population size tier."""
    tiers = {}
    for tier_name, (lo, hi) in config.SIZE_TIERS.items():
        mask = (population >= lo) & (population < hi)
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask], population[mask])
        m["count"] = int(mask.sum())
        tiers[tier_name] = m
    return tiers


def cross_validate(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Expanding-window time-series cross-validation.
    Returns CV metrics and OOF predictions.
    """
    print("\n── Cross-Validation (Expanding Window) ──")

    all_metrics = []
    oof_preds = []

    for fold_idx, fold in enumerate(config.CV_FOLDS):
        train_start, train_end = fold["train"]
        val_start, val_end = fold["val"]

        train_mask = df["year"].between(train_start, train_end)
        val_mask = df["year"].between(val_start, val_end)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, TARGET]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, TARGET]
        pop_val = df.loc[val_mask, "population"].values

        # Drop rows with NaN target
        train_valid = y_train.notna() & X_train.notna().all(axis=1)
        val_valid = y_val.notna() & X_val.notna().all(axis=1)

        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_val, y_val = X_val[val_valid], y_val[val_valid]
        pop_val = pop_val[val_valid.values]

        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],  # Suppress logging
        )

        preds = model.predict(X_val)
        fold_metrics = compute_metrics(y_val.values, preds, pop_val)
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["train_years"] = f"{train_start}-{train_end}"
        fold_metrics["val_year"] = f"{val_start}"
        fold_metrics["train_size"] = len(X_train)
        fold_metrics["val_size"] = len(X_val)
        all_metrics.append(fold_metrics)

        # Store OOF predictions
        oof_df = df.loc[val_mask].copy()
        oof_df = oof_df[val_valid]
        oof_df["pred_growth_rate"] = preds
        oof_preds.append(oof_df)

        mape = fold_metrics.get("MAPE", float("nan"))
        r2 = fold_metrics.get("R2", float("nan"))
        print(
            f"  Fold {fold_idx+1}: Train [{train_start}-{train_end}] → "
            f"Val [{val_start}] | MAPE={mape:.3f}% | R²={r2:.4f} | "
            f"n_train={len(X_train)}, n_val={len(X_val)}"
        )

    # Average metrics
    avg = {}
    metric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], float)]
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        avg[k] = np.mean(vals) if vals else float("nan")

    print(f"\n  ── CV Average ──")
    print(f"  MAPE:  {avg.get('MAPE', float('nan')):.3f}%")
    print(f"  MALPE: {avg.get('MALPE', float('nan')):.4f}%")
    print(f"  MedAPE: {avg.get('MedAPE', float('nan')):.3f}%")
    print(f"  R²:    {avg.get('R2', float('nan')):.4f}")

    oof = pd.concat(oof_preds, ignore_index=True) if oof_preds else pd.DataFrame()

    return {
        "fold_metrics": all_metrics,
        "avg_metrics": avg,
        "oof_predictions": oof,
    }


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list,
    train_years: tuple = None,
    test_years: tuple = None,
) -> tuple:
    """
    Train the final model on train years, evaluate on test years.
    Returns (model, test_metrics, test_predictions_df, feature_importance_df).
    """
    if train_years is None:
        train_years = config.FINAL_TRAIN_YEARS
    if test_years is None:
        test_years = config.TEST_YEARS

    print(f"\n── Training Final Model ──")
    print(f"  Train: {train_years[0]}-{train_years[1]}")
    print(f"  Test:  {test_years[0]}-{test_years[1]}")

    train_mask = df["year"].between(*train_years)
    test_mask = df["year"].between(*test_years)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, TARGET]
    pop_test = df.loc[test_mask, "population"].values

    # Clean NaNs
    train_valid = y_train.notna() & X_train.notna().all(axis=1)
    test_valid = y_test.notna() & X_test.notna().all(axis=1)
    X_train, y_train = X_train[train_valid], y_train[train_valid]
    X_test, y_test = X_test[test_valid], y_test[test_valid]
    pop_test = pop_test[test_valid.values]

    print(f"  Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Predictions
    preds = model.predict(X_test)
    test_metrics = compute_metrics(y_test.values, preds, pop_test)
    tier_metrics = tiered_metrics(y_test.values, preds, pop_test)

    print(f"\n  ── Test Set Results ──")
    print(f"  MAPE:  {test_metrics.get('MAPE', float('nan')):.3f}%")
    print(f"  MALPE: {test_metrics.get('MALPE', float('nan')):.4f}%")
    print(f"  MedAPE: {test_metrics.get('MedAPE', float('nan')):.3f}%")
    print(f"  R²:    {test_metrics.get('R2', float('nan')):.4f}")
    print(f"  MAE (population): {test_metrics.get('MAE_population', float('nan')):.0f}")

    print(f"\n  ── Tiered Metrics ──")
    for tier_name, tm in tier_metrics.items():
        lo, hi = config.SIZE_TIERS[tier_name]
        hi_str = f"{hi:,.0f}" if hi != float("inf") else "∞"
        print(
            f"  {tier_name.capitalize():8s} ({lo:>7,} - {hi_str:>10}): "
            f"MAPE={tm.get('MAPE', float('nan')):.3f}%  |  n={tm['count']}"
        )

    # Test predictions DataFrame
    test_df = df.loc[test_mask].copy()
    test_df = test_df[test_valid]
    test_df["pred_growth_rate"] = preds
    test_df["pred_population"] = test_df["pop_lag1"] * (1 + preds)

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, test_metrics, tier_metrics, test_df, fi


def train_full_model(df: pd.DataFrame, feature_cols: list) -> lgb.LGBMRegressor:
    """
    Train on ALL available data for production forecasting.
    """
    print(f"\n── Training Full Model (all data) ──")
    train_years = config.FULL_TRAIN_YEARS

    mask = df["year"].between(*train_years)
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, TARGET]

    valid = y.notna() & X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    print(f"  Training on {len(X)} rows ({train_years[0]}-{train_years[1]})")

    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    model.fit(X, y)

    return model


def recursive_forecast(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Generate recursive multi-step forecasts for 2025-2027.
    Each year's prediction feeds into the next year's features.
    """
    print(f"\n── Recursive Forecasting ({config.FORECAST_YEARS}) ──")

    # Get the latest available data for each county
    latest_year = df["year"].max()
    counties = df[df["year"] == latest_year][["FIPS", "state_name", "county_name"]].drop_duplicates()

    # Build a working copy with historical data
    working = df.copy()
    forecasts = []

    for forecast_year in config.FORECAST_YEARS:
        print(f"\n  Forecasting {forecast_year}...")

        # Build feature row for each county using available data
        # Get the most recent 3 years of data for lag computation
        feature_rows = []

        for _, county_row in counties.iterrows():
            fips = county_row["FIPS"]
            county_data = working[working["FIPS"] == fips].sort_values("year")

            if len(county_data) < 3:
                continue

            latest = county_data.iloc[-1]
            prev1 = county_data.iloc[-2] if len(county_data) >= 2 else latest
            prev2 = county_data.iloc[-3] if len(county_data) >= 3 else prev1

            row = {"FIPS": fips, "year": forecast_year}
            row["state_name"] = county_row["state_name"]
            row["county_name"] = county_row["county_name"]
            row["population"] = latest["population"]  # Will be updated after prediction

            # Population features
            row["log_population"] = np.log1p(latest["population"])
            row["pop_lag1"] = latest["population"]
            row["pop_lag2"] = prev1["population"]
            row["pop_lag3"] = prev2["population"]

            # Growth rate lags
            if latest["population"] > 0 and prev1["population"] > 0:
                row["pop_growth_lag1"] = (latest["population"] - prev1["population"]) / prev1["population"]
            else:
                row["pop_growth_lag1"] = 0
            if prev1["population"] > 0 and prev2["population"] > 0:
                row["pop_growth_lag2"] = (prev1["population"] - prev2["population"]) / prev2["population"]
            else:
                row["pop_growth_lag2"] = 0

            # CAGR features
            for window in config.ROLLING_WINDOWS:
                if len(county_data) >= window:
                    older = county_data.iloc[-(window + 1)]["population"] if len(county_data) > window else county_data.iloc[0]["population"]
                    if older > 0:
                        row[f"pop_cagr_{window}yr"] = (latest["population"] / older) ** (1 / window) - 1
                    else:
                        row[f"pop_cagr_{window}yr"] = 0
                else:
                    row[f"pop_cagr_{window}yr"] = 0

            # Permit features (lagged from historical data)
            row["permits_lag1"] = latest.get("total_permits", 0)
            row["permits_lag2"] = prev1.get("total_permits", 0)
            row["permits_lag3"] = prev2.get("total_permits", 0)

            row["sf_permits_lag1"] = latest.get("sf_permits", 0)
            row["sf_permits_lag2"] = prev1.get("sf_permits", 0)
            row["sf_permits_lag3"] = prev2.get("sf_permits", 0)

            row["mf_permits_lag1"] = latest.get("mf_permits", 0)
            row["mf_permits_lag2"] = prev1.get("mf_permits", 0)
            row["mf_permits_lag3"] = prev2.get("mf_permits", 0)

            # Rolling permit sums
            for window in config.ROLLING_WINDOWS:
                recent_permits = county_data.tail(window)["total_permits"].sum()
                row[f"permits_rolling_{window}yr"] = recent_permits

            # Permit density
            if latest["population"] > 0:
                row["permit_density_lag1"] = (latest.get("total_permits", 0) / latest["population"]) * 1000
            else:
                row["permit_density_lag1"] = 0

            # Permit growth YoY
            if prev1.get("total_permits", 0) > 0:
                row["permit_growth_lag1"] = (
                    latest.get("total_permits", 0) - prev1.get("total_permits", 0)
                ) / prev1.get("total_permits", 0)
            else:
                row["permit_growth_lag1"] = 0

            # Permit surge
            permit_5yr_mean = county_data.tail(5)["total_permits"].mean()
            if permit_5yr_mean > 0:
                row["permit_surge_lag1"] = latest.get("total_permits", 0) / permit_5yr_mean
            else:
                row["permit_surge_lag1"] = 1.0

            # SF/MF ratio
            mf = latest.get("mf_permits", 0)
            sf = latest.get("sf_permits", 0)
            row["sf_mf_ratio_lag1"] = min(sf / max(mf, 1), 100)

            # Interaction features
            row["implied_pop_add"] = (
                row["permits_lag1"] * config.AVG_PERSONS_PER_HOUSEHOLD * config.OCCUPANCY_RATE
            )
            if row["pop_lag1"] > 0:
                row["implied_growth_rate"] = row["implied_pop_add"] / row["pop_lag1"]
            else:
                row["implied_growth_rate"] = 0

            row["discrepancy_lag1"] = row["pop_growth_lag1"] - row["implied_growth_rate"]

            if row["permits_lag2"] > 0:
                row["permit_momentum"] = row["permits_lag1"] / row["permits_lag2"]
            else:
                row["permit_momentum"] = 1.0
            row["permit_momentum"] = max(min(row["permit_momentum"], 10), -10)

            feature_rows.append(row)

        forecast_df = pd.DataFrame(feature_rows)

        # Predict
        X_forecast = forecast_df[feature_cols].fillna(0)
        predicted_growth = model.predict(X_forecast)
        forecast_df["pred_growth_rate"] = predicted_growth
        forecast_df["pred_population"] = (
            forecast_df["pop_lag1"] * (1 + predicted_growth)
        ).astype(int)

        forecasts.append(forecast_df)

        # Add predictions back to working dataset for next year's lag computation
        new_rows = forecast_df[["FIPS", "state_name", "county_name"]].copy()
        new_rows["year"] = forecast_year
        new_rows["population"] = forecast_df["pred_population"]
        new_rows["total_permits"] = forecast_df["permits_lag1"]  # Assume flat permits for future
        new_rows["sf_permits"] = forecast_df["sf_permits_lag1"]
        new_rows["mf_permits"] = forecast_df["mf_permits_lag1"]
        new_rows["valuation"] = 0

        working = pd.concat([working, new_rows], ignore_index=True)

        print(f"  ✓ {forecast_year}: {len(forecast_df)} counties predicted")
        print(f"    Mean predicted growth: {predicted_growth.mean()*100:.3f}%")
        print(f"    Median predicted growth: {np.median(predicted_growth)*100:.3f}%")

    all_forecasts = pd.concat(forecasts, ignore_index=True)

    # Save forecasts
    output_cols = [
        "FIPS", "state_name", "county_name", "year",
        "pop_lag1", "pred_growth_rate", "pred_population",
    ]
    output = all_forecasts[output_cols].copy()
    output = output.rename(columns={"pop_lag1": "base_population"})
    output_path = os.path.join(config.PREDICTIONS_DIR, config.PREDICTIONS_FILENAME)
    output.to_csv(output_path, index=False)
    print(f"\n  💾 Saved predictions to: {output_path}")

    return all_forecasts


def run_model_pipeline(df: pd.DataFrame = None) -> dict:
    """Run the full model training, evaluation, and forecasting pipeline."""
    print("\n" + "═" * 70)
    print("  US COUNTY POPULATION MODEL — MODEL TRAINING & EVALUATION")
    print("═" * 70)

    if df is None:
        features_path = os.path.join(config.PROCESSED_DIR, config.FEATURES_FILENAME)
        df = pd.read_csv(features_path)
        print(f"  Loaded features from: {features_path}")

    feature_cols = get_feature_columns()

    # Filter to usable years
    usable = df[df["year"] >= config.FIRST_USABLE_YEAR].copy()
    print(f"  Usable data: {len(usable)} rows ({usable['year'].min()}-{usable['year'].max()})")

    # Cross-validation
    cv_results = cross_validate(usable, feature_cols)

    # Final model (train + test split)
    model, test_metrics, tier_metrics, test_df, feature_importance = train_final_model(
        usable, feature_cols
    )

    # Full model for forecasting
    full_model = train_full_model(usable, feature_cols)

    # Recursive forecast
    forecast_df = recursive_forecast(full_model, df, feature_cols)

    # Save model
    model_path = os.path.join(config.OUTPUT_DIR, "model_final.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(full_model, f)
    print(f"\n  💾 Saved model to: {model_path}")

    # Save metrics
    metrics_output = {
        "cv_avg": cv_results["avg_metrics"],
        "test": test_metrics,
        "test_tiered": tier_metrics,
    }
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2, default=str)
    print(f"  💾 Saved metrics to: {metrics_path}")

    return {
        "model": full_model,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "tier_metrics": tier_metrics,
        "test_predictions": test_df,
        "feature_importance": feature_importance,
        "forecasts": forecast_df,
    }


if __name__ == "__main__":
    run_model_pipeline()
