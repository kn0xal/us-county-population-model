"""
Visualization Module
Generates charts and analysis outputs for the population prediction model.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Style configuration
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#ccc",
    "ytick.color": "#ccc",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Color palette
COLORS = {
    "primary": "#e94560",
    "secondary": "#0f3460",
    "accent": "#533483",
    "highlight": "#16c79a",
    "warning": "#f5a623",
    "info": "#4fc3f7",
    "bg_dark": "#1a1a2e",
    "bg_mid": "#16213e",
    "text": "#eee",
    "grid": "#2a2a4a",
}

TIER_COLORS = {"small": "#4fc3f7", "medium": "#f5a623", "large": "#e94560"}


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20) -> str:
    """Plot top N feature importances."""
    print("  📊 Feature Importance plot...")

    fi = feature_importance.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.magma(np.linspace(0.3, 0.8, len(fi)))

    ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_xlabel("Importance (split count)")
    ax.set_title(f"Top {top_n} Feature Importances — LightGBM", fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_actual_vs_predicted(test_df: pd.DataFrame) -> str:
    """Scatter plot of predicted vs actual population for the test set."""
    print("  📊 Actual vs Predicted scatter...")

    if "pred_population" not in test_df.columns or "population" not in test_df.columns:
        print("    ✗ Missing columns for actual vs predicted plot")
        return ""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by population tier
    for tier_name, (lo, hi) in config.SIZE_TIERS.items():
        mask = test_df["population"].between(lo, hi if hi != float("inf") else 1e12)
        subset = test_df[mask]
        ax.scatter(
            subset["population"],
            subset["pred_population"],
            alpha=0.4,
            s=10,
            label=f"{tier_name.capitalize()} (n={len(subset)})",
            color=TIER_COLORS.get(tier_name, COLORS["primary"]),
        )

    # Perfect prediction line
    lim_max = max(test_df["population"].max(), test_df["pred_population"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "--", color=COLORS["highlight"], alpha=0.8, label="Perfect prediction")

    ax.set_xlabel("Actual Population")
    ax.set_ylabel("Predicted Population")
    ax.set_title("Actual vs Predicted County Population (Test Set)", fontweight="bold", pad=15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(test_df["population"], test_df["pred_population"])
    ax.text(
        0.95, 0.05, f"R² = {r2:.4f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["accent"], alpha=0.8),
    )

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "actual_vs_predicted.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_growing_shrinking(forecasts: pd.DataFrame, n: int = 15) -> str:
    """Bar chart of counties with highest predicted growth and decline."""
    print("  📊 Top growing/shrinking counties plot...")

    # Use 2025 predictions (first year forecast)
    yr = forecasts[forecasts["year"] == config.FORECAST_YEARS[0]].copy()
    if yr.empty:
        yr = forecasts.copy()

    yr = yr.sort_values("pred_growth_rate")

    top_growing = yr.tail(n).copy()
    top_shrinking = yr.head(n).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Growing
    labels_g = (top_growing["county_name"] + ", " + top_growing["state_name"].str[:2]).values
    rates_g = (top_growing["pred_growth_rate"] * 100).values
    colors_g = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_growing)))
    ax1.barh(range(len(top_growing)), rates_g, color=colors_g, edgecolor="#333")
    ax1.set_yticks(range(len(top_growing)))
    ax1.set_yticklabels(labels_g, fontsize=9)
    ax1.set_xlabel("Predicted Growth Rate (%)")
    ax1.set_title(f"Top {n} Fastest Growing Counties ({config.FORECAST_YEARS[0]})", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Shrinking
    labels_s = (top_shrinking["county_name"] + ", " + top_shrinking["state_name"].str[:2]).values
    rates_s = (top_shrinking["pred_growth_rate"] * 100).values
    colors_s = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_shrinking)))
    ax2.barh(range(len(top_shrinking)), rates_s, color=colors_s[::-1], edgecolor="#333")
    ax2.set_yticks(range(len(top_shrinking)))
    ax2.set_yticklabels(labels_s, fontsize=9)
    ax2.set_xlabel("Predicted Growth Rate (%)")
    ax2.set_title(f"Top {n} Fastest Shrinking Counties ({config.FORECAST_YEARS[0]})", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("", y=1.0)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "top_growing_shrinking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_county_timeseries(
    df: pd.DataFrame,
    forecasts: pd.DataFrame,
    fips_list: list = None,
) -> str:
    """Line charts showing historical + predicted population for selected counties."""
    print("  📊 County time series plot...")

    if fips_list is None:
        # Auto-select: pick top 3 by population and top 3 by predicted growth
        top_pop = df.loc[df.groupby("FIPS")["population"].idxmax()].nlargest(3, "population")
        top_growth = forecasts[forecasts["year"] == config.FORECAST_YEARS[0]].nlargest(3, "pred_growth_rate")
        fips_list = list(set(top_pop["FIPS"].tolist() + top_growth["FIPS"].tolist()))[:6]

    n = len(fips_list)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, fips in enumerate(fips_list):
        ax = axes[i]

        # Historical
        hist = df[df["FIPS"] == fips].sort_values("year")
        if not hist.empty:
            ax.plot(
                hist["year"], hist["population"],
                "o-", color=COLORS["info"], markersize=4, linewidth=1.5, label="Historical"
            )

        # Forecast
        fcast = forecasts[forecasts["FIPS"] == fips].sort_values("year")
        if not fcast.empty:
            # Connect historical to forecast if we have history
            if not hist.empty:
                last_hist = hist.iloc[-1]
                connect_years = [last_hist["year"]] + fcast["year"].tolist()
                connect_pops = [last_hist["population"]] + fcast["pred_population"].tolist()
            else:
                connect_years = fcast["year"].tolist()
                connect_pops = fcast["pred_population"].tolist()
            ax.plot(
                connect_years, connect_pops,
                "s--", color=COLORS["primary"], markersize=5, linewidth=1.5, label="Forecast"
            )

        # Labels — prefer historical, fall back to forecast
        if not hist.empty:
            county_name = hist["county_name"].iloc[0]
            state_name = hist["state_name"].iloc[0]
        elif not fcast.empty and "county_name" in fcast.columns:
            county_name = fcast["county_name"].iloc[0]
            state_name = fcast["state_name"].iloc[0] if "state_name" in fcast.columns else ""
        else:
            county_name = fips
            state_name = ""
        ax.set_title(f"{county_name}\n{state_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "county_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_distribution(test_df: pd.DataFrame) -> str:
    """Histogram of prediction errors by county size tier."""
    print("  📊 Error distribution plot...")

    if "pred_growth_rate" not in test_df.columns:
        return ""

    test_df = test_df.copy()
    test_df["error_pct"] = (test_df["pred_growth_rate"] - test_df["target_growth_rate"]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (tier_name, (lo, hi)) in enumerate(config.SIZE_TIERS.items()):
        ax = axes[idx]
        mask = test_df["population"].between(lo, hi if hi != float("inf") else 1e12)
        errors = test_df.loc[mask, "error_pct"].dropna()

        if len(errors) > 0:
            ax.hist(
                errors, bins=50, color=TIER_COLORS.get(tier_name, COLORS["primary"]),
                alpha=0.7, edgecolor="#333"
            )
            ax.axvline(0, color=COLORS["highlight"], linestyle="--", alpha=0.8)
            ax.axvline(errors.mean(), color=COLORS["warning"], linestyle="-", alpha=0.8, label=f"Mean: {errors.mean():.3f}%")

        hi_str = f"{hi/1000:.0f}K" if hi != float("inf") else "∞"
        ax.set_title(f"{tier_name.capitalize()} ({lo/1000:.0f}K – {hi_str})", fontweight="bold")
        ax.set_xlabel("Prediction Error (%)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Prediction Error Distribution by County Size", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "error_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_growth_choropleth(forecasts: pd.DataFrame) -> str:
    """
    Create a choropleth map of predicted growth rates using plotly.
    Uses county-level FIPS codes to map to geographic regions.
    """
    print("  📊 Growth rate choropleth map...")

    try:
        import plotly.express as px
        from urllib.request import urlopen
        import json as json_mod
    except ImportError:
        print("    ✗ plotly not available, skipping choropleth")
        return ""

    # Use first forecast year
    yr = forecasts[forecasts["year"] == config.FORECAST_YEARS[0]].copy()
    if yr.empty:
        yr = forecasts.copy()

    yr["growth_pct"] = yr["pred_growth_rate"] * 100

    # Load GeoJSON for US counties
    try:
        geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urlopen(geojson_url) as response:
            counties_geojson = json_mod.load(response)
    except Exception as e:
        print(f"    ✗ Could not load county GeoJSON: {e}")
        # Create a simple bar chart instead
        return _plot_growth_bar_chart(yr)

    # Clip extreme values for better color scale
    vmin, vmax = yr["growth_pct"].quantile(0.02), yr["growth_pct"].quantile(0.98)

    fig = px.choropleth(
        yr,
        geojson=counties_geojson,
        locations="FIPS",
        color="growth_pct",
        color_continuous_scale="RdYlGn",
        range_color=[vmin, vmax],
        scope="usa",
        labels={"growth_pct": "Growth Rate (%)"},
        hover_data=["county_name", "state_name", "pred_population"],
        title=f"Predicted County Population Growth Rate ({config.FORECAST_YEARS[0]})",
    )

    fig.update_layout(
        geo=dict(
            bgcolor="#1a1a2e",
            lakecolor="#16213e",
            landcolor="#16213e",
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )

    path = os.path.join(config.OUTPUT_DIR, "growth_choropleth.html")
    fig.write_html(path)
    print(f"    ✓ Interactive choropleth: {path}")

    # Also save as static PNG if kaleido is available
    try:
        png_path = os.path.join(config.OUTPUT_DIR, "growth_choropleth.png")
        fig.write_image(png_path, width=1400, height=800, scale=2)
        print(f"    ✓ Static choropleth: {png_path}")
        return png_path
    except Exception:
        return path


def _plot_growth_bar_chart(yr: pd.DataFrame) -> str:
    """Fallback: state-level average growth bar chart if choropleth fails."""
    state_avg = yr.groupby("state_name")["growth_pct"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(state_avg)))
    ax.barh(state_avg.index, state_avg.values, color=colors, edgecolor="#333", linewidth=0.3)
    ax.set_xlabel("Avg Predicted Growth Rate (%)")
    ax.set_title(f"State-Level Average Predicted Growth ({config.FORECAST_YEARS[0]})", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0, color=COLORS["text"], alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "state_growth_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cv_fold_performance(cv_results: dict) -> str:
    """Plot CV fold performance metrics."""
    print("  📊 CV fold performance plot...")

    fold_metrics = cv_results.get("fold_metrics", [])
    if not fold_metrics:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    folds = [f"Fold {m['fold']}\n({m['val_year']})" for m in fold_metrics]
    mapes = [m.get("MAPE", 0) for m in fold_metrics]
    r2s = [m.get("R2", 0) for m in fold_metrics]

    # MAPE by fold
    bars1 = ax1.bar(folds, mapes, color=COLORS["info"], edgecolor="#333", alpha=0.8)
    ax1.axhline(np.mean(mapes), color=COLORS["primary"], linestyle="--", label=f"Avg: {np.mean(mapes):.3f}%")
    ax1.set_ylabel("MAPE (%)")
    ax1.set_title("MAPE by CV Fold", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # R² by fold
    bars2 = ax2.bar(folds, r2s, color=COLORS["highlight"], edgecolor="#333", alpha=0.8)
    ax2.axhline(np.mean(r2s), color=COLORS["primary"], linestyle="--", label=f"Avg: {np.mean(r2s):.4f}")
    ax2.set_ylabel("R²")
    ax2.set_title("R² by CV Fold", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Validation Performance", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "cv_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_visualization(results: dict) -> list:
    """Generate all visualization outputs."""
    print("\n" + "═" * 70)
    print("  US COUNTY POPULATION MODEL — VISUALIZATION")
    print("═" * 70)

    output_paths = []

    # 1. Feature importance
    if "feature_importance" in results:
        p = plot_feature_importance(results["feature_importance"])
        output_paths.append(p)

    # 2. Actual vs Predicted
    if "test_predictions" in results:
        p = plot_actual_vs_predicted(results["test_predictions"])
        if p:
            output_paths.append(p)

    # 3. Top growing / shrinking
    if "forecasts" in results:
        p = plot_top_growing_shrinking(results["forecasts"])
        output_paths.append(p)

    # 4. County time series (need original panel data)
    features_path = os.path.join(config.PROCESSED_DIR, config.PANEL_FILENAME)
    if os.path.exists(features_path) and "forecasts" in results:
        panel = pd.read_csv(features_path)
        p = plot_county_timeseries(panel, results["forecasts"])
        output_paths.append(p)

    # 5. Error distribution
    if "test_predictions" in results:
        p = plot_error_distribution(results["test_predictions"])
        if p:
            output_paths.append(p)

    # 6. Choropleth
    if "forecasts" in results:
        p = plot_growth_choropleth(results["forecasts"])
        if p:
            output_paths.append(p)

    # 7. CV performance
    if "cv_results" in results:
        p = plot_cv_fold_performance(results["cv_results"])
        if p:
            output_paths.append(p)

    print(f"\n  ✓ Generated {len(output_paths)} visualizations in: {config.OUTPUT_DIR}")
    return output_paths


if __name__ == "__main__":
    # Load results if available
    print("Run via main.py for full visualization pipeline.")
