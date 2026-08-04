"""
US County Population Prediction Model
End-to-end pipeline runner.

Predicts county-level population (2025-2027) using:
- Census Population Estimates (2010-2024)
- Building Permits Survey data (2010-2024)
- LightGBM panel regression with lagged permit features
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    start_time = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  US COUNTY POPULATION PREDICTION MODEL                           ║")
    print("║  Census Population + Building Permits → LightGBM Forecast        ║")
    print("╚" + "═" * 68 + "╝")

    # ── Phase 1: Data Acquisition ────────────────────────────────────────────
    print("\n\n" + "▓" * 70)
    print("  PHASE 1/7: DATA ACQUISITION")
    print("▓" * 70)

    from src.data_acquisition import run_acquisition
    pop_files, permit_files = run_acquisition()

    # ── Phase 2-3: Data Processing ───────────────────────────────────────────
    print("\n\n" + "▓" * 70)
    print("  PHASE 2-3/7: DATA PROCESSING & MERGING")
    print("▓" * 70)

    from src.data_processing import run_processing
    panel = run_processing()

    # ── Phase 4: Feature Engineering ─────────────────────────────────────────
    print("\n\n" + "▓" * 70)
    print("  PHASE 4/7: FEATURE ENGINEERING")
    print("▓" * 70)

    from src.feature_engineering import run_feature_engineering
    featured_df = run_feature_engineering(panel)

    # ── Phase 5-6: Model Training, Evaluation & Forecasting ──────────────────
    print("\n\n" + "▓" * 70)
    print("  PHASE 5-6/7: MODEL TRAINING, EVALUATION & FORECASTING")
    print("▓" * 70)

    from src.model import run_model_pipeline
    results = run_model_pipeline(featured_df)

    # ── Phase 7: Visualization ───────────────────────────────────────────────
    print("\n\n" + "▓" * 70)
    print("  PHASE 7/7: VISUALIZATION")
    print("▓" * 70)

    from src.visualize import run_visualization
    viz_paths = run_visualization(results)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n\n" + "╔" + "═" * 68 + "╗")
    print("║  PIPELINE COMPLETE                                               ║")
    print("╚" + "═" * 68 + "╝")

    test_metrics = results.get("test_metrics", {})
    print(f"""
  ⏱  Elapsed time:     {elapsed:.1f}s
  📊 Panel size:        {len(panel)} rows
  🧪 Test MAPE:         {test_metrics.get('MAPE', 'N/A')}%
  🧪 Test MALPE:        {test_metrics.get('MALPE', 'N/A')}%
  🧪 Test R²:           {test_metrics.get('R2', 'N/A')}
  🔮 Forecast years:    {config.FORECAST_YEARS}
  📈 Visualizations:    {len(viz_paths)} charts generated
  💾 Output directory:  {config.OUTPUT_DIR}
  💾 Predictions:       {os.path.join(config.PREDICTIONS_DIR, config.PREDICTIONS_FILENAME)}
""")

    # Show top predicted growth counties
    forecasts = results.get("forecasts")
    if forecasts is not None:
        yr1 = forecasts[forecasts["year"] == config.FORECAST_YEARS[0]]
        if not yr1.empty:
            print("  🏆 Top 10 Fastest Growing Counties (predicted {}):\n".format(config.FORECAST_YEARS[0]))
            top = yr1.nlargest(10, "pred_growth_rate")[
                ["FIPS", "county_name", "state_name", "pred_growth_rate", "pred_population"]
            ]
            top["pred_growth_rate"] = (top["pred_growth_rate"] * 100).round(2).astype(str) + "%"
            top["pred_population"] = top["pred_population"].apply(lambda x: f"{x:,.0f}")
            print(top.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
