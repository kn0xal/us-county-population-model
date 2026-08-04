"""
Data Processing Module
Cleans, reshapes, and merges Census population and building permits data
into a unified panel dataset.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def load_population_data() -> pd.DataFrame:
    """
    Load and merge the two Census PEP population CSV files (2010-2020, 2020-2024).
    Reshape from wide to long format: (FIPS, year, population).
    """
    print("\n── Loading Population Data ──")

    frames = []

    # --- 2010-2020 file ---
    path_2010 = os.path.join(config.RAW_POP_DIR, "co-est2020-alldata.csv")
    if os.path.exists(path_2010):
        df = pd.read_csv(path_2010, encoding="latin-1")
        df = df[df["SUMLEV"] == config.COUNTY_SUMLEV].copy()

        # Construct 5-digit FIPS
        df["FIPS"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(str).str.zfill(3)

        # Extract population columns for 2010-2019 (2020 will come from the newer file)
        pop_cols = {f"POPESTIMATE{y}": y for y in range(2010, 2020)}
        available = {c: y for c, y in pop_cols.items() if c in df.columns}

        melted = df[["FIPS", "STNAME", "CTYNAME"] + list(available.keys())].melt(
            id_vars=["FIPS", "STNAME", "CTYNAME"],
            var_name="year_col",
            value_name="population",
        )
        melted["year"] = melted["year_col"].map({c: y for c, y in available.items()})
        melted = melted.drop(columns=["year_col"])
        frames.append(melted)
        print(f"  ✓ 2010-2020 file: {len(df)} counties, {len(available)} years")
    else:
        print(f"  ✗ Missing: {path_2010}")

    # --- 2020-2024 file ---
    path_2020 = os.path.join(config.RAW_POP_DIR, "co-est2024-alldata.csv")
    if os.path.exists(path_2020):
        df = pd.read_csv(path_2020, encoding="latin-1")
        df = df[df["SUMLEV"] == config.COUNTY_SUMLEV].copy()
        df["FIPS"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(str).str.zfill(3)

        # Extract population columns for 2020-2024
        pop_cols = {f"POPESTIMATE{y}": y for y in range(2020, 2025)}
        available = {c: y for c, y in pop_cols.items() if c in df.columns}

        melted = df[["FIPS", "STNAME", "CTYNAME"] + list(available.keys())].melt(
            id_vars=["FIPS", "STNAME", "CTYNAME"],
            var_name="year_col",
            value_name="population",
        )
        melted["year"] = melted["year_col"].map({c: y for c, y in available.items()})
        melted = melted.drop(columns=["year_col"])
        frames.append(melted)
        print(f"  ✓ 2020-2024 file: {len(df)} counties, {len(available)} years")
    else:
        print(f"  ✗ Missing: {path_2020}")

    if not frames:
        raise FileNotFoundError("No population data files found. Run data acquisition first.")

    pop = pd.concat(frames, ignore_index=True)

    # Remove duplicates (2020 appears in both files — keep the newer estimate)
    pop = pop.sort_values(["FIPS", "year"]).drop_duplicates(
        subset=["FIPS", "year"], keep="last"
    )

    # Filter out territories (state FIPS > 56)
    pop["state_fips"] = pop["FIPS"].str[:2].astype(int)
    pop = pop[pop["state_fips"] <= config.MAX_STATE_FIPS].drop(columns=["state_fips"])

    pop = pop.rename(columns={"STNAME": "state_name", "CTYNAME": "county_name"})
    pop["population"] = pop["population"].astype(int)

    print(f"  → Combined: {pop['FIPS'].nunique()} counties × {pop['year'].nunique()} years = {len(pop)} rows")
    return pop


def _parse_permits_txt(filepath: str, year: int) -> pd.DataFrame:
    """
    Parse a BPS county-level .txt file.
    
    Known BPS county file format (comma-separated, 2-row header):
      Row 0: Survey, FIPS, FIPS, Region, Division, County, , 1-unit, , , 2-units, ...
      Row 1: Date, State, County, Code, Code, Name, Bldgs, Units, Value, Bldgs, Units, Value, ...
      Row 2: (blank)
      Row 3+: data rows
    
    Data columns (0-indexed):
      0: Survey Date (year)
      1: State FIPS (2-digit)
      2: County FIPS (3-digit)
      3: Region Code
      4: Division Code
      5: County Name
      6: 1-unit Bldgs,  7: 1-unit Units,  8: 1-unit Value
      9: 2-unit Bldgs, 10: 2-unit Units, 11: 2-unit Value
     12: 3-4 unit Bldgs, 13: 3-4 unit Units, 14: 3-4 unit Value
     15: 5+ unit Bldgs, 16: 5+ unit Units, 17: 5+ unit Value
     18+: Reported versions of the same (1-unit rep, 2-unit rep, etc.)
    """
    try:
        df = pd.read_csv(
            filepath,
            encoding="latin-1",
            dtype=str,
            on_bad_lines="skip",
            header=None,
            skipinitialspace=True,
            skiprows=2,  # Skip the 2-row header
        )
    except Exception as e:
        print(f"    ✗ Could not parse {filepath}: {e}")
        return pd.DataFrame()

    # Drop blank/empty rows
    df = df.dropna(how="all").reset_index(drop=True)
    df = df[df.iloc[:, 0].astype(str).str.strip() != ""].reset_index(drop=True)

    if df.empty or len(df.columns) < 18:
        print(f"    ✗ Insufficient columns ({len(df.columns)}) in {filepath}")
        return pd.DataFrame()

    # Use known column positions
    result = pd.DataFrame()
    result["state_fips"] = df.iloc[:, 1].astype(str).str.strip().str.zfill(2)
    result["county_fips"] = df.iloc[:, 2].astype(str).str.strip().str.zfill(3)
    result["FIPS"] = result["state_fips"] + result["county_fips"]

    def to_int(series):
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", ""),
            errors="coerce",
        ).fillna(0).astype(int)

    # Extract units columns (Bldgs=col+0, Units=col+1, Value=col+2 for each type)
    result["sf_units"] = to_int(df.iloc[:, 7])        # 1-unit Units
    result["two_unit"] = to_int(df.iloc[:, 10])        # 2-unit Units
    result["three_four_unit"] = to_int(df.iloc[:, 13]) # 3-4 unit Units
    result["five_plus_unit"] = to_int(df.iloc[:, 16])  # 5+ unit Units

    result["mf_units"] = result["two_unit"] + result["three_four_unit"] + result["five_plus_unit"]
    result["total_units"] = result["sf_units"] + result["mf_units"]

    # Total valuation (sum of all type values)
    result["total_valuation"] = (
        to_int(df.iloc[:, 8])   # 1-unit value
        + to_int(df.iloc[:, 11])  # 2-unit value
        + to_int(df.iloc[:, 14])  # 3-4 unit value
        + to_int(df.iloc[:, 17])  # 5+ unit value
    )

    result["year"] = year

    # Filter to valid US state FIPS (1-56, excluding territories)
    result["state_num"] = pd.to_numeric(result["state_fips"], errors="coerce")
    result = result[result["state_num"].between(1, config.MAX_STATE_FIPS)]

    # Filter out county FIPS 000 (state totals)
    result = result[result["county_fips"] != "000"]

    result = result[["FIPS", "year", "sf_units", "mf_units", "total_units", "total_valuation"]]

    # Aggregate by FIPS in case of duplicates
    result = result.groupby(["FIPS", "year"], as_index=False).sum()

    return result


def _parse_permits_xlsx(filepath: str, year: int) -> pd.DataFrame:
    """Parse a BPS county-level .xlsx file."""
    try:
        df = pd.read_excel(filepath, dtype=str, header=None)
    except Exception as e:
        print(f"    ✗ Could not read {filepath}: {e}")
        return pd.DataFrame()

    # The XLSX files usually have headers — try to find them
    # Look for a row containing "State" or "FIPS" 
    header_row = None
    for idx in range(min(10, len(df))):
        row_str = " ".join(df.iloc[idx].astype(str).tolist()).lower()
        if "state" in row_str and ("county" in row_str or "fips" in row_str):
            header_row = idx
            break

    if header_row is not None:
        df.columns = df.iloc[header_row].astype(str).str.strip()
        df = df.iloc[header_row + 1:]
    
    # Try to use the same TXT parser logic by writing to CSV and re-parsing
    # Actually, let's try a direct approach for XLSX
    cols_lower = {c: c.lower().strip() for c in df.columns}

    # Find state and county FIPS columns
    state_col = None
    county_col = None
    for c, cl in cols_lower.items():
        if "state" in cl and "fips" in cl:
            state_col = c
        elif "state" in cl and state_col is None:
            state_col = c
        if "county" in cl and "fips" in cl:
            county_col = c

    if state_col is None or county_col is None:
        # Fall back to TXT parser approach
        return _parse_permits_txt(filepath, year)

    result = pd.DataFrame()
    result["FIPS"] = (
        df[state_col].astype(str).str.strip().str.zfill(2)
        + df[county_col].astype(str).str.strip().str.zfill(3)
    )

    # Look for unit columns
    unit_mapping = {}
    for c, cl in cols_lower.items():
        if "1-unit" in cl or "1 unit" in cl or "single" in cl.replace("-", ""):
            if "unit" in cl and "bldg" not in cl and "val" not in cl:
                unit_mapping["sf_units"] = c
        elif "2-unit" in cl or "2 unit" in cl:
            if "unit" in cl and "bldg" not in cl and "val" not in cl:
                unit_mapping["two_unit"] = c
        elif ("3-4" in cl or "3 and 4" in cl):
            if "unit" in cl and "bldg" not in cl and "val" not in cl:
                unit_mapping["three_four_unit"] = c
        elif ("5+" in cl or "5 unit" in cl or "five" in cl):
            if "unit" in cl and "bldg" not in cl and "val" not in cl:
                unit_mapping["five_plus_unit"] = c

    for col_name in ["sf_units", "two_unit", "three_four_unit", "five_plus_unit"]:
        if col_name in unit_mapping:
            result[col_name] = pd.to_numeric(
                df[unit_mapping[col_name]].astype(str).str.replace(",", ""),
                errors="coerce",
            ).fillna(0).astype(int)
        else:
            result[col_name] = 0

    result["mf_units"] = result["two_unit"] + result["three_four_unit"] + result["five_plus_unit"]
    result["total_units"] = result["sf_units"] + result["mf_units"]
    result["total_valuation"] = 0  # Simplified for XLSX
    result["year"] = year

    result["state_num"] = pd.to_numeric(result["FIPS"].str[:2], errors="coerce")
    result = result[result["state_num"].between(1, config.MAX_STATE_FIPS)]
    result = result[result["FIPS"].str[2:] != "000"]

    result = result[["FIPS", "year", "sf_units", "mf_units", "total_units", "total_valuation"]]
    result = result.groupby(["FIPS", "year"], as_index=False).sum()

    return result


def load_permits_data() -> pd.DataFrame:
    """
    Load all annual BPS building permits files and combine into a long-format DataFrame.
    """
    print("\n── Loading Building Permits Data ──")

    frames = []
    for year in config.PERMITS_YEARS:
        # Try .txt first, then .xlsx, then .csv
        for ext in [".txt", ".xlsx", ".csv"]:
            filepath = os.path.join(config.RAW_PERMITS_DIR, f"co{year}a{ext}")
            if os.path.exists(filepath):
                print(f"  Parsing {os.path.basename(filepath)}...", end=" ")
                if ext == ".xlsx":
                    df = _parse_permits_xlsx(filepath, year)
                else:
                    df = _parse_permits_txt(filepath, year)
                if not df.empty:
                    print(f"→ {len(df)} counties")
                    frames.append(df)
                else:
                    print("→ 0 rows (parse failed)")
                break
        else:
            print(f"  ✗ No file found for {year}")

    if not frames:
        raise FileNotFoundError("No building permits data files found. Run data acquisition first.")

    permits = pd.concat(frames, ignore_index=True)
    print(f"  → Combined: {permits['FIPS'].nunique()} counties × {permits['year'].nunique()} years = {len(permits)} rows")
    return permits


def merge_panel(pop: pd.DataFrame, permits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge population and permits data into a unified panel dataset.
    Uses inner join on (FIPS, year).
    """
    print("\n── Merging Panel Dataset ──")

    panel = pd.merge(pop, permits, on=["FIPS", "year"], how="left")

    # Fill missing permits with 0 (some small counties may not report)
    permit_cols = ["sf_units", "mf_units", "total_units", "total_valuation"]
    for col in permit_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)

    # Rename for clarity
    panel = panel.rename(columns={
        "sf_units": "sf_permits",
        "mf_units": "mf_permits",
        "total_units": "total_permits",
        "total_valuation": "valuation",
    })

    # Sort by FIPS and year
    panel = panel.sort_values(["FIPS", "year"]).reset_index(drop=True)

    # Validate
    n_counties = panel["FIPS"].nunique()
    n_years = panel["year"].nunique()
    print(f"  ✓ Panel: {n_counties} counties × {n_years} years = {len(panel)} rows")
    print(f"  ✓ Year range: {panel['year'].min()} – {panel['year'].max()}")
    print(f"  ✓ Population range: {panel['population'].min():,} – {panel['population'].max():,}")
    print(f"  ✓ Total permits range: {panel['total_permits'].min():,} – {panel['total_permits'].max():,}")

    # Check for issues
    null_count = panel.isnull().sum().sum()
    if null_count > 0:
        print(f"  ⚠ {null_count} null values remaining")

    return panel


def run_processing() -> pd.DataFrame:
    """Run the full data processing pipeline."""
    print("\n" + "═" * 70)
    print("  US COUNTY POPULATION MODEL — DATA PROCESSING")
    print("═" * 70)

    pop = load_population_data()
    permits = load_permits_data()
    panel = merge_panel(pop, permits)

    # Save processed panel
    output_path = os.path.join(config.PROCESSED_DIR, config.PANEL_FILENAME)
    panel.to_csv(output_path, index=False)
    print(f"\n  💾 Saved panel dataset to: {output_path}")

    return panel


if __name__ == "__main__":
    run_processing()
