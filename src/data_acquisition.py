"""
Data Acquisition Module
Downloads US Census population estimates and building permits data.

Population data: Bulk CSV files from the Census Population Estimates Program (PEP).
Building permits: Annual county-level files from the Census Building Permits Survey (BPS).
"""

import os
import time
import requests
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_file(url: str, dest_path: str, retries: int = 3, timeout: int = 60) -> bool:
    """Download a file from a URL with retry logic."""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 0:
            print(f"  ✓ Already exists: {os.path.basename(dest_path)} ({size:,} bytes)")
            return True

    for attempt in range(1, retries + 1):
        try:
            print(f"  ↓ Downloading: {url} (attempt {attempt}/{retries})")
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size = os.path.getsize(dest_path)
            print(f"  ✓ Saved: {os.path.basename(dest_path)} ({size:,} bytes)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Attempt {attempt} failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  ✗ FAILED after {retries} attempts: {url}")
    return False


def download_population_data() -> dict:
    """
    Download Census Population Estimates Program (PEP) bulk CSV files.
    Two files cover 2010-2020 and 2020-2024.
    Returns dict mapping period label to local file path.
    """
    print("\n" + "=" * 70)
    print("📊 DOWNLOADING POPULATION ESTIMATES")
    print("=" * 70)

    downloaded = {}
    for period, url in config.POPULATION_URLS.items():
        filename = url.split("/")[-1]
        dest = os.path.join(config.RAW_POP_DIR, filename)
        if download_file(url, dest):
            downloaded[period] = dest
        else:
            print(f"  ⚠ Could not download population data for {period}")

    return downloaded


def download_permits_data() -> dict:
    """
    Download Building Permits Survey (BPS) annual county-level files.
    Tries .txt first, falls back to .xlsx for recent years.
    Returns dict mapping year to local file path.
    """
    print("\n" + "=" * 70)
    print("🏗️  DOWNLOADING BUILDING PERMITS DATA")
    print("=" * 70)

    downloaded = {}
    for year in config.PERMITS_YEARS:
        # Try .txt first (most years)
        txt_filename = f"co{year}a.txt"
        txt_url = config.PERMITS_BASE_URL + txt_filename
        txt_dest = os.path.join(config.RAW_PERMITS_DIR, txt_filename)

        if download_file(txt_url, txt_dest):
            downloaded[year] = txt_dest
            continue

        # Fall back to .xlsx
        xlsx_filename = f"co{year}a.xlsx"
        xlsx_url = config.PERMITS_BASE_URL + xlsx_filename
        xlsx_dest = os.path.join(config.RAW_PERMITS_DIR, xlsx_filename)

        if download_file(xlsx_url, xlsx_dest):
            downloaded[year] = xlsx_dest
            continue

        # Try alternate CSV format
        csv_filename = f"co{year}a.csv"
        csv_url = config.PERMITS_BASE_URL + csv_filename
        csv_dest = os.path.join(config.RAW_PERMITS_DIR, csv_filename)

        if download_file(csv_url, csv_dest):
            downloaded[year] = csv_dest
        else:
            print(f"  ⚠ Could not download permits data for {year}")

    return downloaded


def run_acquisition() -> tuple:
    """Run the full data acquisition pipeline."""
    print("\n" + "═" * 70)
    print("  US COUNTY POPULATION MODEL — DATA ACQUISITION")
    print("═" * 70)

    pop_files = download_population_data()
    permit_files = download_permits_data()

    print("\n" + "=" * 70)
    print("📋 ACQUISITION SUMMARY")
    print("=" * 70)
    print(f"  Population files downloaded: {len(pop_files)}/2")
    print(f"  Permits files downloaded:    {len(permit_files)}/{len(config.PERMITS_YEARS)}")

    if len(pop_files) < 2:
        print("\n  ⚠ WARNING: Missing population data files. Pipeline may fail.")
    if len(permit_files) < len(config.PERMITS_YEARS):
        missing = set(config.PERMITS_YEARS) - set(permit_files.keys())
        print(f"\n  ⚠ WARNING: Missing permits for years: {sorted(missing)}")

    return pop_files, permit_files


if __name__ == "__main__":
    run_acquisition()
