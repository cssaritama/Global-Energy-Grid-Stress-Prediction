"""Download and prepare the OPSD time-series dataset.

This script downloads the Open Power System Data (OPSD) hourly time series CSV and
builds a daily tabular ML dataset suitable for classification.

Output:
  - data/raw/time_series_60min_singleindex.csv  (large)
  - data/processed/energy_grid_daily.csv        (features + target)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


OPSD_URL_DEFAULT = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)

DEFAULT_COUNTRIES = ["DE", "FR", "ES", "IT", "NL", "PL", "SE"]


def download_file(url: str, dest: Path, chunk_size: int = 2**20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000_000:
        # Already downloaded
        return

    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    print(f"Saved to: {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def _pick_columns(header_cols: List[str], countries: List[str]) -> Dict[str, Dict[str, str]]:
    """Pick the best-available load / wind / solar columns for each country."""
    mapping: Dict[str, Dict[str, str]] = {}

    for c in countries:
        load_candidates = [
            f"{c}_load_actual_entsoe_transparency",
            f"{c}_load_actual",
            f"{c}_load",
        ]
        solar_candidates = [
            f"{c}_solar_generation_actual",
            f"{c}_solar_generation",
            f"{c}_solar",
        ]
        wind_candidates = [
            # Prefer split onshore/offshore if available
            f"{c}_wind_onshore_generation_actual",
            f"{c}_wind_offshore_generation_actual",
            f"{c}_wind_generation_actual",
            f"{c}_wind_generation",
            f"{c}_wind",
        ]

        def first_existing(cands: List[str]) -> str | None:
            for x in cands:
                if x in header_cols:
                    return x
            return None

        load_col = first_existing(load_candidates)
        solar_col = first_existing(solar_candidates)

        # For wind, we may have both onshore and offshore (we'll combine if both exist)
        wind_on = f"{c}_wind_onshore_generation_actual"
        wind_off = f"{c}_wind_offshore_generation_actual"
        if wind_on in header_cols or wind_off in header_cols:
            wind_cols = [x for x in [wind_on, wind_off] if x in header_cols]
        else:
            w = first_existing(wind_candidates)
            wind_cols = [w] if w else []

        if load_col is None:
            print(f"[WARN] No load column found for {c}. Skipping this country.")
            continue

        mapping[c] = {
            "load": load_col,
            "solar": solar_col or "",
            "wind_cols": ",".join([x for x in wind_cols if x]),
        }

    return mapping


def build_daily_dataset(raw_csv: Path, countries: List[str]) -> pd.DataFrame:
    # Read header to select columns
    header = pd.read_csv(raw_csv, nrows=0)
    cols = list(header.columns)

    if "utc_timestamp" not in cols:
        raise ValueError("Expected 'utc_timestamp' column not found in OPSD CSV.")

    mapping = _pick_columns(cols, countries)
    if not mapping:
        raise RuntimeError("No countries had usable columns. Check dataset / country codes.")

    usecols = ["utc_timestamp"]
    for c, m in mapping.items():
        usecols.append(m["load"])
        if m["solar"]:
            usecols.append(m["solar"])
        if m["wind_cols"]:
            usecols.extend(m["wind_cols"].split(","))

    # Load only the selected columns
    df = pd.read_csv(
        raw_csv,
        usecols=sorted(set(usecols)),
        parse_dates=["utc_timestamp"],
    )
    df = df.sort_values("utc_timestamp").set_index("utc_timestamp")

    daily_rows = []
    for c, m in mapping.items():
        load = df[m["load"]].astype("float64")
        solar = df[m["solar"]].astype("float64") if m["solar"] else pd.Series(0.0, index=df.index)
        wind = pd.Series(0.0, index=df.index)
        if m["wind_cols"]:
            for wcol in m["wind_cols"].split(","):
                wind = wind + df[wcol].astype("float64")

        # Basic cleaning: keep only non-negative values where possible
        load = load.where(load >= 0)
        solar = solar.where(solar >= 0).fillna(0.0)
        wind = wind.where(wind >= 0).fillna(0.0)

        # Drop timestamps with missing load
        tmp = pd.DataFrame({"load": load, "wind": wind, "solar": solar}).dropna(subset=["load"])

        # Derived hourly signals
        tmp["renewables"] = tmp["wind"] + tmp["solar"]
        tmp["renew_ratio"] = (tmp["renewables"] / tmp["load"]).clip(0, 2)  # allow >1 when exports happen
        tmp["residual_load"] = tmp["load"] - tmp["renewables"]

        # Daily aggregation
        g = tmp.resample("D")
        daily = pd.DataFrame({
            "country": c,
            "load_mean": g["load"].mean(),
            "load_max": g["load"].max(),
            "renewables_mean": g["renewables"].mean(),
            "renew_ratio_mean": g["renew_ratio"].mean(),
            "renew_ratio_std": g["renew_ratio"].std(),
            "residual_load_max": g["residual_load"].max(),
        }).dropna()

        daily["demand_peak_ratio"] = (daily["load_max"] / daily["load_mean"]).replace([np.inf, -np.inf], np.nan)
        daily["renewable_share"] = (daily["renewables_mean"] / daily["load_mean"]).clip(0, 2)
        daily["renewable_volatility"] = daily["renew_ratio_std"].fillna(0.0)

        # Load growth rate vs previous week average (robust to daily noise)
        prev7 = daily["load_mean"].shift(7).rolling(7, min_periods=7).mean()
        daily["load_growth_rate"] = ((daily["load_mean"] - prev7) / prev7).replace([np.inf, -np.inf], np.nan)

        # Capacity margin proxy:
        # Compare max residual load to the rolling max peak load (30 days).
        rolling_peak = daily["load_max"].rolling(30, min_periods=7).max()
        daily["capacity_margin"] = (1.0 - (daily["residual_load_max"] / rolling_peak)).clip(-1, 1)

        # Keep core features
        keep = daily[[
            "country",
            "demand_peak_ratio",
            "renewable_share",
            "renewable_volatility",
            "load_growth_rate",
            "capacity_margin",
            "load_max",
        ]].dropna()

        # Define target per country using quantiles (avoids country-scale leakage)
        q_load = keep["load_max"].quantile(0.90)
        q_margin = keep["capacity_margin"].quantile(0.10)

        keep["grid_stress"] = ((keep["load_max"] > q_load) & (keep["capacity_margin"] < q_margin)).astype(int)

        keep = keep.drop(columns=["load_max"])
        daily_rows.append(keep)

    out = pd.concat(daily_rows, axis=0).reset_index(drop=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=OPSD_URL_DEFAULT, help="OPSD CSV download URL")
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES), help="Comma-separated country codes")
    args = parser.parse_args()

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    raw_path = Path("data/raw/time_series_60min_singleindex.csv")
    processed_path = Path("data/processed/energy_grid_daily.csv")

    download_file(args.url, raw_path)
    dataset = build_daily_dataset(raw_path, countries)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed_path, index=False)
    print(f"Processed dataset saved: {processed_path} (rows={len(dataset):,})")
    print(dataset.head())


if __name__ == "__main__":
    main()
