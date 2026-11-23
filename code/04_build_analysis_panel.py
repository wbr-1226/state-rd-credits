"""
04_build_analysis_panel.py
Merge everything into one state-year table for the analysis.

Inputs
    data/clean/policy_state_year.csv                    (from 02) credit timing
    data/clean/patents_state_year.csv                   (from 03) patent counts
    data/raw/rd_spending/ncses_rdperformed_business.json   business R&D, $ millions, 1997-2023
    data/raw/rd_spending/ncses_privateindustryoutput.json  private-industry output, $ millions
    data/raw/controls/bea_SAINC.zip                      BEA table SAINC1: population, 1929-2025

Output
    data/clean/analysis_panel.csv   one row per state (50 + DC) and year, 1976-2025

New variables
    population, patents_per_100k, ln_patents   (patents = fractional count by filing year)
    business_rd, ln_business_rd, rd_intensity  (business R&D / private-industry output, %)
    patents_complete   1 if the filing year is early enough that nearly all of its
                       patents have been granted (filing year <= 2019; see script 03)

Run from the project folder:  python3 code/04_build_analysis_panel.py
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

LAST_COMPLETE_FILING_YEAR = 2019


def load_ncses(filename, name):
    """NCSES State Indicators series -> state_fips, year, value. Suppressed cells become missing."""
    points = json.loads((RAW / "rd_spending" / filename).read_text())["data_points"]
    df = pd.DataFrame(points)
    hidden = df["suppressed"] | df["not_available"]
    df.loc[hidden, "value"] = np.nan
    df = df[df["region"] != "00"]                       # drop the U.S. total
    df = df.rename(columns={"region": "state_fips", "value": name})
    assert not df.duplicated(["state_fips", "year"]).any()
    print(f"  {name}: {df['year'].min()}-{df['year'].max()}, "
          f"{int(hidden.sum())} suppressed or unavailable cells")
    return df[["state_fips", "year", name]]


def load_population():
    """BEA SAINC1, line 2 = population (persons). The file is wide: one column per year."""
    with zipfile.ZipFile(RAW / "controls" / "bea_SAINC.zip") as z:
        with z.open("SAINC1__ALL_AREAS_1929_2025.csv") as f:
            wide = pd.read_csv(f, dtype=str, encoding="latin-1")
    wide["GeoFIPS"] = wide["GeoFIPS"].str.strip().str.strip('"')
    wide = wide[(wide["LineCode"].str.strip() == "2") & wide["GeoFIPS"].str.endswith("000")]
    wide["state_fips"] = wide["GeoFIPS"].str[:2]
    year_columns = [c for c in wide.columns if c.isdigit()]
    long = wide.melt(id_vars="state_fips", value_vars=year_columns,
                     var_name="year", value_name="population")
    long["year"] = long["year"].astype(int)
    long["population"] = pd.to_numeric(long["population"], errors="coerce")
    return long


def main():
    print("Loading inputs...")
    policy = pd.read_csv(CLEAN / "policy_state_year.csv", dtype={"state_fips": str})
    patents = pd.read_csv(CLEAN / "patents_state_year.csv", dtype={"state_fips": str})
    rd = load_ncses("ncses_rdperformed_business.json", "business_rd")
    output = load_ncses("ncses_privateindustryoutput.json", "private_output")
    population = load_population()

    # Merge: the policy table defines the rows (51 states x 1976-2025); everything else is added
    n_rows = len(policy)
    panel = (policy
             .merge(patents, on=["state_fips", "year"], how="left", validate="one_to_one")
             .merge(rd, on=["state_fips", "year"], how="left", validate="one_to_one")
             .merge(output, on=["state_fips", "year"], how="left", validate="one_to_one")
             .merge(population, on=["state_fips", "year"], how="left", validate="one_to_one"))
    assert len(panel) == n_rows, "a merge added or dropped rows"

    # Analysis variables
    panel["patents_per_100k"] = panel["patents_frac"] / panel["population"] * 1e5
    panel["ln_patents"] = np.log(panel["patents_frac"].where(panel["patents_frac"] > 0))
    panel["ln_business_rd"] = np.log(panel["business_rd"].where(panel["business_rd"] > 0))
    panel["rd_intensity"] = panel["business_rd"] / panel["private_output"] * 100
    panel["patents_complete"] = (panel["year"] <= LAST_COMPLETE_FILING_YEAR).astype(int)

    # Checks
    print("\nCHECK 1  Every state-year has population, and patent counts are never missing")
    assert panel["population"].notna().all(), "missing population"
    assert panel["patents_frac"].notna().all()
    zero = panel[(panel["patents_frac"] == 0) & (panel["patents_complete"] == 1)]
    print(f"         population complete; {len(zero)} state-years with zero patents before 2020")

    print("\nCHECK 2  Coverage of business R&D (state-years with data, by period)")
    has_rd = panel.dropna(subset=["business_rd"])
    print(f"         {has_rd['year'].min()}-{has_rd['year'].max()}: {len(has_rd):,} state-years "
          f"({has_rd['state_fips'].nunique()} states)")

    print("\nCHECK 3  Sums across states vs. U.S. totals")
    us_pop_2020 = panel.loc[panel["year"] == 2020, "population"].sum()
    print(f"         population 2020, 50 states + DC: {us_pop_2020 / 1e6:.1f} million (Census: about 331)")
    us_rd = json.loads((RAW / "rd_spending" / "ncses_rdperformed_business.json").read_text())
    us_rd_2019 = next(p["value"] for p in us_rd["data_points"] if p["region"] == "00" and p["year"] == 2019)
    state_rd_2019 = panel.loc[panel["year"] == 2019, "business_rd"].sum()
    print(f"         business R&D 2019: states sum to ${state_rd_2019:,.0f}m, U.S. total ${us_rd_2019:,.0f}m "
          f"({state_rd_2019 / us_rd_2019:.1%}; the rest is undistributed or suppressed)")

    print("\nCHECK 4  Main-sample patents per 100,000 people, by credit status (descriptive only)")
    main_sample = panel[panel["main_sample"] & (panel["patents_complete"] == 1)]
    print(main_sample.groupby("credit_active")["patents_per_100k"].describe()[["count", "mean", "50%"]]
          .round(1).to_string())

    panel.to_csv(CLEAN / "analysis_panel.csv", index=False)
    print(f"\nSaved data/clean/analysis_panel.csv ({len(panel):,} rows, {panel.shape[1]} columns)")


if __name__ == "__main__":
    main()
