"""
03_build_patents_state_year.py
Count U.S. utility patents by state and year, using the location of each patent's inventors.

Inputs (data/raw/patents/, USPTO PatentsView release 2026-04-10)
    g_patent.tsv.zip                   one row per granted patent (type, grant date)
    g_application.tsv.zip              one row per patent: filing (application) date
    g_inventor_disambiguated.tsv.zip   one row per patent-inventor pair, with location_id
    g_location_disambiguated.tsv.zip   location_id -> state FIPS code
    ncses_patents_all.json             NCSES state patent counts (used only as a check)

Output
    data/clean/patents_state_year.csv   one row per state (50 + DC) and year, 1976-2025:
        patents_frac         fractional count by FILING year (main outcome):
                             a patent with 3 inventors, 2 in CA and 1 in TX,
                             adds 2/3 to CA and 1/3 to TX
        patents_any          patents with at least one inventor in the state, by filing year
        patents_frac_grant   fractional count by GRANT year (only for the check against NCSES)

Why filing year: the time from filing to grant is 2-4 years and changed over time,
so grant year would blur when the innovation happened. Recent filing years are
incomplete (many applications are still pending); the analysis script handles this.

The inventor file has about 23 million rows, so it is read in chunks of 2 million rows.
Run from the project folder:  python3 code/03_build_patents_state_year.py   (about 3-5 minutes)
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "patents"
CLEAN = ROOT / "data" / "clean"
INTERMEDIATE = ROOT / "data" / "intermediate"   # large in-between files reused by later scripts
CHUNK_ROWS = 2_000_000
FIRST_YEAR, LAST_YEAR = 1976, 2025

# 50 states + DC (FIPS codes); territories such as Puerto Rico (72) are left out
STATE_FIPS = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17",
              "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
              "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
              "47", "48", "49", "50", "51", "53", "54", "55", "56"]


def read_tsv(name, columns, chunksize=CHUNK_ROWS):
    """Read selected columns of a zipped PatentsView table in chunks; values are read as text."""
    return pd.read_csv(RAW / name, sep="\t", usecols=columns, dtype=str,
                       compression="zip", chunksize=chunksize)


def numeric_id(ids):
    """Utility patent numbers are all digits; design, plant and reissue ids contain letters
    (D, PP, RE, ...) and become missing here. Whole numbers take far less memory than text."""
    return pd.to_numeric(ids, errors="coerce").astype("Int64")


# ---------------------------------------------------------------------------
# 1. Which patents to count, and in which year
# ---------------------------------------------------------------------------
def load_patent_years():
    parts, n_all = [], 0
    for chunk in read_tsv("g_patent.tsv.zip", ["patent_id", "patent_type", "patent_date"]):
        n_all += len(chunk)
        chunk = chunk[chunk["patent_type"] == "utility"]            # drop design, plant, reissue
        parts.append(pd.DataFrame({"patent_id": numeric_id(chunk["patent_id"]),
                                   "grant_year": chunk["patent_date"].str[:4].astype("int16")}))
    patents = pd.concat(parts, ignore_index=True)
    # A few 1976 reissues ("RE28707", ...) are labelled "utility"; they are not new inventions.
    mislabelled = patents["patent_id"].isna()
    print(f"  {mislabelled.sum():,} reissue patents mislabelled as utility dropped")
    patents = patents[~mislabelled]
    assert patents["patent_id"].is_unique

    parts = []
    for chunk in read_tsv("g_application.tsv.zip", ["patent_id", "filing_date"]):
        parts.append(pd.DataFrame({"patent_id": numeric_id(chunk["patent_id"]),
                                   "filing_year": pd.to_numeric(chunk["filing_date"].str[:4],
                                                                errors="coerce")}))
    filing = pd.concat(parts, ignore_index=True).dropna().drop_duplicates("patent_id")

    years = patents.merge(filing, on="patent_id", how="left", validate="one_to_one")

    # Some filing dates are typos (e.g. "1074-08-14" for 1974). A filing year before 1900
    # or after the grant year is impossible, so it is set to missing rather than guessed.
    impossible = (years["filing_year"] < 1900) | (years["filing_year"] > years["grant_year"])
    years.loc[impossible, "filing_year"] = pd.NA
    print(f"  patents: {n_all:,} granted, {len(years):,} utility")
    print(f"    {impossible.sum():,} impossible filing dates set to missing; "
          f"{years['filing_year'].isna().mean():.2%} of utility patents have no usable filing year")
    print(f"    {(years['filing_year'] < FIRST_YEAR).sum():,} were filed before {FIRST_YEAR} "
          f"(granted later; outside the filing-year panel)")
    return years.set_index("patent_id")


# ---------------------------------------------------------------------------
# 2. Location id -> state
# ---------------------------------------------------------------------------
def load_location_states():
    loc = pd.concat(read_tsv("g_location_disambiguated.tsv.zip",
                             ["location_id", "disambig_country", "state_fips"]))
    us = loc[(loc["disambig_country"] == "US") & loc["state_fips"].isin(STATE_FIPS)]
    print(f"  locations: {len(loc):,} total, {len(us):,} in the 50 states + DC")
    return us.set_index("location_id")["state_fips"]


# ---------------------------------------------------------------------------
# 3. Inventors -> each patent's share in each state (read in chunks)
# ---------------------------------------------------------------------------
def patent_state_shares(utility_ids, location_state):
    totals, by_state = [], []
    reader = read_tsv("g_inventor_disambiguated.tsv.zip", ["patent_id", "location_id"])
    n_rows = n_us = 0
    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.assign(patent_id=numeric_id(chunk["patent_id"]))
        chunk = chunk[chunk["patent_id"].isin(utility_ids)].copy()
        n_rows += len(chunk)
        totals.append(chunk.groupby("patent_id").size())                # all inventors, any country
        chunk["state_fips"] = chunk["location_id"].map(location_state)  # missing if not a U.S. state
        us = chunk.dropna(subset=["state_fips"])
        n_us += len(us)
        by_state.append(us.groupby(["patent_id", "state_fips"]).size())
        print(f"    chunk {i}: {n_rows:,} utility inventor rows so far")

    # A patent's inventors can fall in two chunks, so add the chunk results up again
    n_inventors = pd.concat(totals).groupby(level=0).sum().rename("n_inventors")
    shares = pd.concat(by_state).groupby(level=[0, 1]).sum().rename("n_in_state").reset_index()
    shares = shares.merge(n_inventors, left_on="patent_id", right_index=True, validate="many_to_one")
    shares["share"] = shares["n_in_state"] / shares["n_inventors"]
    print(f"  inventors: {n_rows:,} on utility patents, {n_us / n_rows:.1%} located in a U.S. state")

    assert (shares["share"] > 0).all() and (shares["share"] <= 1).all()
    return shares


# ---------------------------------------------------------------------------
# 4. Add up to state x year
# ---------------------------------------------------------------------------
def to_state_year(shares, years):
    shares = shares.merge(years, left_on="patent_id", right_index=True, how="left")
    by_filing = shares.dropna(subset=["filing_year"]).groupby(["state_fips", "filing_year"]).agg(
        patents_frac=("share", "sum"), patents_any=("patent_id", "nunique"))
    by_grant = shares.groupby(["state_fips", "grant_year"])["share"].sum().rename("patents_frac_grant")

    grid = pd.MultiIndex.from_product([STATE_FIPS, range(FIRST_YEAR, LAST_YEAR + 1)],
                                      names=["state_fips", "year"])
    out = pd.DataFrame(index=grid)
    out = out.join(by_filing.rename_axis(["state_fips", "year"]))
    out = out.join(by_grant.rename_axis(["state_fips", "year"]))
    return out.fillna(0).reset_index()     # a state-year with no patents has a count of 0


# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------
def validate(out):
    assert len(out) == 51 * (LAST_YEAR - FIRST_YEAR + 1), "state-year grid is incomplete"
    assert not out.duplicated(["state_fips", "year"]).any()

    print("\nCHECK 1  U.S.-invented utility patents by grant year (fractional count, selected years)")
    totals = out.groupby("year")["patents_frac_grant"].sum()
    print(totals.loc[[1980, 1990, 2000, 2010, 2020, 2024]].round(0).to_string())

    print("\nCHECK 2  Compare with NCSES patents by state and year (NCSES uses a different count)")
    ncses = json.loads((RAW / "ncses_patents_all.json").read_text())["data_points"]
    ncses = pd.DataFrame(ncses)
    ncses = ncses[ncses["region"].isin(STATE_FIPS) & ~ncses["suppressed"] & ~ncses["not_available"]]
    ncses = ncses.rename(columns={"region": "state_fips", "value": "ncses_patents"})
    both = out.merge(ncses[["state_fips", "year", "ncses_patents"]], on=["state_fips", "year"])
    corr = both[["patents_frac_grant", "ncses_patents"]].corr().iloc[0, 1]
    ratio = both["patents_frac_grant"].sum() / both["ncses_patents"].sum()
    print(f"         {len(both):,} state-years matched; correlation {corr:.3f}; "
          f"our total / NCSES total = {ratio:.2f}")
    assert corr > 0.95, "our counts do not track the NCSES counts; investigate before using them"

    print("\nCHECK 3  Largest states by patents filed in 2015 (fractional count)")
    top = out[out["year"] == 2015].nlargest(5, "patents_frac")[["state_fips", "patents_frac"]]
    print(top.round(0).to_string(index=False))


def main():
    print("Reading patent tables...")
    years = load_patent_years()
    location_state = load_location_states()
    shares = patent_state_shares(years.index, location_state)
    out = to_state_year(shares, years)
    validate(out)
    CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN / "patents_state_year.csv", index=False)
    print(f"\nSaved data/clean/patents_state_year.csv ({len(out):,} rows)")

    # Keep each patent's state shares so script 06 can split patents by firm size
    # without reading the 23-million-row inventor file again.
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    detail = shares[["patent_id", "state_fips", "share"]].merge(
        years, left_on="patent_id", right_index=True, how="left")
    detail.to_pickle(INTERMEDIATE / "patent_state_shares.pkl")
    print(f"Saved data/intermediate/patent_state_shares.pkl ({len(detail):,} patent-state rows)")


if __name__ == "__main__":
    main()
