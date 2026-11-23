"""
02_build_policy_panel.py
Build the policy data: in which years did each state offer an R&D tax credit?

Inputs (data/raw/policy/)
    state_rd_credit_sources_handcoded.csv
        One row per state (50 states + DC), transcribed by hand from
        Wilson (2009 working paper version) Table 1 and
        Iowa Department of Revenue (2021) Table 1. No judgment calls in this file.
    iowa_table2_state_counts.csv
        Iowa (2021) Table 2: number of states with a credit in each tax year.

Outputs
    data/clean/policy_state.csv              one row per state
    data/clean/policy_state_year.csv         one row per state-year, 1976-2025
    output/tables/policy_source_comparison.csv   states where the sources disagree

Coding rules for the main specification (all judgment calls live here):
    R1  Start year = Wilson "year enacted" for the 32 states in Wilson Table 1
        (general credits on company-funded R&D, as of 2006).
    R2  States not in Wilson whose credit started after 2006 -> Iowa initial tax year.
    R3  States Wilson leaves out although Iowa shows a credit starting in 2006 or earlier
        (AK, AR, CO, NM, NY, WA) have non-standard credits -> excluded from the main sample.
    R4  End year = Iowa sunset year, but only if it is 2020 or earlier
        (an expiry that had already happened when Iowa published in 2021).
        Later sunsets are scheduled, not observed, so the credit is treated as ongoing.
    R5  States whose two sources disagree by more than one year, or whose credit
        was switched off and on again, are excluded from the main sample until verified.

Run from the project folder:  python3 code/02_build_policy_panel.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "policy"
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "output" / "tables"

FIRST_YEAR, LAST_YEAR = 1976, 2025   # patent data cover 1976-2025
IOWA_PUBLICATION_YEAR = 2021

ON_OFF_ON = {"NH"}   # 1993-94 credit (Wilson text), new credit from 2007 (Iowa)


# ---------------------------------------------------------------------------
# 1. Load the hand-coded sources and check them before using them
# ---------------------------------------------------------------------------
def load_sources():
    df = pd.read_csv(RAW / "state_rd_credit_sources_handcoded.csv",
                     dtype={"state_fips": str}, keep_default_na=False)

    # Turn year columns into numbers; "", "NA" and "No" become missing (NaN).
    for col in ["wilson_year_enacted", "iowa_initial_tax_year", "iowa_sunset"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    assert len(df) == 51, f"expected 50 states + DC, got {len(df)} rows"
    assert df["state_abbr"].is_unique and df["state_fips"].is_unique
    assert df["wilson_year_enacted"].notna().sum() == 32, "Wilson Table 1 lists 32 states"
    for col in ["wilson_year_enacted", "iowa_initial_tax_year"]:
        years = df[col].dropna()
        assert years.between(1981, LAST_YEAR).all(), f"{col} has an implausible year"
    return df


# ---------------------------------------------------------------------------
# 2. Apply the coding rules: one start year and one end year per state
# ---------------------------------------------------------------------------
def apply_rules(df):
    df = df.copy()
    gap = (df["wilson_year_enacted"] - df["iowa_initial_tax_year"]).abs()

    # Start year (R1, R2)
    not_in_wilson = df["wilson_year_enacted"].isna()
    df["start_year"] = df["wilson_year_enacted"]
    post_2006_new = not_in_wilson & (df["iowa_initial_tax_year"] > 2006)
    df.loc[post_2006_new, "start_year"] = df.loc[post_2006_new, "iowa_initial_tax_year"]
    nonstandard = not_in_wilson & (df["iowa_initial_tax_year"] <= 2006)          # R3

    # End year (R4): only expiries already observed when Iowa published
    observed_expiry = df["iowa_sunset"] <= IOWA_PUBLICATION_YEAR - 1
    df["end_year"] = df["iowa_sunset"].where(observed_expiry)

    # Why a state is left out of the main sample (empty = included)
    df["exclusion_reason"] = ""
    df.loc[nonstandard, "exclusion_reason"] = "nonstandard_credit"                           # R3
    df.loc[gap > 1, "exclusion_reason"] = "sources_disagree"                                 # R5
    df.loc[df["state_abbr"].isin(ON_OFF_ON), "exclusion_reason"] = "credit_switched_off_and_on"
    df["main_sample"] = df["exclusion_reason"] == ""

    # Useful labels
    df["one_year_gap"] = gap == 1             # for a robustness check using Iowa's year
    df["never_treated"] = df["start_year"].isna() & (df["exclusion_reason"] == "")
    df["repealed"] = df["end_year"].notna()
    return df


# ---------------------------------------------------------------------------
# 3. Expand to one row per state and year
# ---------------------------------------------------------------------------
def build_state_year(states):
    years = pd.DataFrame({"year": range(FIRST_YEAR, LAST_YEAR + 1)})
    panel = states[["state", "state_abbr", "state_fips", "start_year", "end_year",
                    "main_sample", "exclusion_reason"]].merge(years, how="cross")

    started = panel["year"] >= panel["start_year"]          # False when start_year is missing
    not_ended = panel["end_year"].isna() | (panel["year"] <= panel["end_year"])
    panel["credit_active"] = (started & not_ended).astype(int)

    # Event time: years since the credit started (negative = before; missing = never treated)
    panel["event_time"] = panel["year"] - panel["start_year"]
    panel["post_repeal"] = (panel["end_year"].notna() & (panel["year"] > panel["end_year"])).astype(int)
    return panel.sort_values(["state_abbr", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Validation: compare against what the sources themselves report
# ---------------------------------------------------------------------------
def validate(states, panel):
    print("\nCHECK 1  Wilson says 32 states had a credit in 2006.")
    wilson_states = set(states.loc[states["wilson_year_enacted"].notna(), "state_abbr"])
    wilson_2006 = panel[(panel["year"] == 2006) & panel["state_abbr"].isin(wilson_states)]
    still_active = wilson_2006["credit_active"].sum()
    print(f"         Under our coding, {still_active} of those 32 are active in 2006.")
    for _, row in wilson_2006[wilson_2006["credit_active"] == 0].iterrows():
        print(f"         - {row['state']}: listed by Wilson for 2006, but Iowa records an earlier expiry")

    print("\nCHECK 2  Rebuild Iowa Table 2 (states with a credit, by year) from Iowa Table 1 dates.")
    iowa = states.dropna(subset=["iowa_initial_tax_year"])
    counts = []
    for year in range(1981, 2022):
        active = (iowa["iowa_initial_tax_year"] <= year) & (
            iowa["iowa_sunset"].isna() | (iowa["iowa_sunset"] >= year))
        counts.append({"tax_year": year, "rebuilt": int(active.sum())})
    compare = pd.DataFrame(counts).merge(pd.read_csv(RAW / "iowa_table2_state_counts.csv"), on="tax_year")
    compare["difference"] = compare["rebuilt"] - compare["states_with_credit"]
    mismatched = compare[compare["difference"] != 0]
    print(f"         Matches in {len(compare) - len(mismatched)} of {len(compare)} years.")
    if len(mismatched):
        print("         Years that differ (Table 1 lists only each state's current program,")
        print("         so earlier programs that were later replaced are missing):")
        print(mismatched.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    sources = load_sources()
    states = apply_rules(sources)
    panel = build_state_year(states)

    CLEAN.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    states.to_csv(CLEAN / "policy_state.csv", index=False)
    panel.to_csv(CLEAN / "policy_state_year.csv", index=False)

    disagree = states[(states["wilson_year_enacted"] != states["iowa_initial_tax_year"])
                      & (states["wilson_year_enacted"].notna() | states["iowa_initial_tax_year"].notna())]
    disagree[["state", "wilson_year_enacted", "iowa_initial_tax_year", "iowa_sunset",
              "exclusion_reason", "source_note"]].to_csv(TABLES / "policy_source_comparison.csv", index=False)

    print("Policy data built.")
    print(f"  States in main sample: {states['main_sample'].sum()} of {len(states)}")
    print(f"    ever treated:  {(states['main_sample'] & states['start_year'].notna()).sum()}")
    print(f"    never treated: {states['never_treated'].sum()}")
    print("  Excluded from main sample:")
    for reason, group in states[~states["main_sample"]].groupby("exclusion_reason"):
        print(f"    {reason}: {', '.join(group['state_abbr'])}")
    print(f"  Sources disagree for {len(disagree)} states -> output/tables/policy_source_comparison.csv")

    cohorts = states[states["main_sample"]].dropna(subset=["start_year"])
    print("\n  Adoption cohorts in the main sample (start year: states)")
    for year, group in cohorts.groupby("start_year"):
        print(f"    {int(year)}: {', '.join(group['state_abbr'])}")

    validate(states, panel)


if __name__ == "__main__":
    main()
