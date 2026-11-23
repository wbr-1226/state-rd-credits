"""
06_firm_size_extension.py
Extension: do state R&D tax credits change patenting by small firms and large firms differently?

Inputs
    data/intermediate/patent_state_shares.pkl          (from 03) each patent's inventor share in
                                                       each state, with filing and grant years
    data/raw/patents/g_assignee_disambiguated.tsv.zip   who owns each patent (USPTO PatentsView)
    data/clean/analysis_panel.csv                       (from 04) policy timing and sample flags

Outputs
    data/clean/patents_by_size_state_year.csv   patents by state, filing year and firm size
    output/tables/event_study_by_size.csv       CS event-study estimates for each size group
    output/tables/firm_size_summary.csv         average effects by group
    output/figures/event_study_by_size.png

Firm size (fixed before looking at any results):
    The owner of a patent is its first-listed assignee. If the USPTO classifies that owner
    as a U.S. or foreign company or corporation (assignee types 2, 3, 12, 13), its size is
    the number of U.S. utility patents it owns in the whole data, 1976-2025.
        small   fewer than 50 patents
        medium  50 to 499 patents
        large   500 or more patents
        none    no such owner (individual inventors, governments, or no assignee)
    Patents are a proxy for firm size (the data have no employment or sales), and a firm's
    size is measured over the whole period, including after the credits; see the paper.

Outcome: log(1 + patents) by group, because small groups in small states can have zero
patents in a year. Patents are counted by inventor location and filing year, as in 03.

Run from the project folder:  python3 code/06_firm_size_extension.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from event_study_tools import (WINDOW, average_effect, callaway_santanna, plot_event_study,
                               prepare, pretrend_test)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "patents"
CLEAN = ROOT / "data" / "clean"
INTERMEDIATE = ROOT / "data" / "intermediate"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

YEARS = (1976, 2019)
SMALL_MAX, LARGE_MIN = 50, 500
ORGANIZATION_TYPES = {"2", "3", "12", "13"}   # U.S. and foreign organizations (incl. part interest)
GROUPS = ["small", "medium", "large", "none"]


# ---------------------------------------------------------------------------
# 1. Owner and size of every utility patent
# ---------------------------------------------------------------------------
def patent_size_groups():
    columns = ["patent_id", "assignee_sequence", "assignee_id", "assignee_type",
               "disambig_assignee_organization"]
    parts, names = [], []
    reader = pd.read_csv(RAW / "g_assignee_disambiguated.tsv.zip", sep="\t", usecols=columns,
                         dtype=str, compression="zip", chunksize=2_000_000)
    for chunk in reader:
        chunk = chunk[chunk["assignee_sequence"] == "0"]                 # first-listed owner
        chunk = chunk.assign(patent_id=pd.to_numeric(chunk["patent_id"], errors="coerce"))
        chunk = chunk.dropna(subset=["patent_id"])                      # utility patents only
        # Organization names are only needed for a check, so keep one row per owner
        names.append(chunk[["assignee_id", "disambig_assignee_organization"]].drop_duplicates("assignee_id"))
        parts.append(chunk[["patent_id", "assignee_id", "assignee_type"]])
    owners = pd.concat(parts, ignore_index=True)
    owners["patent_id"] = owners["patent_id"].astype("int64")
    owners["assignee_id"] = owners["assignee_id"].astype("category")     # saves memory
    assert owners["patent_id"].is_unique, "a patent has two first-listed owners"
    names = pd.concat(names).drop_duplicates("assignee_id").set_index("assignee_id")

    is_org = owners["assignee_type"].isin(ORGANIZATION_TYPES)
    size = owners[is_org].groupby("assignee_id", observed=True)["patent_id"].size()
    size = size.rename("org_patents")
    owners["org_patents"] = owners["assignee_id"].map(size).astype(float)
    owners["size_group"] = np.select(
        [~is_org, owners["org_patents"] < SMALL_MAX, owners["org_patents"] < LARGE_MIN],
        ["none", "small", "medium"], default="large")

    print("CHECK 1  Largest patent owners (should be familiar large firms)")
    top = size.nlargest(5).to_frame("patents").join(names)
    print(top[["disambig_assignee_organization", "patents"]].to_string(index=False))
    n_orgs = size.groupby(pd.cut(size, [0, SMALL_MAX - 1, LARGE_MIN - 1, np.inf],
                                 labels=["small", "medium", "large"]), observed=False).size()
    print(f"         organizations by size group: {n_orgs.to_dict()}")
    return owners[["patent_id", "size_group"]]


# ---------------------------------------------------------------------------
# 2. Patents by state, filing year and size group
# ---------------------------------------------------------------------------
def patents_by_size(groups):
    shares = pd.read_pickle(INTERMEDIATE / "patent_state_shares.pkl")
    shares = shares.merge(groups, on="patent_id", how="left")
    shares["size_group"] = shares["size_group"].fillna("none")      # no assignee record at all
    shares = shares.dropna(subset=["filing_year"])
    shares["year"] = shares["filing_year"].astype(int)

    wide = (shares.groupby(["state_fips", "year", "size_group"])["share"].sum()
            .unstack("size_group").reindex(columns=GROUPS).fillna(0))
    wide.columns = [f"patents_{g}" for g in GROUPS]
    wide = wide.reset_index()

    in_window = shares["year"].between(*YEARS)
    by_group = shares[in_window].groupby("size_group")["share"].sum()
    print("\nCHECK 2  Share of U.S.-invented patents (filed 1976-2019) by owner size")
    print((by_group / by_group.sum()).reindex(GROUPS).map("{:.1%}".format).to_string())

    # The four groups must add up to the total patent count from script 03
    totals = pd.read_csv(CLEAN / "patents_state_year.csv", dtype={"state_fips": str})
    check = totals.merge(wide, on=["state_fips", "year"], how="left").fillna(0)
    gap = (check[[f"patents_{g}" for g in GROUPS]].sum(axis=1) - check["patents_frac"]).abs().max()
    print(f"\nCHECK 3  Groups add up to the total in every state-year (largest gap {gap:.6f})")
    assert gap < 1e-6
    return wide


# ---------------------------------------------------------------------------
# 3. Event study for small and large firms
# ---------------------------------------------------------------------------
def main():
    groups = patent_size_groups()
    by_size = patents_by_size(groups)
    by_size.to_csv(CLEAN / "patents_by_size_state_year.csv", index=False)

    panel = pd.read_csv(CLEAN / "analysis_panel.csv", dtype={"state_fips": str})
    panel = panel.merge(by_size, on=["state_fips", "year"], how="left")
    for g in GROUPS:
        panel[f"ln1_patents_{g}"] = np.log1p(panel[f"patents_{g}"])

    print("\nCS event study of log(1 + patents) by owner size")
    series, tables, summary = {}, [], []
    for g in ["small", "large", "none"]:
        outcome = f"ln1_patents_{g}"
        d = prepare(panel, outcome, YEARS, verbose=(g == "small"))
        cs, res = callaway_santanna(d, outcome)
        early, early_se = average_effect(res, 0, 2)
        late, late_se = average_effect(res, 0, WINDOW[1])
        pre_p = pretrend_test(res)
        print(f"  {g:6s} years 0-2: {early:6.3f} ({early_se:.3f})   "
              f"years 0-20: {late:6.3f} ({late_se:.3f})   pre-trend p = {pre_p:.3f}")
        summary.append({"group": g, "effect_years_0_2": early, "se_0_2": early_se,
                        "effect_years_0_20": late, "se_0_20": late_se, "pretrend_p": pre_p})
        tables.append(cs.assign(group=g))
        if g != "none":
            series[f"{g.capitalize()} firms"] = cs

    # Is the small-firm effect different from the large-firm effect?
    # The CS estimator is linear in the outcome, so estimating it on the gap
    # log(1 + small) - log(1 + large) gives the difference between the two effects,
    # with a standard error that accounts for both groups coming from the same states.
    panel["gap_small_large"] = panel["ln1_patents_small"] - panel["ln1_patents_large"]
    d = prepare(panel, "gap_small_large", YEARS, verbose=False)
    cs, res = callaway_santanna(d, "gap_small_large")
    early, early_se = average_effect(res, 0, 2)
    late, late_se = average_effect(res, 0, WINDOW[1])
    p_early = 2 * stats.norm.sf(abs(early / early_se))
    p_late = 2 * stats.norm.sf(abs(late / late_se))
    print(f"  small minus large  years 0-2: {early:6.3f} ({early_se:.3f}), p = {p_early:.3f}   "
          f"years 0-20: {late:6.3f} ({late_se:.3f}), p = {p_late:.3f}")
    summary.append({"group": "small_minus_large", "effect_years_0_2": early, "se_0_2": early_se,
                    "effect_years_0_20": late, "se_0_20": late_se, "p_0_2": p_early,
                    "p_0_20": p_late, "pretrend_p": pretrend_test(res)})
    tables.append(cs.assign(group="small_minus_large"))

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(TABLES / "event_study_by_size.csv", index=False)
    pd.DataFrame(summary).round(4).to_csv(TABLES / "firm_size_summary.csv", index=False)
    plot_event_study(series, "Log(1 + patents) by size of the patent owner",
                     "Effect on log(1 + patents)", FIGURES / "event_study_by_size.png",
                     f"Callaway-Sant'Anna estimates. Small: owner has <{SMALL_MAX} patents; "
                     f"large: {LARGE_MIN}+. Bars: 95% CIs clustered by state.")
    print("\nSaved tables to output/tables/ and the figure to output/figures/")


if __name__ == "__main__":
    main()
