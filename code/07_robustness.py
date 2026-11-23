"""
07_robustness.py
Robustness checks for the main patent result (Callaway and Sant'Anna estimator).

Each row changes ONE choice from the main specification and re-estimates:
    main               main specification (script 05)
    with_repealers     keep the 4 states that later repealed (counted as treated after adoption)
    iowa_years         use Iowa's start year for the 4 states where the sources differ by one year
    never_treated      compare only with the 9 never-adopting states (not with later adopters)
    per_capita         outcome: log patents per 100,000 residents
    drop_california    drop the largest patenting state

Input
    data/clean/analysis_panel.csv, data/clean/policy_state.csv
Output
    output/tables/robustness_patents.csv

Run from the project folder:  python3 code/07_robustness.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from event_study_tools import WINDOW, average_effect, callaway_santanna, prepare, pretrend_test

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "output" / "tables"
YEARS = (1976, 2019)


def estimate(name, panel, outcome="ln_patents", include_repealers=False,
             control_group="notyettreated"):
    d = prepare(panel, outcome, YEARS, include_repealers=include_repealers, verbose=False)
    cs, res = callaway_santanna(d, outcome, control_group=control_group)
    early, early_se = average_effect(res, 0, 2)
    late, late_se = average_effect(res, 0, WINDOW[1])
    pre_p = pretrend_test(res)
    row = {"specification": name,
           "adopting_states": d.loc[d["cohort"] > 0, "sid"].nunique(),
           "never_adopting_states": d.loc[d["cohort"] == 0, "sid"].nunique(),
           "effect_years_0_2": early, "se_0_2": early_se,
           "effect_years_0_20": late, "se_0_20": late_se,
           "pretrend_p": pre_p}
    print(f"  {name:16s} years 0-2: {early:6.3f} ({early_se:.3f})   "
          f"years 0-20: {late:6.3f} ({late_se:.3f})   pre-trend p = {pre_p:.3f}   "
          f"states: {row['adopting_states']} + {row['never_adopting_states']}")
    return row


def main():
    panel = pd.read_csv(CLEAN / "analysis_panel.csv", dtype={"state_fips": str})
    policy = pd.read_csv(CLEAN / "policy_state.csv", dtype={"state_fips": str})
    print("CS average effects on log patents (standard errors in parentheses)")

    rows = [estimate("main", panel),
            estimate("with_repealers", panel, include_repealers=True)]

    # Iowa start years for states whose two sources differ by exactly one year
    gap = policy.loc[policy["one_year_gap"], ["state_abbr", "iowa_initial_tax_year"]]
    alt = panel.merge(gap, on="state_abbr", how="left")
    alt["start_year"] = alt["iowa_initial_tax_year"].fillna(alt["start_year"])
    print(f"  (iowa_years changes: {', '.join(gap['state_abbr'])})")
    rows.append(estimate("iowa_years", alt))

    rows.append(estimate("never_treated", panel, control_group="nevertreated"))

    pc = panel.assign(ln_patents_pc=np.log(panel["patents_per_100k"]))
    rows.append(estimate("per_capita", pc, outcome="ln_patents_pc"))

    rows.append(estimate("drop_california", panel[panel["state_abbr"] != "CA"]))

    table = pd.DataFrame(rows)
    for col in ["effect_years_0_2", "effect_years_0_20"]:
        table[col.replace("effect", "pct")] = 100 * (np.exp(table[col]) - 1)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(TABLES / "robustness_patents.csv", index=False)
    print("\nSaved output/tables/robustness_patents.csv")


if __name__ == "__main__":
    main()
