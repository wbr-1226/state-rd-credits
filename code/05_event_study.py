"""
05_event_study.py
Main results: how do patents and business R&D change in the years before and after
a state introduces an R&D tax credit?

Input
    data/clean/analysis_panel.csv   (from 04)

Outputs
    output/tables/event_study_<outcome>.csv     coefficients by event time, both estimators
    output/figures/event_study_<outcome>.png    the event-study figure

Two estimators, reported side by side (functions in event_study_tools.py):
    TWFE  Two-way fixed effects regression with one dummy per event time
          (state and year fixed effects; endpoints binned at -10 and +20).
          Standard, but biased when effects differ across adoption cohorts,
          because already-treated states end up serving as controls.
    CS    Callaway and Sant'Anna (2021): compares each adoption cohort only with
          states that have not adopted yet (or never adopt), then averages by event time.
          Every estimate is a change since the year before adoption ("universal" base
          year), so pre-period points are read the same way as the TWFE ones.

Sample: main-sample states (script 02) except the four that later repealed the credit
(they return in script 07). Event time -1 is the reference period.

Run from the project folder:  python3 code/05_event_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from event_study_tools import (WINDOW, average_effect, callaway_santanna, plot_event_study,
                               prepare, pretrend_test, twfe)

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

OUTCOMES = {
    "ln_patents": {"label": "Log patents (fractional count, by filing year)",
                   "ylabel": "Effect on log patents", "years": (1976, 2019)},
    "ln_business_rd": {"label": "Log business R&D spending",
                       "ylabel": "Effect on log business R&D", "years": (1997, 2023)},
}
NOTE = ("Both estimators: change relative to the year before adoption (TWFE endpoints binned). "
        "Bars: 95% CIs clustered by state.")


def main():
    panel = pd.read_csv(CLEAN / "analysis_panel.csv", dtype={"state_fips": str})
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    for outcome, spec in OUTCOMES.items():
        print(f"\n=== {spec['label']} ===")
        d = prepare(panel, outcome, spec["years"])
        tw, p_pre = twfe(d, outcome)
        cs, res = callaway_santanna(d, outcome)
        att, att_se = average_effect(res, 0, WINDOW[1])
        early, early_se = average_effect(res, 0, 2)
        cs_pre_p = pretrend_test(res)
        pre_avg, pre_avg_se = average_effect(res, WINDOW[0], -2)

        table = tw.merge(cs, on="event_time", how="outer", suffixes=("_twfe", "_cs"))
        table.to_csv(TABLES / f"event_study_{outcome}.csv", index=False)
        plot_event_study({"TWFE": tw, "CS": cs}, spec["label"], spec["ylabel"],
                         FIGURES / f"event_study_{outcome}.png", NOTE)

        pre = cs[cs["event_time"] < -1]
        print(f"  TWFE pre-trend test (event times -10 to -2 jointly zero): p = {p_pre:.3f}")
        print(f"  CS   pre-trend test (event times -10 to -2 jointly zero): p = {cs_pre_p:.3f}")
        print(f"  CS pre-period estimates: average {pre_avg:.3f} (s.e. {pre_avg_se:.3f}), "
              f"{(pre['estimate'] > 0).sum()} of {len(pre)} positive")
        print(f"  CS average effect, years 0-2:  {early:.3f} (s.e. {early_se:.3f}) "
              f"-> about {100 * (np.exp(early) - 1):.0f}%")
        print(f"  CS average effect, years 0-{WINDOW[1]}: {att:.3f} (s.e. {att_se:.3f}) "
              f"-> about {100 * (np.exp(att) - 1):.0f}%")

    print("\nSaved tables to output/tables/ and figures to output/figures/")


if __name__ == "__main__":
    main()
