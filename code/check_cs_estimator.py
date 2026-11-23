"""
check_cs_estimator.py
Check the hand-written Callaway-Sant'Anna estimator in event_study_tools.py against the
independent `csdid` package (a port of the Stata/R implementations), cell by cell.

For every adoption cohort g and year t it compares ATT(g,t) for the main outcome
(log patents, 1976-2019), for both comparison groups and both base-year choices.
It also shows the comparison with moderndid 0.0.3, the package used at first:
it agrees for never-treated comparisons but not for not-yet-treated ones.

Optional packages (not needed to reproduce the paper):  pip install csdid moderndid
Run from the project folder:  python3 code/check_cs_estimator.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from event_study_tools import callaway_santanna, prepare

ROOT = Path(__file__).resolve().parents[1]
YEARS = (1976, 2019)


def ours(d, control_group, base_period):
    _, res = callaway_santanna(d, "ln_patents", control_group, base_period)
    return res.att_gt.set_index(["cohort", "year"])["att"]


def csdid_cells(d, control_group, base_period):
    from csdid.att_gt import ATTgt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ATTgt(yname="ln_patents", gname="cohort", idname="sid", tname="year", data=d,
                    control_group=[control_group], bstrap=False).fit(
                        est_method="reg", base_period=base_period)
    r = fit.results
    return pd.Series(np.asarray(r["att"], dtype=float),
                     index=pd.MultiIndex.from_arrays([np.asarray(r["group"]).astype(int),
                                                      np.asarray(r["year"]).astype(int)],
                                                     names=["cohort", "year"]))


def moderndid_cells(d, control_group):
    import moderndid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = moderndid.att_gt(data=d, yname="ln_patents", tname="year", idname="sid",
                              gname="cohort", control_group=control_group,
                              base_period="varying", biters=100, random_state=1)
    return pd.Series(np.asarray(gt.att_gt, dtype=float),
                     index=pd.MultiIndex.from_arrays([np.asarray(gt.groups).astype(int),
                                                      np.asarray(gt.times).astype(int)],
                                                     names=["cohort", "year"]))


def compare(name, a, b):
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    gap = (both["a"] - both["b"]).abs()
    print(f"  {name:48s} {len(both):4d} cells   largest gap {gap.max():.2e}")
    return gap.max()


def main():
    panel = pd.read_csv(ROOT / "data" / "clean" / "analysis_panel.csv", dtype={"state_fips": str})
    d = prepare(panel, "ln_patents", YEARS, verbose=False)[["sid", "year", "cohort", "ln_patents"]]

    print("ATT(g,t) for log patents: this project's code vs csdid")
    worst = 0.0
    for cg in ["notyettreated", "nevertreated"]:
        for bp in ["universal", "varying"]:
            worst = max(worst, compare(f"{cg}, {bp} base", ours(d, cg, bp), csdid_cells(d, cg, bp)))
    assert worst < 1e-8, "the two implementations disagree"
    print("  -> identical (up to rounding)")

    try:
        print("\nFor reference: this project's code vs moderndid 0.0.3 (varying base)")
        for cg in ["nevertreated", "notyettreated"]:
            compare(f"{cg}", ours(d, cg, "varying"), moderndid_cells(d, cg))
    except ImportError:
        print("  moderndid not installed; skipped")


if __name__ == "__main__":
    main()
