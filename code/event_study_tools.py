"""
event_study_tools.py
Shared event-study functions used by scripts 05 (main results), 06 (firm-size
extension) and 07 (robustness checks), so every script estimates in exactly the same way.

    prepare()            build a balanced estimation sample
    twfe()               two-way fixed effects event study (+ joint pre-trend test)
    callaway_santanna()  Callaway and Sant'Anna (2021) event study, written out here
    average_effect()     CS average effect over a range of event times
    pretrend_test()      CS joint test that all pre-period estimates are zero
    plot_event_study()   standard figure

Why the CS estimator is written out by hand: the Python package used at first
(moderndid 0.0.3) agrees with a direct calculation when the comparison group is
the never-treated states, but not when it is the not-yet-treated states (our main
comparison group). Two independent calculations agree with each other and not with
that package: the `csdid` package and the formulas below. check_cs_estimator.py
repeats this comparison.
"""

import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")                    # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

WINDOW = (-10, 20)                       # event times shown; TWFE bins the endpoints
N_BOOT = 999                             # bootstrap replications (states resampled)
SEED = 42
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical colours, used in order
MARKERS = ["o", "s", "^"]


# ---------------------------------------------------------------------------
# Estimation sample
# ---------------------------------------------------------------------------
def prepare(panel, outcome, years, include_repealers=False, verbose=True):
    """Main-sample states, balanced over `years`. Drops states that later repealed the
    credit (unless include_repealers), states with a missing outcome in any year, and
    states treated before the first year (no pre-period)."""
    d = panel[panel["main_sample"] & panel["year"].between(*years)].copy()
    repealers = sorted(d.loc[d["end_year"].notna(), "state_abbr"].unique())
    if not include_repealers:
        d = d[~d["state_abbr"].isin(repealers)]
    d = d.dropna(subset=[outcome])

    n_years = years[1] - years[0] + 1
    complete = d.groupby("state_abbr")["year"].transform("size") == n_years
    incomplete = sorted(d.loc[~complete, "state_abbr"].unique())
    d = d[complete]

    always = d["start_year"] <= years[0]
    always_treated = sorted(d.loc[always, "state_abbr"].unique())
    d = d[~always].copy()

    d["sid"] = d["state_fips"].astype(int)
    d["cohort"] = d["start_year"].fillna(0).astype(int)      # 0 = never treated
    if verbose:
        n_treated = d.loc[d["cohort"] > 0, "sid"].nunique()
        n_never = d.loc[d["cohort"] == 0, "sid"].nunique()
        print(f"  sample {years[0]}-{years[1]}: {n_treated} adopting + {n_never} never-adopting "
              f"states, {len(d):,} state-years")
        dropped_rep = [] if include_repealers else repealers
        print(f"    dropped: repealed later {dropped_rep}; missing years {incomplete}; "
              f"adopted before {years[0]} {always_treated}")
    return d


# ---------------------------------------------------------------------------
# Two-way fixed effects
# ---------------------------------------------------------------------------
def twfe(d, outcome, windows=()):
    """Event-study regression with state and year fixed effects, s.e. clustered by state.
    Returns (table, joint pre-trend p-value) and, if `windows` are given, a third item:
    {(first, last): (average coefficient over that window, s.e.)}."""
    d = d.copy()
    lo, hi = WINDOW
    rel = (d["year"] - d["start_year"]).clip(lo, hi)
    dummies = {}
    for k in range(lo, hi + 1):
        if k == -1:
            continue                                         # reference period
        name = f"et_m{-k}" if k < 0 else f"et_p{k}"
        d[name] = ((rel == k) & d["start_year"].notna()).astype(int)
        if d[name].sum() > 0:
            dummies[k] = name

    fit = pf.feols(f"{outcome} ~ {' + '.join(dummies.values())} | sid + year",
                   data=d, vcov={"CRV1": "sid"})
    coef, se = fit.coef(), fit.se()
    rows = [{"event_time": k, "estimate": coef[n], "se": se[n]} for k, n in dummies.items()]
    rows.append({"event_time": -1, "estimate": 0.0, "se": 0.0})

    pre = [n for k, n in dummies.items() if k < -1]       # joint test: pre-periods all zero
    b = coef[pre].to_numpy()
    V = pd.DataFrame(fit._vcov, index=coef.index, columns=coef.index).loc[pre, pre].to_numpy()
    p_pre = stats.chi2.sf(float(b @ np.linalg.solve(V, b)), df=len(pre))
    table = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    if not windows:
        return table, p_pre

    V_all = pd.DataFrame(fit._vcov, index=coef.index, columns=coef.index)
    averages = {}
    for w in windows:              # (first, last), or (first, last, base_first, base_last)
        a = pd.Series(0.0, index=coef.index)
        names = [n for k, n in dummies.items() if w[0] <= k <= w[1]]
        a[names] += 1 / len(names)
        if len(w) == 4:            # minus the average over a baseline window
            base = [n for k, n in dummies.items() if w[2] <= k <= w[3]]
            a[base] -= 1 / len(base)
        averages[tuple(w)] = (float(a @ coef), float(np.sqrt(a @ V_all @ a)))
    return table, p_pre, averages


# ---------------------------------------------------------------------------
# Callaway and Sant'Anna (2021), no covariates
# ---------------------------------------------------------------------------
# Step 1. Group-time effect for adoption cohort g in year t:
#     ATT(g,t) = [mean change in g from base year b to t]
#              - [mean change in the comparison states from b to t]
#   base year b = g - 1 (the year before adoption) for every t ("universal" base),
#                 or t - 1 for pre-adoption years ("varying" base, R's default);
#                 after adoption both choices give the same numbers.
#   comparison states: never-treated states, plus (for "notyettreated") states that
#                 adopt only after both t and b.
# Step 2. Event-time effect theta(e): average of ATT(g, g+e) over the cohorts observed
#   at event time e, weighted by the number of states in each cohort.
# Step 3. Standard errors: resample states with replacement (all years of a state move
#   together, which is clustering by state) and repeat steps 1-2 on each sample.
#   A resampled state that is drawn k times simply gets weight k.

@dataclass
class CSResult:
    event_times: np.ndarray    # e = WINDOW[0], ..., WINDOW[1]
    estimate: np.ndarray       # theta(e) in the data
    draws: np.ndarray          # theta(e) in each bootstrap sample, shape (N_BOOT, n_e)
    att_gt: pd.DataFrame       # group-time effects in the data (cohort, year, att)


def _group_time_effects(Y, cohorts, years, weights, control_group, base_period):
    """Y: (n_years, n_states) outcomes; cohorts: (n_states,) adoption year, 0 = never;
    weights: (n_samples, n_states) state weights. Returns {(g, t): att per sample}."""
    row = {y: i for i, y in enumerate(years)}
    att = {}
    for g in np.unique(cohorts[cohorts > 0]):
        treated = cohorts == g
        for t in years:
            b = g - 1 if (t >= g or base_period == "universal") else t - 1
            if b not in row or t == b:
                continue
            if control_group == "notyettreated":
                control = ((cohorts == 0) | (cohorts > max(t, b))) & ~treated
            elif control_group == "nevertreated":
                control = cohorts == 0
            else:
                raise ValueError(control_group)
            change = Y[row[t]] - Y[row[b]]
            w_t, w_c = weights[:, treated], weights[:, control]
            with np.errstate(invalid="ignore", divide="ignore"):
                att[(g, t)] = (w_t @ change[treated] / w_t.sum(axis=1)
                               - w_c @ change[control] / w_c.sum(axis=1))
    return att


def _event_time_effects(att, cohorts, weights, event_times, keep_cohorts=None):
    """theta(e) for each sample: cohort-size-weighted average of ATT(g, g+e),
    over all adoption cohorts or only those in keep_cohorts."""
    theta = np.full((weights.shape[0], len(event_times)), np.nan)
    use = [g for g in np.unique(cohorts[cohorts > 0]) if keep_cohorts is None or g in keep_cohorts]
    for j, e in enumerate(event_times):
        num = np.zeros(weights.shape[0])
        den = np.zeros(weights.shape[0])
        for g in use:
            if (g, g + e) not in att:
                continue
            a = att[(g, g + e)]
            n_g = weights[:, cohorts == g].sum(axis=1)
            ok = np.isfinite(a) & (n_g > 0)
            num[ok] += n_g[ok] * a[ok]
            den[ok] += n_g[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            theta[:, j] = np.where(den > 0, num / den, np.nan)
    return theta


def callaway_santanna(d, outcome, control_group="notyettreated", base_period="universal",
                      keep_cohorts=None):
    """CS event study on a balanced panel from prepare(). Returns (table, CSResult).
    keep_cohorts: average only these adoption cohorts (all states still serve as
    comparison states while they have not adopted)."""
    wide = d.pivot(index="year", columns="sid", values=outcome).sort_index()
    years = wide.index.to_numpy()
    cohorts = d.groupby("sid")["cohort"].first().reindex(wide.columns).to_numpy()
    Y = wide.to_numpy()
    n = Y.shape[1]

    rng = np.random.default_rng(SEED)
    resampled = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)   # times each state is drawn
    weights = np.vstack([np.ones(n), resampled])                     # row 0 = the actual data

    att = _group_time_effects(Y, cohorts, years, weights, control_group, base_period)
    event_times = np.arange(WINDOW[0], WINDOW[1] + 1)
    theta = _event_time_effects(att, cohorts, weights, event_times, keep_cohorts)

    est, draws = theta[0], theta[1:]
    with warnings.catch_warnings():               # event time -1 has no draws (base year)
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.nanstd(draws, axis=0, ddof=1)
    table = pd.DataFrame({"event_time": event_times, "estimate": est, "se": se})
    if base_period == "universal":                                   # the base year itself
        table.loc[table["event_time"] == -1, ["estimate", "se"]] = 0.0
    table = table.dropna(subset=["estimate"]).reset_index(drop=True)
    att_table = pd.DataFrame([{"cohort": g, "year": t, "att": a[0]} for (g, t), a in att.items()])
    return table, CSResult(event_times, est, draws, att_table)


def average_effect(res, first, last):
    """Average of theta(e) over event times first..last (inclusive), with bootstrap s.e."""
    keep = (res.event_times >= first) & (res.event_times <= last)
    point = np.nanmean(res.estimate[keep])
    boot = np.nanmean(res.draws[:, keep], axis=1)
    return float(point), float(np.nanstd(boot, ddof=1))


def window_difference(res, window, baseline):
    """Average effect over `window` minus average over `baseline` (both (first, last)),
    e.g. years 0-2 compared with years -5 to -2 instead of with year -1 alone."""
    def avg(x, w):
        keep = (res.event_times >= w[0]) & (res.event_times <= w[1])
        return np.nanmean(x[..., keep], axis=-1)
    point = avg(res.estimate, window) - avg(res.estimate, baseline)
    boot = avg(res.draws, window) - avg(res.draws, baseline)
    return float(point), float(np.nanstd(boot, ddof=1))


def difference_between(res_a, res_b, window):
    """Average effect over `window` in res_a minus res_b. Valid when both come from the
    same sample `d` (same states, so the bootstrap samples line up draw by draw)."""
    keep = (res_a.event_times >= window[0]) & (res_a.event_times <= window[1])
    point = np.nanmean(res_a.estimate[keep]) - np.nanmean(res_b.estimate[keep])
    boot = np.nanmean(res_a.draws[:, keep], axis=1) - np.nanmean(res_b.draws[:, keep], axis=1)
    return float(point), float(np.nanstd(boot, ddof=1))


def pretrend_test(res, first=WINDOW[0], last=-2):
    """Wald test that theta(first..last) are all zero, bootstrap covariance. Returns p."""
    keep = (res.event_times >= first) & (res.event_times <= last)
    b = res.estimate[keep]
    draws = res.draws[:, keep]
    draws = draws[np.isfinite(draws).all(axis=1)]
    V = np.cov(draws, rowvar=False)
    stat = float(b @ np.linalg.solve(V, b))
    return stats.chi2.sf(stat, df=len(b))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_event_study(series, title, ylabel, path, note):
    """series: {legend label: DataFrame with event_time, estimate, se}.
    Pass title="" and note="" for paper figures (the caption carries them)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.axvline(-0.5, color="#898781", linewidth=1, linestyle="--")
    offsets = np.linspace(-0.15, 0.15, len(series)) if len(series) > 1 else [0.0]
    for i, (name, res) in enumerate(series.items()):
        ax.errorbar(res["event_time"] + offsets[i], res["estimate"], yerr=1.96 * res["se"],
                    fmt=MARKERS[i], color=COLORS[i], markersize=5, elinewidth=1, capsize=0,
                    label=name)
    ax.set_xlabel("Years since the state's R&D tax credit started")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(WINDOW[0], WINDOW[1] + 1, 5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
