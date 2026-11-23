"""
08_paper_outputs.py
Tables and figures for the paper that scripts 05-07 do not produce, plus LaTeX versions
of every table. Run it after 05, 06 and 07.

Inputs
    data/clean/analysis_panel.csv                 (from 04)
    data/clean/policy_state.csv                   (from 02)
    data/clean/patents_by_size_state_year.csv     (from 06)
    output/tables/robustness_patents.csv          (from 07)
    output/tables/firm_size_summary.csv           (from 06)

Outputs
    paper/tables/table1_descriptives.tex     Table 1  descriptive statistics
    paper/tables/table2_main.tex             Table 2  effects on log patents by horizon
    paper/tables/table3_robustness.tex       Table 3  robustness checks
    paper/tables/table4_heterogeneity.tex    Table 4  owner size and credit generosity
    paper/tables/tableA1_policy.tex          Table A1 policy coding for all 51 states
    paper/tables/tableA2_rd.tex              Table A2 business R&D spending
    paper/figures/fig1_adoption_timeline.pdf   Figure 1 (PNG copy in output/figures/)
    paper/figures/fig2_event_study_patents.pdf Figure 2 (same estimates as script 05's figure)
    paper/figures/fig3_event_study_by_size.pdf Figure 3 (same estimates as script 06's figure)
    output/tables/paper_numbers.csv          every number quoted in the text, in one place

Credit generosity (fixed before estimating): Wilson (2009) Table 1 reports each credit's
effective rate, which accounts for how the base is defined. Credits whose base is a moving
average of the firm's own past R&D have effective rates below 1 percent ("low-value");
all other credits have effective rates of 2.5 percent or more ("high-value"). Florida and
Virginia are not in Wilson's table and are left out of this split.

Run from the project folder:  python3 code/08_paper_outputs.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from event_study_tools import (COLORS, WINDOW, average_effect, callaway_santanna,
                               difference_between, plot_event_study, prepare, pretrend_test,
                               twfe, window_difference)

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
PAPER_TABLES = ROOT / "paper" / "tables"
PAPER_FIGURES = ROOT / "paper" / "figures"

YEARS = (1976, 2019)
HORIZONS = [(0, 2), (3, 10), (11, 20), (0, 20)]
BASELINE = (-5, -2)            # pre-adoption years used as an alternative baseline
NUMBERS = []                   # (name, value) pairs quoted in the text


def keep(name, value, digits=3):
    NUMBERS.append({"name": name, "value": round(float(value), digits)})
    return value


def p_value(est, se):
    return 2 * stats.norm.sf(abs(est / se))


def cell(est, se, digits=3):
    """'0.035 (0.021)' with significance stars from a normal approximation."""
    p = p_value(est, se)
    star = "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""
    return f"${est:.{digits}f}{star}$", f"$({se:.{digits}f})$"


def write_tex(name, body):
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLES / name).write_text(body)


# ---------------------------------------------------------------------------
# Table 1: descriptive statistics
# ---------------------------------------------------------------------------
def table1(panel):
    d = prepare(panel, "ln_patents", YEARS, verbose=False)
    d["group"] = np.where(d["cohort"] == 0, "never",
                          np.where(d["year"] < d["start_year"], "before", "after"))
    d["pop_m"] = d["population"] / 1e6
    rd = d[d["year"] >= 1997]

    cols = ["before", "after", "never"]
    rows = []
    for label, var, data, fmt in [
            ("Patents per year (fractional count)", "patents_frac", d, "{:,.0f}"),
            ("Patents per 100,000 residents", "patents_per_100k", d, "{:.1f}"),
            ("Population (millions)", "pop_m", d, "{:.1f}"),
            ("Business R\\&D, \\$ millions (1997--2019)", "business_rd", rd, "{:,.0f}")]:
        means = [fmt.format(data.loc[data["group"] == c, var].mean()) for c in cols]
        sds = ["(" + fmt.format(data.loc[data["group"] == c, var].std()) + ")" for c in cols]
        rows.append(f"{label} & " + " & ".join(means) + r" \\")
        rows.append(" & " + " & ".join(sds) + r" \\[2pt]")

    shares = []
    for c in cols:
        sub = d[d["group"] == c]
        tot = sub[["patents_small", "patents_medium", "patents_large", "patents_none"]].sum()
        shares.append(tot / tot.sum())
    share_rows = []
    for g, label in [("small", "small owners ($<$50 patents)"),
                     ("medium", "medium owners (50--499)"),
                     ("large", "large owners (500+)"),
                     ("none", "no organization owner")]:
        vals = [f"{100 * s[f'patents_{g}']:.1f}" for s in shares]
        share_rows.append(f"\\quad {label} & " + " & ".join(vals) + r" \\")

    n_states = [d.loc[d["group"] == c, "sid"].nunique() for c in cols]
    n_obs = [int((d["group"] == c).sum()) for c in cols]
    body = r"""\begin{table}[t]
\centering
\caption{Descriptive statistics, estimation sample, 1976--2019}
\label{tab:descriptives}
\begin{threeparttable}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{2}{c}{Adopting states} & Never-adopting \\
\cmidrule(lr){2-3}
 & Before adoption & After adoption & states \\
\midrule
""" + "\n".join(rows) + r"""
Share of patents (\%) held by: & & & \\
""" + "\n".join(share_rows) + r"""
\midrule
States & """ + " & ".join(str(n) for n in n_states) + r""" \\
State-years & """ + " & ".join(f"{n:,}" for n in n_obs) + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Means across state-years, standard deviations in parentheses. Patents are
utility patents counted by the filing year and divided equally among inventors, so a patent with
inventors in two states adds one half to each. Owner size is the number of patents the first-listed
organization owner holds in the whole data. Business R\&D comes from NCSES and starts in 1997;
some state-years are suppressed. The before and after columns contain the same 23 states.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write_tex("table1_descriptives.tex", body)
    keep("n_adopting", n_states[0], 0)
    keep("n_never", n_states[2], 0)
    keep("n_state_years", len(d), 0)
    keep("n_cohorts", d.loc[d["cohort"] > 0, "cohort"].nunique(), 0)
    keep("share_large_all", 100 * d["patents_large"].sum() / d["patents_frac"].sum(), 1)
    keep("share_small_all", 100 * d["patents_small"].sum() / d["patents_frac"].sum(), 1)


# ---------------------------------------------------------------------------
# Table 2: main estimates by horizon
# ---------------------------------------------------------------------------
def table2(panel):
    d = prepare(panel, "ln_patents", YEARS, verbose=False)
    cs_nyt, res_nyt = callaway_santanna(d, "ln_patents", "notyettreated")
    cs_nev, res_nev = callaway_santanna(d, "ln_patents", "nevertreated")
    windows = HORIZONS + [BASELINE] + [(0, 2, *BASELINE)]
    tw_table, p_tw, tw = twfe(d, "ln_patents", windows=windows)

    columns = []
    for name, res in [("nyt", res_nyt), ("nev", res_nev)]:
        col = {h: average_effect(res, *h) for h in HORIZONS}
        col["baseline_level"] = average_effect(res, *BASELINE)
        col["adjusted"] = window_difference(res, (0, 2), BASELINE)
        col["pretrend_p"] = pretrend_test(res)
        columns.append(col)
    tw_col = {h: tw[h] for h in HORIZONS}
    tw_col["baseline_level"] = tw[BASELINE]
    tw_col["adjusted"] = tw[(0, 2, *BASELINE)]
    tw_col["pretrend_p"] = p_tw
    columns.append(tw_col)

    labels = {(0, 2): "Years 0--2", (3, 10): "Years 3--10", (11, 20): "Years 11--20",
              (0, 20): "Years 0--20 (all)",
              "baseline_level": "Years $-5$ to $-2$ (before adoption)",
              "adjusted": "Years 0--2 minus years $-5$ to $-2$"}
    lines = []
    for key in [(0, 2), (3, 10), (11, 20), (0, 20)]:
        ests, ses = zip(*[cell(*c[key]) for c in columns])
        lines.append(f"{labels[key]} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\[2pt]")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Compared with the years before adoption}} \\")
    for key in ["baseline_level", "adjusted"]:
        ests, ses = zip(*[cell(*c[key]) for c in columns])
        lines.append(f"{labels[key]} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\[2pt]")
    lines.append(r"\midrule")
    lines.append("Pre-trend test, $p$-value & " +
                 " & ".join(f"{c['pretrend_p']:.3f}" for c in columns) + r" \\")

    body = r"""\begin{table}[t]
\centering
\caption{Effect of adopting an R\&D tax credit on log patents}
\label{tab:main}
\begin{threeparttable}
\begin{tabular}{lccc}
\toprule
 & (1) & (2) & (3) \\
 & CS, not-yet-treated & CS, never-treated & TWFE \\
 & comparison & comparison & event study \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
Adopting / never-adopting states & 23 / 9 & 23 / 9 & 23 / 9 \\
State-years & 1,408 & 1,408 & 1,408 \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Each entry averages the event-time estimates over the years shown; year 0 is
the first year with a credit. All estimates measure log patents relative to the year before adoption
(year $-1$), except the last estimate row, which compares years 0--2 with the average of years $-5$ to $-2$.
Columns (1) and (2) use the estimator of \citet{callaway2021difference}; column (1) compares each
adoption cohort with states that never adopt or adopt later, column (2) with never-adopting states only.
Column (3) regresses log patents on event-time indicators with state and year fixed effects (endpoints
binned at $-10$ and $+20$). Standard errors clustered by state: bootstrap with 999 state resamples in
(1)--(2), analytic in (3). The pre-trend test asks whether the estimates for years $-10$ to $-2$ are jointly
zero. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write_tex("table2_main.tex", body)

    main = columns[0]
    for h in HORIZONS:
        keep(f"cs_{h[0]}_{h[1]}", main[h][0]); keep(f"cs_{h[0]}_{h[1]}_se", main[h][1])
        keep(f"cs_{h[0]}_{h[1]}_p", p_value(*main[h]))
        keep(f"cs_{h[0]}_{h[1]}_pct", 100 * (np.exp(main[h][0]) - 1), 1)
        keep(f"cs_{h[0]}_{h[1]}_lo", main[h][0] - 1.96 * main[h][1])
        keep(f"cs_{h[0]}_{h[1]}_hi", main[h][0] + 1.96 * main[h][1])
    keep("cs_baseline_level", main["baseline_level"][0])
    keep("cs_baseline_level_se", main["baseline_level"][1])
    keep("cs_baseline_level_p", p_value(*main["baseline_level"]))
    keep("cs_adjusted", main["adjusted"][0]); keep("cs_adjusted_se", main["adjusted"][1])
    keep("cs_adjusted_hi", main["adjusted"][0] + 1.96 * main["adjusted"][1])
    keep("cs_pretrend_p", main["pretrend_p"])
    keep("twfe_pretrend_p", p_tw)
    keep("cs_nev_0_2", columns[1][(0, 2)][0]); keep("cs_nev_0_2_se", columns[1][(0, 2)][1])
    keep("twfe_0_2", tw[(0, 2)][0]); keep("twfe_0_2_se", tw[(0, 2)][1])
    keep("twfe_0_20", tw[(0, 20)][0]); keep("twfe_0_20_se", tw[(0, 20)][1])
    pre = cs_nyt[(cs_nyt["event_time"] < -1)]
    keep("cs_pre_positive", (pre["estimate"] > 0).sum(), 0)
    keep("cs_pre_count", len(pre), 0)
    for e in [0, 1, 2]:
        keep(f"cs_e{e}", cs_nyt.loc[cs_nyt["event_time"] == e, "estimate"].iloc[0])
    return cs_nyt, tw_table


# ---------------------------------------------------------------------------
# Table 3: robustness (from 07)
# ---------------------------------------------------------------------------
def table3():
    r = pd.read_csv(TABLES / "robustness_patents.csv")
    names = {"main": "Main specification",
             "with_repealers": "Keep 4 states that later repealed",
             "iowa_years": "Iowa start years for AZ, CA, IN, MN",
             "never_treated": "Never-adopting comparison only",
             "per_capita": "Log patents per 100,000 residents",
             "drop_california": "Drop California"}
    lines = []
    for _, row in r.iterrows():
        e1, s1 = cell(row["effect_years_0_2"], row["se_0_2"])
        e2, s2 = cell(row["effect_years_0_20"], row["se_0_20"])
        lines.append(f"{names[row['specification']]} & {int(row['adopting_states'])} / "
                     f"{int(row['never_adopting_states'])} & {e1} & {s1} & {e2} & {s2} & "
                     f"{row['pretrend_p']:.3f} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Robustness of the patent estimates}
\label{tab:robustness}
\small
\begin{threeparttable}
\begin{tabular}{lcr@{\,}lr@{\,}lc}
\toprule
 & States & \multicolumn{2}{c}{Years 0--2} & \multicolumn{2}{c}{Years 0--20} & Pre-trend \\
 & (adopt / never) & \multicolumn{2}{c}{} & \multicolumn{2}{c}{} & $p$-value \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Each row changes one choice from the main specification and re-estimates the
Callaway--Sant'Anna event study. Entries are average effects on log patents relative to the year before
adoption, with bootstrap standard errors clustered by state (999 resamples) in parentheses.
The main specification is column (1) of Table 2. Arizona, California, Indiana and Minnesota are the four
states whose two sources differ by one year. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write_tex("table3_robustness.tex", body)
    lo = r["effect_years_0_2"].min(); hi = r["effect_years_0_2"].max()
    keep("robust_0_2_min", lo); keep("robust_0_2_max", hi)
    keep("robust_0_20_min", r["effect_years_0_20"].min())
    keep("robust_0_20_max", r["effect_years_0_20"].max())


# ---------------------------------------------------------------------------
# Table 4: heterogeneity by owner size (from 06) and by credit generosity
# ---------------------------------------------------------------------------
def generosity(panel, policy):
    d = prepare(panel, "ln_patents", YEARS, verbose=False)
    rates = policy.set_index("state_abbr")["wilson_effective_rate"]
    adopters = d.loc[d["cohort"] > 0, ["state_abbr", "cohort"]].drop_duplicates()
    adopters["rate"] = adopters["state_abbr"].map(rates)
    adopters = adopters.dropna(subset=["rate"])                    # FL, VA not in Wilson
    adopters["value"] = np.where(adopters["rate"] < 1, "low", "high")
    # every adoption cohort must be all-high or all-low, so cohorts can be split cleanly
    assert adopters.groupby("cohort")["value"].nunique().max() == 1
    out = {}
    for v in ["high", "low"]:
        cohorts = set(adopters.loc[adopters["value"] == v, "cohort"])
        _, res = callaway_santanna(d, "ln_patents", keep_cohorts=cohorts)
        out[v] = {"res": res, "states": ", ".join(sorted(adopters.loc[adopters["value"] == v, "state_abbr"])),
                  "n": int((adopters["value"] == v).sum()),
                  "early": average_effect(res, 0, 2), "late": average_effect(res, 0, WINDOW[1]),
                  "pre_p": pretrend_test(res)}
    out["diff"] = {"early": difference_between(out["high"]["res"], out["low"]["res"], (0, 2)),
                   "late": difference_between(out["high"]["res"], out["low"]["res"], (0, WINDOW[1]))}
    return out


def table4(gen):
    fs = pd.read_csv(TABLES / "firm_size_summary.csv").set_index("group")
    lines = [r"\multicolumn{6}{l}{\textit{A. By size of the patent owner; outcome $\log(1+\text{patents})$}} \\"]
    for g, label in [("small", "Small owners ($<$50 patents)"), ("large", "Large owners (500+)"),
                     ("none", "No organization owner"), ("small_minus_large", "Small minus large")]:
        e1, s1 = cell(fs.loc[g, "effect_years_0_2"], fs.loc[g, "se_0_2"])
        e2, s2 = cell(fs.loc[g, "effect_years_0_20"], fs.loc[g, "se_0_20"])
        lines.append(f"\\quad {label} & {e1} & {s1} & {e2} & {s2} & {fs.loc[g, 'pretrend_p']:.3f} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{B. By effective value of the credit; outcome log patents}} \\")
    for v, label in [("high", f"High-value credits ({gen['high']['n']} states)"),
                     ("low", f"Low-value credits ({gen['low']['n']} states)")]:
        e1, s1 = cell(*gen[v]["early"]); e2, s2 = cell(*gen[v]["late"])
        lines.append(f"\\quad {label} & {e1} & {s1} & {e2} & {s2} & {gen[v]['pre_p']:.3f} \\\\")
    e1, s1 = cell(*gen["diff"]["early"]); e2, s2 = cell(*gen["diff"]["late"])
    lines.append(f"\\quad High minus low & {e1} & {s1} & {e2} & {s2} & \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Heterogeneity: owner size and credit value}
\label{tab:heterogeneity}
\begin{threeparttable}
\begin{tabular}{lr@{\,}lr@{\,}lc}
\toprule
 & \multicolumn{2}{c}{Years 0--2} & \multicolumn{2}{c}{Years 0--20} & Pre-trend $p$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Callaway--Sant'Anna estimates, not-yet-treated comparison, 23 adopting and 9
never-adopting states, 1976--2019. Panel A splits each state's patents by the size of the first-listed
organization owner (its total patents in the whole data); the difference row re-estimates the model on
the gap $\log(1+\text{small})-\log(1+\text{large})$. Panel B averages only the adoption cohorts whose
credit has a low effective rate (below 1 percent; all have a moving-average base: """ + gen["low"]["states"] + r""")
or a high one (2.5 percent or more: """ + gen["high"]["states"] + r"""), using the effective rates in
\citet{wilson2009beggar}; all states still serve as comparisons while they have not adopted.
Bootstrap standard errors clustered by state in parentheses. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write_tex("table4_heterogeneity.tex", body)
    for g in ["small", "large", "none", "small_minus_large"]:
        keep(f"size_{g}_0_2", fs.loc[g, "effect_years_0_2"]); keep(f"size_{g}_0_2_se", fs.loc[g, "se_0_2"])
        keep(f"size_{g}_0_20", fs.loc[g, "effect_years_0_20"]); keep(f"size_{g}_0_20_se", fs.loc[g, "se_0_20"])
        keep(f"size_{g}_pretrend_p", fs.loc[g, "pretrend_p"])
    for v in ["high", "low"]:
        keep(f"gen_{v}_0_2", gen[v]["early"][0]); keep(f"gen_{v}_0_2_se", gen[v]["early"][1])
        keep(f"gen_{v}_0_20", gen[v]["late"][0]); keep(f"gen_{v}_0_20_se", gen[v]["late"][1])
        keep(f"gen_{v}_pretrend_p", gen[v]["pre_p"]); keep(f"gen_{v}_n", gen[v]["n"], 0)
    keep("gen_diff_0_2", gen["diff"]["early"][0]); keep("gen_diff_0_2_se", gen["diff"]["early"][1])
    keep("gen_diff_0_2_p", p_value(*gen["diff"]["early"]))
    keep("gen_diff_0_20", gen["diff"]["late"][0]); keep("gen_diff_0_20_se", gen["diff"]["late"][1])


# ---------------------------------------------------------------------------
# Table A1: policy coding; Table A2: business R&D
# ---------------------------------------------------------------------------
def tableA1(policy):
    def yr(x):
        return "" if pd.isna(x) else str(int(x))
    status = {"": "", "nonstandard_credit": "Excluded: non-standard credit",
              "sources_disagree": "Excluded: sources disagree",
              "credit_switched_off_and_on": "Excluded: credit off and on"}
    lines = []
    for _, r in policy.sort_values("state").iterrows():
        reason = r["exclusion_reason"] if isinstance(r["exclusion_reason"], str) else ""
        if reason:
            s = status[reason]
        elif r["never_treated"]:
            s = "Never adopted"
        elif r["repealed"]:
            s = "Adopted, later repealed"
        else:
            s = "Adopted"
        rate = "" if pd.isna(r["wilson_effective_rate"]) else f"{r['wilson_effective_rate']:.1f}"
        lines.append(f"{r['state']} & {yr(r['wilson_year_enacted'])} & {rate} & "
                     f"{yr(r['iowa_initial_tax_year'])} & {yr(r['iowa_sunset'])} & "
                     f"{yr(r['start_year']) if not reason else ''} & "
                     f"{yr(r['end_year']) if not reason else ''} & {s} \\\\")
    body = r"""\begin{longtable}{lccccccl}
\caption{State R\&D tax credits: sources and coding}\label{tab:policy}\\
\toprule
 & \multicolumn{2}{c}{Wilson (2009)} & \multicolumn{2}{c}{Iowa DOR (2021)} & \multicolumn{2}{c}{Used here} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
State & Enacted & Eff.\ rate & First year & Sunset & Start & End & Status \\
\midrule
\endfirsthead
\toprule
State & Enacted & Eff.\ rate & First year & Sunset & Start & End & Status \\
\midrule
\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{8}{p{0.95\textwidth}}{\footnotesize \textit{Notes:} Wilson (2009) lists the 32 states with a
general credit on company-funded R\&D as of 2006, with the year enacted and the effective credit rate
(percent). Iowa Department of Revenue (2021) lists each state's current program with its first tax year and
sunset year. The start year is Wilson's year where available and Iowa's year for credits created after
2006; the end year is an Iowa sunset that had already passed by 2020. States are excluded when Iowa
shows a pre-2006 credit that Wilson, who covers only general credits, leaves out,
when the two sources differ by more than one year, or when the credit was switched off and on.}\\
\end{longtable}
"""
    write_tex("tableA1_policy.tex", body)


def tableA2(panel):
    d = prepare(panel, "ln_business_rd", (1997, 2023), verbose=False)
    _, res = callaway_santanna(d, "ln_business_rd")
    _, p_tw, tw = twfe(d, "ln_business_rd", windows=[(0, 2), (0, 20)])
    n_t = d.loc[d["cohort"] > 0, "sid"].nunique(); n_n = d.loc[d["cohort"] == 0, "sid"].nunique()
    rows = []
    for label, key in [("Years 0--2", (0, 2)), ("Years 0--20", (0, 20))]:
        a, b = cell(*average_effect(res, *key)); c, e = cell(*tw[key])
        rows.append(f"{label} & {a} & {c} \\\\")
        rows.append(f" & {b} & {e} \\\\[2pt]")
    rows.append(f"Pre-trend test, $p$-value & {pretrend_test(res):.3f} & {p_tw:.3f} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Business R\&D spending (appendix)}
\label{tab:rd}
\begin{threeparttable}
\begin{tabular}{lcc}
\toprule
 & CS, not-yet-treated & TWFE \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Adopting / never-adopting states & """ + f"{n_t} / {n_n} & {n_t} / {n_n}" + r""" \\
State-years & """ + f"{len(d):,} & {len(d):,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Outcome: log business-performed R\&D (NCSES), 1997--2023. Only states that adopted
after 1997 and have no suppressed years can contribute, which leaves few states, and the pre-trend test
rejects. These estimates are reported for completeness and are not used for inference in the text.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write_tex("tableA2_rd.tex", body)
    keep("rd_n_adopting", n_t, 0); keep("rd_n_never", n_n, 0)
    keep("rd_pretrend_p", pretrend_test(res))


# ---------------------------------------------------------------------------
# Figure 1: adoption timeline
# ---------------------------------------------------------------------------
def figure1(policy):
    p = policy[policy["main_sample"]].copy()
    p = p.sort_values(["start_year", "state_abbr"], na_position="last").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 7.2))
    for i, r in p.iterrows():
        y = len(p) - i
        if pd.isna(r["start_year"]):
            ax.text(1977, y, "never adopted", va="center", fontsize=7, color="#52514e")
            continue
        end = r["end_year"] if pd.notna(r["end_year"]) else 2019
        color = COLORS[1] if pd.notna(r["end_year"]) else COLORS[0]
        ax.barh(y, end - r["start_year"] + 1, left=r["start_year"] - 0.5, height=0.62,
                color=color, edgecolor="none")
    ax.set_yticks(range(len(p), 0, -1))
    ax.set_yticklabels(p["state_abbr"], fontsize=7)
    ax.set_xlim(1976, 2020.5)
    ax.set_ylim(0.3, len(p) + 0.7)
    ax.axvline(2019.5, color="#898781", linewidth=1, linestyle="--")
    ax.set_xlabel("Year")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[0]), plt.Rectangle((0, 0), 1, 1, color=COLORS[1])]
    ax.legend(handles, ["Credit in force (estimation sample)", "Credit later repealed (robustness only)"],
              frameon=False, loc="lower right", fontsize=8)
    ax.text(2019.3, len(p) + 0.4, "last filing year used", ha="right", fontsize=7, color="#52514e")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "adoption_timeline.png", dpi=200)
    fig.savefig(PAPER_FIGURES / "fig1_adoption_timeline.pdf")
    plt.close(fig)


def figures_2_3(cs_nyt, tw_table):
    """Paper versions of the event-study figures: no title or note (the caption has them)."""
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    plot_event_study({"Two-way fixed effects": tw_table, "Callaway-Sant'Anna": cs_nyt}, "",
                     "Effect on log patents", PAPER_FIGURES / "fig2_event_study_patents.pdf", "")
    by_size = pd.read_csv(TABLES / "event_study_by_size.csv")
    series = {f"{g.capitalize()} owners": by_size[by_size["group"] == g] for g in ["small", "large"]}
    plot_event_study(series, "", "Effect on log(1 + patents)",
                     PAPER_FIGURES / "fig3_event_study_by_size.pdf", "")


def main():
    panel = pd.read_csv(CLEAN / "analysis_panel.csv", dtype={"state_fips": str})
    sizes = pd.read_csv(CLEAN / "patents_by_size_state_year.csv", dtype={"state_fips": str})
    panel = panel.merge(sizes, on=["state_fips", "year"], how="left")
    policy = pd.read_csv(CLEAN / "policy_state.csv", dtype={"state_fips": str})

    table1(panel)
    cs_nyt, tw_table = table2(panel)
    figures_2_3(cs_nyt, tw_table)
    table3()
    gen = generosity(panel, policy)
    table4(gen)
    tableA1(policy)
    tableA2(panel)
    figure1(policy)

    numbers = pd.DataFrame(NUMBERS)
    numbers.to_csv(TABLES / "paper_numbers.csv", index=False)
    print(numbers.to_string(index=False))
    print("\nSaved LaTeX tables to paper/tables/, Figure 1 to output/figures/, "
          "numbers to output/tables/paper_numbers.csv")


if __name__ == "__main__":
    main()
