# Do the Effects of State R&D Tax Credits Build Up over Time?

Code and data construction for the paper *Do the Effects of State R&D Tax Credits Build Up over
Time? Evidence from U.S. Patenting, 1976–2019* (Bingrui Wang, 2026). The paper is in
[`paper/main.pdf`](paper/main.pdf).

**Question.** U.S. states introduced R&D tax credits at different times between 1982 and 2012. Did
the credits raise patenting by inventors in the adopting state, and did any effect grow or fade the
longer a credit stayed in force?

**Design.** Event studies around each state's adoption year, using the Callaway and Sant'Anna (2021)
estimator (compared with a two-way fixed effects event study), for 23 adopting and 9
never-adopting states, 1976–2019. Patents are all U.S. utility patents, counted fractionally by
inventor state and by filing year.

**Finding.** No detectable effect at any horizon. An apparent short-run increase measured against the
year before adoption disappears when the post-adoption years are compared with earlier pre-adoption
years, and there is no sign that effects build up over twenty years.

## Where to start reading

| File | Why |
|---|---|
| [`code/event_study_tools.py`](code/event_study_tools.py) | The estimators: two-way fixed effects event study, the Callaway–Sant'Anna estimator written out from its formulas, a state-cluster bootstrap, and pre-trend tests. Shared by every analysis script. |
| [`code/03_build_patents_state_year.py`](code/03_build_patents_state_year.py) | Turns 22 million inventor records into patent counts by state and filing year, reading the files in chunks, with checks against published counts. |
| [`code/02_build_policy_panel.py`](code/02_build_policy_panel.py) | Codes when each state's credit started and ended from two sources. All judgment calls are written down as five rules in one place and checked against what the sources report. |

## Layout

```
code/                       analysis scripts, run in numbered order
  01_download_data.py       download the public raw files (logged with SHA-256 hashes)
  02_build_policy_panel.py  credit start and end years for 50 states + DC
  03_build_patents_state_year.py   patents by state and filing year
  04_build_analysis_panel.py       merge into one state-year panel
  05_event_study.py         main event studies (patents; business R&D)
  06_firm_size_extension.py patents split by the size of the patent owner
  07_robustness.py          robustness checks
  08_paper_outputs.py       all tables (LaTeX) and figures used in the paper
  event_study_tools.py      shared estimation functions
  check_cs_estimator.py     optional: compare the estimator with the csdid package
  run_all.py                run the steps in order and log each one
data/
  raw/                      raw files, never edited (only the hand-coded policy files are included)
  clean/                    cleaned panels written by scripts 02-06 (included)
output/
  tables/                   CSV results; paper_numbers.csv lists every number quoted in the paper
  figures/                  figures
  logs/                     printed output of each step, including its checks
paper/                      LaTeX source of the paper, its tables and figures, and the PDF
```

## How to reproduce

1. Python 3.10 or later: `pip install -r requirements.txt`
2. Raw data: run `python3 code/01_download_data.py` for the public files, then download five
   USPTO PatentsView files by hand (a free USPTO.gov account is required). Exact files, release and
   links are in [`data/raw/README_data_sources.md`](data/raw/README_data_sources.md).
3. From the project folder: `python3 code/run_all.py`. This runs steps 02–08 in about four minutes
   and writes each step's output to `output/logs/`.

The cleaned panels in `data/clean/` are included, so the estimation steps can be rerun without
downloading the 1.4 GB of raw patent files: `python3 code/run_all.py 05 07 08`. (Step 06 rebuilds the
owner-size counts from the raw assignee file.)

## Where each result in the paper comes from

| Paper | Produced by | File |
|---|---|---|
| Table 1, descriptive statistics | `08_paper_outputs.py` | `paper/tables/table1_descriptives.tex` |
| Table 2, main estimates by horizon | `08_paper_outputs.py` | `paper/tables/table2_main.tex` |
| Table 3, robustness | `07_robustness.py` (formatted by 08) | `output/tables/robustness_patents.csv` |
| Table 4, heterogeneity | `06_firm_size_extension.py` (panel A), `08_paper_outputs.py` (panel B) | `paper/tables/table4_heterogeneity.tex` |
| Figure 1, adoption timeline | `08_paper_outputs.py` | `paper/figures/fig1_adoption_timeline.pdf` |
| Figure 2, event study for patents | `05_event_study.py`, `08_paper_outputs.py` | `paper/figures/fig2_event_study_patents.pdf` |
| Figure 3, event study by owner size | `06_firm_size_extension.py`, `08_paper_outputs.py` | `paper/figures/fig3_event_study_by_size.pdf` |
| Tables A1–A2, policy coding and business R&D | `08_paper_outputs.py` | `paper/tables/tableA1_policy.tex`, `tableA2_rd.tex` |
| Every number quoted in the text | `08_paper_outputs.py` | `output/tables/paper_numbers.csv` |

## Checks built into the scripts

Each script prints checks that compare what it built with an outside benchmark, and stops (`assert`)
where a failure would mean a bug:

- **02**: of the 32 states Wilson (2009) lists for 2006, 31 have an active credit under the coding (the
  exception, Missouri, had expired according to the Iowa source); rebuilding the Iowa study's count of
  states with a credit in each year from its program dates matches in 27 of 41 years, and the log
  explains the differences.
- **03**: drops reissue patents mislabelled as utility patents, sets impossible filing dates to
  missing, and compares state totals by grant year with the counts published by NCSES (correlation
  1.000 across 1,479 state-years).
- **04**: every state-year has population and patent data; state business R&D sums to 99.8% of the
  U.S. total.
- **06**: the largest owners are familiar large firms, and the four size groups add up exactly to the
  total count in every state-year.
- **Estimator**: `check_cs_estimator.py` compares every group-time effect with the `csdid` package.
  All 774 cohort-year cells agree to within 1e-15 for both comparison groups and both base-period
  choices.

## A note on the estimator

The first version of the analysis used the Python package `moderndid` (version 0.0.3). Its results
agree with `csdid` and with a direct calculation when the comparison group is never-treated states,
but not when it is not-yet-treated states, the comparison group used in the paper (differences of up to
1.1 log points in individual cells, and 35 missing cells). The estimator in `event_study_tools.py` is
therefore written out from the formulas in Callaway and Sant'Anna (2021) and checked against `csdid`.
`check_cs_estimator.py` reproduces the comparison.

## Conventions

- Raw files are never edited; every cleaning step is a script.
- Judgment calls are made in one place and fixed before estimation: the policy coding rules
  (script 02), the owner-size thresholds (script 06), and the split by credit value (script 08).
- Paths are relative to the project folder, and the bootstrap uses a fixed seed, so reruns give
  identical numbers.
