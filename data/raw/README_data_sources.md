# Raw data sources

Raw files in this folder are never edited. Cleaning happens in the `code/` scripts, which read from
here and write to `data/clean/`. Downloads are logged in `download_log.csv` (URL, size, SHA-256
hash, date), so anyone can check that they are using the same files.

Only the small hand-coded files in `policy/` are included in the repository. Everything else is
public but large, or someone else's publication, and has to be downloaded as described below.

## A. Downloaded by `code/01_download_data.py`

| Saved as (under `data/raw/`) | What it is | Used in | Source URL |
|---|---|---|---|
| `rd_spending/ncses_rdperformed_business.json` | Business-performed R&D by state, $ millions, 1997–2023 | 04 (appendix outcome) | https://ncsesdata.nsf.gov/indicators/states/api/v2/dataseries/rdperformed-business.json |
| `rd_spending/ncses_privateindustryoutput.json` | Private-industry output by state, $ millions | 04 | https://ncsesdata.nsf.gov/indicators/states/api/v2/dataseries/privateindustryoutput.json |
| `patents/ncses_patents_all.json` | Patents by state (NCSES tabulation) | 03 (check only) | https://ncsesdata.nsf.gov/indicators/states/api/v2/dataseries/patents-all.json |
| `controls/bea_SAINC.zip` | BEA state personal income; table SAINC1 gives population | 04 | https://apps.bea.gov/regional/zip/SAINC.zip |
| `policy/wilson_frbsf_wp2005-08.pdf` | Wilson, "Beggar Thy Neighbor?" working paper; Table 1 lists each state's credit, year enacted, base and rates as of 2006 | hand-coded (C) | https://www.frbsf.org/wp-content/uploads/wp05-08bk.pdf |
| `policy/iowa_dor_research_credit_study_2021.pdf` | Iowa Department of Revenue (2021), Research Activities Tax Credit evaluation study; Table 1 lists state programs with first and sunset years, Table 2 counts states with a credit by year | hand-coded (C) | https://revenue.iowa.gov/media/3287/download |

## B. Downloaded by hand (USPTO.gov sign-in required)

USPTO PatentsView, Granted Patent Disambiguated Data (product PVGPATDIS), release 2026-04-10,
covering patents granted 1976-01-01 to 2025-12-31. Page: https://data.uspto.gov/bulkdata/datasets/pvgpatdis
(PatentsView moved to the USPTO Open Data Portal in 2026; downloads need a free USPTO.gov account).
Save these five files, unchanged, in `data/raw/patents/`:

| File | Used for |
|---|---|
| `g_patent.tsv.zip` | patent type and grant date |
| `g_application.tsv.zip` | filing date |
| `g_inventor_disambiguated.tsv.zip` | inventors of each patent and their `location_id` |
| `g_location_disambiguated.tsv.zip` | `location_id` to state FIPS code |
| `g_assignee_disambiguated.tsv.zip` | first-listed owner of each patent and its type (script 06) |

Links between files: `patent_id` connects all patent tables; `location_id` connects inventors to locations.

## C. Hand-coded from the two policy documents (included in the repository)

| File | Contents |
|---|---|
| `policy/state_rd_credit_sources_handcoded.csv` | One row per state (50 + DC). Wilson Table 1: year enacted, statutory rate, base, effective rate. Iowa Table 1: first tax year, sunset year, rate text. A note where the sources differ. Transcribed as printed, with no judgment calls; the coding rules are applied in `code/02_build_policy_panel.py`. |
| `policy/iowa_table2_state_counts.csv` | Iowa Table 2: number of states with a credit in each tax year, 1981–2021 (used only as a check in script 02). |

## Download notes

| Date | Files | Notes |
|---|---|---|
| 2026-11-02 | Section A files | Downloaded with a browser from the same URLs as the script; all pass the script's file check. Hashes in `download_log.csv`. |
| 2026-11-02 | `g_patent`, `g_inventor_disambiguated`, `g_location_disambiguated` | PVGPATDIS release 2026-04-10. All pass `unzip -t`. |
| 2026-11-03 | `g_application` | Same release. |
| 2026-11-20 | `g_assignee_disambiguated` | Same release. |

Files from an earlier version of the project that are no longer used (`controls/bea_SAGDP*.zip`,
`rd_spending/ncses_rdperformed_all.json`, `rd_spending/ncses_indicator_business_rd_share_of_output.xlsx`)
may still be present locally; no script reads them.
