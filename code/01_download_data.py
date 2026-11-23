"""
01_download_data.py
Download the raw data for the state R&D tax credit project.

Run from the project folder (the folder that contains code/ and data/):
    python3 code/01_download_data.py            # download anything not yet downloaded
    python3 code/01_download_data.py --force    # re-download everything

Uses only the Python standard library.
- Every file is saved under data/raw/ and is never edited afterwards;
  all cleaning happens in later scripts.
- Every download is checked (non-empty, valid JSON / ZIP / PDF) and recorded
  in data/raw/download_log.csv with its URL, size, SHA-256 hash and date,
  so anyone can verify they are using the same raw files.

The USPTO patent files need a USPTO.gov sign-in and cannot be scripted;
see data/raw/README_data_sources.md for how to get them.
"""

import csv
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LOG = RAW / "download_log.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; research data download script)"}
NCSES_API = "https://ncsesdata.nsf.gov/indicators/states/api/v2/dataseries/{}.json"

# Each entry: (URL, destination inside data/raw/, short description)
SOURCES = [
    # Business R&D spending by state (NCSES State Indicators API; appendix outcome)
    (NCSES_API.format("rdperformed-business"),
     "rd_spending/ncses_rdperformed_business.json",
     "Business-performed domestic R&D by state, $ millions, 1997-2023"),
    (NCSES_API.format("privateindustryoutput"),
     "rd_spending/ncses_privateindustryoutput.json",
     "Private-industry output by state, $ millions, 1997-2024"),
    # Patent counts by state published by NCSES: used only to check script 03
    (NCSES_API.format("patents-all"),
     "patents/ncses_patents_all.json",
     "Patents awarded by state, 1996-2024 (check only; main patent files come from USPTO)"),
    # Population: BEA regional accounts (table SAINC1)
    ("https://apps.bea.gov/regional/zip/SAINC.zip",
     "controls/bea_SAINC.zip",
     "BEA state personal income, population and employment"),
    # Policy: when each state adopted (and repealed) its R&D tax credit
    ("https://www.frbsf.org/wp-content/uploads/wp05-08bk.pdf",
     "policy/wilson_frbsf_wp2005-08.pdf",
     "Wilson working paper; Table 1 lists each state's credit start year and rate"),
    ("https://revenue.iowa.gov/media/3287/download",
     "policy/iowa_dor_research_credit_study_2021.pdf",
     "Iowa Dept. of Revenue 2021 study; Table 1 lists state credit start and sunset years"),
]


def download(url, dest, retries=3):
    """Download url to dest. Writes to a .part file first so a broken
    download never leaves a half-finished file under the real name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)  # 1 MB at a time
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(dest)
            return
        except Exception as err:
            if attempt == retries:
                raise
            print(f"    attempt {attempt} failed ({err}); retrying...")
            time.sleep(3 * attempt)


def check_file(path):
    """Return 'ok' if the file looks like what its extension says, else a short reason."""
    if path.stat().st_size == 0:
        return "empty file"
    if path.suffix == ".json":
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
        except ValueError:
            return "not valid JSON"
    elif path.suffix == ".zip":
        if not zipfile.is_zipfile(path):
            return "not a valid ZIP"
    elif path.suffix == ".pdf":
        with open(path, "rb") as f:
            if f.read(4) != b"%PDF":
                return "not a PDF"
    return "ok"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_download(url, rel_path, description, path, status):
    is_new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["downloaded_at", "file", "url", "bytes", "sha256", "check", "description"])
        writer.writerow([
            dt.datetime.now().isoformat(timespec="seconds"), rel_path, url,
            path.stat().st_size, sha256(path), status, description,
        ])


def main():
    force = "--force" in sys.argv
    problems = []

    for url, rel_path, description in SOURCES:
        dest = RAW / rel_path
        print(f"- {rel_path}")
        if dest.exists() and not force:
            print("    already downloaded; skipping (use --force to re-download)")
            continue
        try:
            download(url, dest)
        except Exception as err:
            print(f"    FAILED: {err}")
            problems.append((rel_path, url, str(err)))
            continue
        status = check_file(dest)
        print(f"    saved ({dest.stat().st_size / 1e6:.1f} MB), check: {status}")
        log_download(url, rel_path, description, dest, status)
        if status != "ok":
            # Rename the bad file so the next run tries again instead of skipping it.
            dest.replace(dest.with_name(dest.name + ".bad"))
            problems.append((rel_path, url, status))

    print()
    if problems:
        print(f"{len(problems)} file(s) need attention. Download them by hand in a browser")
        print("and save them under data/raw/ with the name shown:")
        for rel_path, url, reason in problems:
            print(f"  data/raw/{rel_path}\n      from {url}\n      reason: {reason}")
        sys.exit(1)
    print("All scripted downloads finished. Log: data/raw/download_log.csv")


if __name__ == "__main__":
    main()
