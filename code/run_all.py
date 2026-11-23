"""
run_all.py
Rebuild everything from the raw data, in order, and keep a log of each step.

    python3 code/run_all.py              # steps 02-08 (raw data already in data/raw/)
    python3 code/run_all.py 05 06        # only the steps named
    python3 code/run_all.py --download   # also step 01 (needs internet)
    python3 code/run_all.py --check      # also compare the estimator with csdid (optional package)

Each step's printed output, including its built-in checks, is saved to
output/logs/<step>.log. The run stops at the first step that fails.
Total time is about five minutes on a laptop; step 03 reads 1 GB of patent files
and takes about two minutes of that.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "output" / "logs"

STEPS = {
    "01": "01_download_data.py",          # raw data (not run by default)
    "02": "02_build_policy_panel.py",     # credit start and end years
    "03": "03_build_patents_state_year.py",   # patents by state and filing year
    "04": "04_build_analysis_panel.py",   # merge into one state-year panel
    "05": "05_event_study.py",            # main event studies
    "06": "06_firm_size_extension.py",    # patents by owner size
    "07": "07_robustness.py",             # robustness checks
    "08": "08_paper_outputs.py",          # tables and figures for the paper
    "check": "check_cs_estimator.py",     # optional cross-check (needs csdid)
}


def run(step):
    script = STEPS[step]
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{Path(script).stem}.log"
    print(f"[{step}] {script} ...", end=" ", flush=True)
    start = time.time()
    with open(log, "w") as f:
        result = subprocess.run([sys.executable, str(ROOT / "code" / script)], cwd=ROOT,
                                stdout=f, stderr=subprocess.STDOUT)
    print(f"{time.time() - start:5.0f} s  -> {log.relative_to(ROOT)}")
    if result.returncode != 0:
        sys.exit(f"Step {step} failed; see {log.relative_to(ROOT)}")


def main():
    args = sys.argv[1:]
    named = [a for a in args if not a.startswith("--")]
    steps = named or ["02", "03", "04", "05", "06", "07", "08"]
    if "--download" in args and not named:
        steps = ["01"] + steps
    if "--check" in args:
        steps = steps + ["check"]
    for step in steps:
        if step not in STEPS:
            sys.exit(f"Unknown step {step!r}; choose from {', '.join(STEPS)}")
        run(step)
    print("Done.")


if __name__ == "__main__":
    main()
