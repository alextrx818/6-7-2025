#!/usr/bin/env python3
"""
test_pipeline_flow.py

An integration test / watcher that starts the pipeline via start.sh,
then monitors step1.json, step27.json, and step7_matches.log to verify
data is flowing from Step 1 → Step 2 → Step 7 in each 60 second cycle.
Finally, it stops the pipeline via start.sh stop.

Usage:
    ./test_pipeline_flow.py
"""

import subprocess
import time
import os
import json
import sys
from datetime import datetime

# ————— CONFIG —————
STEP1_JSON   = "step1.json"
STEP2_JSON   = "step27.json"
STEP7_LOG    = "step7_matches.log"
PID_FILE     = "step1.pid"
START_SCRIPT = "./start.sh"
TIMEOUT_SEC  =  sixty = 60  # how long to wait (in seconds) for each file to appear/update
# ——————————————————

def now():
    """Return current time string for logging."""
    return datetime.now().strftime("%H:%M:%S")


def wait_for_file(path, timeout=TIMEOUT_SEC):
    """
    Block until `path` exists or until `timeout` seconds elapse.
    Returns True if the file appeared, False on timeout.
    """
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            return False
        time.sleep(1)
    return True


def main():
    # 0) Clean up any stale PID from previous runs
    if os.path.exists(PID_FILE):
        print(f"{now()} | Removing stale PID file '{PID_FILE}' before starting.")
        os.remove(PID_FILE)

    # 1) Launch the pipeline (Step 1 → Step 2 → Step 7, plus alert modules)
    print(f"{now()} | ▶ Running `{START_SCRIPT} start` to launch the entire pipeline.")
    ret = subprocess.run([START_SCRIPT, "start"])
    if ret.returncode != 0:
        print(f"{now()} | ✗ start.sh returned exit code {ret.returncode}. Aborting.")
        sys.exit(1)

    # 2) Wait for step1.json to be created
    print(f"{now()} | Waiting up to {TIMEOUT_SEC}s for '{STEP1_JSON}' to appear…")
    if not wait_for_file(STEP1_JSON, timeout=TIMEOUT_SEC):
        print(f"{now()} | ✗ Timeout: '{STEP1_JSON}' did not appear within {TIMEOUT_SEC}s.")
        subprocess.run([START_SCRIPT, "stop"])
        sys.exit(1)

    # 2a) Inspect step1.json
    t1 = time.time()
    print(f"{now()} | ✓ '{STEP1_JSON}' appeared after {t1:.1f}s.")
    try:
        with open(STEP1_JSON, "r", encoding="utf-8") as f:
            data1 = json.load(f)
    except Exception as e:
        print(f"{now()} | ✗ Failed to parse '{STEP1_JSON}': {e}")
        subprocess.run([START_SCRIPT, "stop"])
        sys.exit(1)

    top_keys = list(data1.keys())
    print(f"{now()} | '{STEP1_JSON}' top-level keys: {top_keys}")

    # 3) Wait for step27.json (Step 2 output)
    print(f"{now()} | Waiting up to {TIMEOUT_SEC}s for '{STEP2_JSON}' to appear…")
    t2_start = time.time()
    if not wait_for_file(STEP2_JSON, timeout=TIMEOUT_SEC):
        print(f"{now()} | ✗ Timeout: '{STEP2_JSON}' did not appear within {TIMEOUT_SEC}s.")
        subprocess.run([START_SCRIPT, "stop"])
        sys.exit(1)

    t2 = time.time()
    print(f"{now()} | ✓ '{STEP2_JSON}' appeared after {t2 - t2_start:.1f}s.")

    # 3a) Inspect step27.json
    try:
        with open(STEP2_JSON, "r", encoding="utf-8") as f:
            data2 = json.load(f)
    except Exception as e:
        print(f"{now()} | ✗ Failed to parse '{STEP2_JSON}': {e}")
        subprocess.run([START_SCRIPT, "stop"])
        sys.exit(1)

    # The "history" key should exist, with at least one batch
    if "history" not in data2 or not data2["history"]:
        print(f"{now()} | ✗ '{STEP2_JSON}' has no 'history' or it's empty.")
    else:
        last_batch = data2["history"][-1]
        matches_dict = last_batch.get("matches", {})
        print(f"{now()} | '{STEP2_JSON}' keys: {list(data2.keys())}")
        print(f"{now()} | Latest batch has {len(matches_dict)} flattened matches.")

    # 4) Wait for step7_matches.log to be created and get its last few lines
    print(f"{now()} | Waiting up to {TIMEOUT_SEC}s for '{STEP7_LOG}' to appear…")
    t3_start = time.time()
    if not wait_for_file(STEP7_LOG, timeout=TIMEOUT_SEC):
        print(f"{now()} | ✗ Timeout: '{STEP7_LOG}' did not appear within {TIMEOUT_SEC}s.")
    else:
        t3 = time.time()
        print(f"{now()} | ✓ '{STEP7_LOG}' appeared after {t3 - t3_start:.1f}s.")
        # Give it a couple seconds to flush a few lines
        time.sleep(2)
        try:
            with open(STEP7_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
                tail = lines[-5:] if len(lines) >= 5 else lines
        except Exception as e:
            print(f"{now()} | ✗ Could not read '{STEP7_LOG}': {e}")
            tail = []

        print(f"{now()} | Last lines of '{STEP7_LOG}':")
        for L in tail:
            print("   " + L.rstrip())

    # 5) Verify a second cycle by seeing if step1.json's mtime changes
    print(f"{now()} | Checking for a second cycle (step1.json modification)…")
    try:
        prev_mtime = os.path.getmtime(STEP1_JSON)
    except OSError:
        prev_mtime = None

    t4_start = time.time()
    second_cycle_happened = False
    while time.time() - t4_start < 2 * TIMEOUT_SEC:
        try:
            curr_mtime = os.path.getmtime(STEP1_JSON)
        except OSError:
            curr_mtime = None

        if prev_mtime is not None and curr_mtime is not None and curr_mtime != prev_mtime:
            print(f"{now()} | ✓ Detected that '{STEP1_JSON}' was updated again (second cycle).")
            second_cycle_happened = True
            break

        time.sleep(1)

    if not second_cycle_happened:
        print(f"{now()} | ⚠ Did not see a second cycle within {2*TIMEOUT_SEC}s.")

    # 6) Finally, shut down the entire pipeline
    print(f"{now()} | ▶ Running `{START_SCRIPT} stop` to shut everything down.")
    subprocess.run([START_SCRIPT, "stop"])

    # Optional: Give a moment to let all processes die, then confirm no PID file
    time.sleep(2)
    if os.path.exists(PID_FILE):
        print(f"{now()} | ⚠ PID file '{PID_FILE}' still exists—something didn't stop cleanly.")
    else:
        print(f"{now()} | ✓ All processes stopped, '{PID_FILE}' removed.")

    print(f"{now()} | TEST COMPLETE.")

if __name__ == "__main__":
    main()
