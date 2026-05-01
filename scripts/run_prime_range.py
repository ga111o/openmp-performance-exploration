#!/usr/bin/env python3
"""
Run ./prime for every 100th limit from 100 to 1,000,000 and save the results as CSV.
"""
from __future__ import annotations

import csv
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent

START = 1000
END = 10_000_000_000
STEP = 1000
TIMEOUT_SECONDS = 60.0
PROGRESS_EVERY = 1
CPU_CORES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

SOURCE = ROOT / "prime_once.c"
BINARY = ROOT / "prime"
OUT = ROOT / "results" / "prime_once_range.csv"

RESULT_LINE_RE = re.compile(r"^RESULT\s+(.*)$", re.MULTILINE)
KV_RE = re.compile(r"(\w+)=([^\s]+)")

CSV_COLUMNS = (
    "limit",
    "primes",
    "cycles",
    "seconds",
    "wall_seconds",
    "status",
    "stderr",
)

log = logging.getLogger("run_prime_range")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_binary() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"missing source: {SOURCE}")

    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-Wall",
        "-Wextra",
        str(SOURCE),
        "-o",
        str(BINARY),
    ]
    log.info("building %s", BINARY)
    subprocess.run(cmd, check=True)


def load_done_limits() -> set[int]:
    if not OUT.exists():
        return set()

    done: set[int] = set()
    with OUT.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                done.add(int(row["limit"]))
    return done


def open_writer():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    is_new = not OUT.exists() or OUT.stat().st_size == 0
    fh = OUT.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def iter_limits():
    return range(START, END + 1, STEP)


def core_for_limit(limit: int) -> int:
    return CPU_CORES[((limit - START) // STEP) % len(CPU_CORES)]


def run_one(limit: int) -> dict[str, str]:
    core = core_for_limit(limit)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            ["taskset", "-c", str(core), str(BINARY), str(limit)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "limit": str(limit),
            "wall_seconds": f"{time.perf_counter() - started:.9f}",
            "status": "timeout",
            "stderr": str(exc),
        }
    except OSError as exc:
        return {
            "limit": str(limit),
            "wall_seconds": f"{time.perf_counter() - started:.9f}",
            "status": "exception",
            "stderr": repr(exc),
        }

    wall_seconds = time.perf_counter() - started
    row = {
        "limit": str(limit),
        "wall_seconds": f"{wall_seconds:.9f}",
        "status": "ok",
        "stderr": proc.stderr.strip(),
    }

    if proc.returncode != 0:
        row["status"] = f"exit_{proc.returncode}"
        return row

    match = RESULT_LINE_RE.search(proc.stdout)
    if match is None:
        row["status"] = "no_result"
        row["stderr"] = (proc.stderr or proc.stdout).strip()
        return row

    row.update(KV_RE.findall(match.group(1)))
    return row


def main() -> int:
    configure_logging()
    build_binary()

    done = load_done_limits()
    total = len(iter_limits())
    skipped = sum(1 for n in iter_limits() if n in done)
    log.info("range=%d..%d step=%d total=%d skipped=%d cores=%s out=%s",
             START, END, STEP, total, skipped, ",".join(map(str, CPU_CORES)), OUT)

    started = time.perf_counter()
    completed = 0
    failures = 0

    fh, writer = open_writer()
    with fh:
        pending = set()
        limits = (n for n in iter_limits() if n not in done)

        with ThreadPoolExecutor(max_workers=len(CPU_CORES)) as executor:
            for limit in limits:
                pending.add(executor.submit(run_one, limit))

                if len(pending) < len(CPU_CORES):
                    continue

                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
                    completed += 1
                    if row.get("status") != "ok":
                        failures += 1

                    if completed % PROGRESS_EVERY == 0:
                        elapsed = time.perf_counter() - started
                        log.info(
                            "completed=%d/%d failures=%d elapsed=%.1fs",
                            completed + skipped,
                            total,
                            failures,
                            elapsed,
                        )
                fh.flush()

            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
                    completed += 1
                    if row.get("status") != "ok":
                        failures += 1

                    if completed % PROGRESS_EVERY == 0:
                        elapsed = time.perf_counter() - started
                        log.info(
                            "completed=%d/%d failures=%d elapsed=%.1fs",
                            completed + skipped,
                            total,
                            failures,
                            elapsed,
                        )
                fh.flush()

    elapsed = time.perf_counter() - started
    log.info("finished completed=%d skipped=%d failures=%d elapsed=%.1fs",
             completed, skipped, failures, elapsed)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
