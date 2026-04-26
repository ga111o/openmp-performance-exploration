from __future__ import annotations

import csv
import itertools
import logging
import os
import shutil
import subprocess
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT   = Path(__file__).resolve().parent
SOURCE = ROOT / "main.c"
BINARY = ROOT / "main"

NA = "NA"

PARAM_ORDER: tuple[str, ...] = (
    "work_sharing",
    "schedule_kind",
    "schedule_chunk",
    "task_chunk_size",
    "num_threads",
    "proc_bind",
    "places",
    "wait_policy",
    "dynamic",
    "iterations",
)

METRIC_KEYS: tuple[str, ...] = (
    "num_threads_actual",
    "pool_create",
    "fork_join_min",
    "serial_min",
    "parallel_min",
    "pure_parallel",
    "speedup_total",
    "speedup_pure",
    "acc",
)

CSV_COLUMNS: tuple[str, ...] = (
    "combo_id",
    *PARAM_ORDER,
    *METRIC_KEYS,
    "wall_time_sec",
    "status",
    "stderr",
)

RESULT_LINE_RE = re.compile(r"^RESULT\s+(.+)$", re.MULTILINE)
KV_RE          = re.compile(r"(\w+)=(\S+)")

log = logging.getLogger("grid_search")


@dataclass(frozen=True)
class Combo:
    work_sharing: str
    schedule_kind: str
    schedule_chunk: Any
    task_chunk_size: Any
    num_threads: Any
    proc_bind: str
    places: str
    wait_policy: str
    dynamic: str
    iterations: Any

    def key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, k)) for k in PARAM_ORDER)


@dataclass
class RunResult:
    metrics: dict[str, str]
    wall: float
    status: str
    stderr: str


def load_grid(path: Path) -> dict[str, list[Any]]:
    with path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    if not isinstance(spec, dict) or "parameters" not in spec:
        raise ValueError(f"{path}: missing top-level 'parameters'")

    grid: dict[str, list[Any]] = {}
    for entry in spec["parameters"]:
        name = entry["name"].split("/", 1)[-1]
        grid[name] = list(entry["values"])

    missing = [k for k in PARAM_ORDER if k not in grid]
    if missing:
        raise ValueError(f"{path}: missing parameters: {missing}")
    return {k: grid[k] for k in PARAM_ORDER}


def canonicalize(combo: dict[str, Any], prune: bool) -> dict[str, Any]:
    if not prune:
        return dict(combo)

    c = dict(combo)
    ws = c["work_sharing"]
    if ws == "parallel_for":
        c["task_chunk_size"] = NA
        if c["schedule_kind"] == "auto":
            c["schedule_chunk"] = NA
    elif ws == "manual":
        c["schedule_kind"]    = NA
        c["schedule_chunk"]   = NA
        c["task_chunk_size"]  = NA
    elif ws == "tasks":
        c["schedule_kind"]    = NA
        c["schedule_chunk"]   = NA

    if str(c["proc_bind"]).lower() == "false":
        c["places"] = NA
    if int(c["num_threads"]) == 1:
        c["proc_bind"] = NA
        c["places"]    = NA
    return c


def iter_combos(grid: dict[str, list[Any]], prune: bool) -> Iterator[Combo]:
    seen: set[tuple[str, ...]] = set()
    for vals in itertools.product(*[grid[k] for k in PARAM_ORDER]):
        canon = canonicalize(dict(zip(PARAM_ORDER, vals)), prune=prune)
        key   = tuple(str(canon[k]) for k in PARAM_ORDER)
        if key in seen:
            continue
        seen.add(key)
        yield Combo(**canon)


def ensure_binary() -> None:
    if BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return
    cc     = os.environ.get("CC", "gcc")
    cflags = os.environ.get(
        "CFLAGS", "-O3 -march=native -fopenmp -Wall -Wextra"
    ).split()
    cmd = [cc, *cflags, "-o", str(BINARY), str(SOURCE)]
    log.info("build: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_env(combo: Combo) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(combo.num_threads)

    if combo.proc_bind != NA:
        env["OMP_PROC_BIND"] = str(combo.proc_bind)
    else:
        env.pop("OMP_PROC_BIND", None)

    if combo.places != NA:
        env["OMP_PLACES"] = str(combo.places)
    else:
        env.pop("OMP_PLACES", None)

    env["OMP_WAIT_POLICY"] = str(combo.wait_policy)
    env["OMP_DYNAMIC"]     = str(combo.dynamic)
    return env


def build_cli(combo: Combo) -> list[str]:
    return [
        str(BINARY),
        str(combo.work_sharing),
        str(combo.schedule_kind)   if combo.schedule_kind   != NA else "static",
        str(combo.schedule_chunk)  if combo.schedule_chunk  != NA else "0",
        str(combo.task_chunk_size) if combo.task_chunk_size != NA else "0",
        str(combo.num_threads),
        str(combo.iterations),
    ]


def run_one(combo: Combo, timeout: float) -> RunResult:
    cli = build_cli(combo)
    env = build_env(combo)
    t0  = time.perf_counter()
    try:
        proc = subprocess.run(
            cli, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult({}, time.perf_counter() - t0, "timeout", str(exc))
    except OSError as exc:
        return RunResult({}, time.perf_counter() - t0, "exception", repr(exc))
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        return RunResult({}, wall, f"exit_{proc.returncode}", proc.stderr.strip())

    match = RESULT_LINE_RE.search(proc.stdout)
    if match is None:
        return RunResult({}, wall, "no_result",
                         (proc.stderr or proc.stdout).strip())

    metrics = dict(KV_RE.findall(match.group(1)))
    return RunResult(metrics, wall, "ok", proc.stderr.strip())


def load_done_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {
            tuple(str(row[k]) for k in PARAM_ORDER)
            for row in csv.DictReader(fh)
        }


def open_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    fh     = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def make_row(combo_id: int, combo: Combo, result: RunResult) -> dict[str, Any]:
    row: dict[str, Any] = {"combo_id": combo_id}
    for k in PARAM_ORDER:
        row[k] = getattr(combo, k)
    for k in METRIC_KEYS:
        row[k] = result.metrics.get(k, "")
    row["wall_time_sec"] = f"{result.wall:.4f}"
    row["status"]        = result.status
    row["stderr"]        = (result.stderr or "")[:400].replace("\n", " | ")
    return row


def run(
    params_path: Path,
    out_path: Path,
    *,
    limit:    int | None = None,
    timeout:  float = 600.0,
    dry_run:  bool = False,
    prune:    bool = True,
    no_build: bool = False,
    fresh:    bool = False,
) -> int:
    if not no_build and not shutil.which(os.environ.get("CC", "gcc")):
        log.error("gcc not found on PATH")
        return 2

    grid   = load_grid(params_path)
    combos = list(iter_combos(grid, prune=prune))
    log.info("unique combos: %d (prune=%s) from %s",
             len(combos), prune, params_path.name)
    if dry_run:
        return 0

    if limit is not None:
        combos = combos[:limit]
        log.info("limited to first %d combos", len(combos))

    if not no_build:
        ensure_binary()
    if fresh and out_path.exists():
        out_path.unlink()

    done = load_done_keys(out_path)
    if done:
        log.info("resuming: %d already in %s", len(done), out_path.name)

    fh, writer = open_writer(out_path)
    started    = time.perf_counter()
    runs       = 0
    skipped    = 0
    try:
        for idx, combo in enumerate(combos):
            if combo.key() in done:
                skipped += 1
                continue

            result = run_one(combo, timeout=timeout)
            writer.writerow(make_row(idx, combo, result))
            fh.flush()
            os.fsync(fh.fileno())
            runs += 1

            log.info(
                "[%d/%d] ws=%-12s sk=%-7s sc=%-6s tc=%-8s nt=%-3s "
                "pb=%-7s pl=%-8s wp=%-7s dy=%-5s it=%-10s "
                "-> %s %.3fs (run=%d skip=%d elapsed=%.0fs)",
                idx + 1, len(combos),
                combo.work_sharing, combo.schedule_kind, combo.schedule_chunk,
                combo.task_chunk_size, combo.num_threads, combo.proc_bind,
                combo.places, combo.wait_policy, combo.dynamic, combo.iterations,
                result.status, result.wall,
                runs, skipped, time.perf_counter() - started,
            )
    finally:
        fh.close()

    log.info("done: wrote %s (run=%d skip=%d)", out_path, runs, skipped)
    return 0
