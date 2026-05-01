#!/usr/bin/env python3
"""
Single entry-point that drives the four OpenMP hyperparameter experiments
through the :mod:`grid_search` module.

Usage examples:
    python run_experiments.py                # run all experiments
    python run_experiments.py A C            # run only A and C
    python run_experiments.py --list         # show experiments and exit
    python run_experiments.py --interactive  # pick interactively
    python run_experiments.py B --fresh -v   # rerun B from scratch, verbose
    python run_experiments.py --smt-off      # save results under results/smt_off/
    python run_experiments.py --smt-on       # save results under results/smt_on/
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
CONFIGS  = ROOT / "configs"
RESULTS  = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_search  # noqa: E402


EXPERIMENTS: dict[str, dict[str, object]] = {
    "A": {
        "params": CONFIGS / "search_space_A_scaling.yaml",
        "out":    RESULTS / "results_A_scaling.csv",
        "desc":   "Strong scaling   (num_threads x iterations)",
    },
    "B": {
        "params": CONFIGS / "search_space_B_schedule.yaml",
        "out":    RESULTS / "results_B_schedule.csv",
        "desc":   "Schedule         (work_sharing x schedule_kind x chunk x task_chunk)",
    },
    "C": {
        "params": CONFIGS / "search_space_C_affinity.yaml",
        "out":    RESULTS / "results_C_affinity.csv",
        "desc":   "Affinity         (proc_bind x places)",
    },
    "D": {
        "params": CONFIGS / "search_space_D_runtime.yaml",
        "out":    RESULTS / "results_D_runtime.csv",
        "desc":   "Runtime policy   (wait_policy x dynamic x iterations)",
    },
    "E": {
        "params": CONFIGS / "search_space_E_cache.yaml",
        "out":    RESULTS / "results_E_cache.csv",
        "desc":   "Cache capacity   (pointer-chase working-set size)",
    },
    "F": {
        "params": CONFIGS / "search_space_F_l3_ccd.yaml",
        "out":    RESULTS / "results_F_l3_ccd.csv",
        "desc":   "L3/CCD sharing   (intra-CCD vs split CCD)",
    },
    "G": {
        "params": CONFIGS / "search_space_G_l1_l2_affinity.yaml",
        "out":    RESULTS / "results_G_l1_l2_affinity.csv",
        "desc":   "L1/L2 locality   (proc_bind x places)",
    },
    "H": {
        "params": CONFIGS / "search_space_H_bind_compute.yaml",
        "out":    RESULTS / "results_H_bind_compute.csv",
        "desc":   "Bind compute     (close vs spread, 2-6 threads)",
    },
    "I": {
        "params": CONFIGS / "search_space_I_bind_cache.yaml",
        "out":    RESULTS / "results_I_bind_cache.csv",
        "desc":   "Bind cache       (close vs spread, private cache walk)",
    },
    "J": {
        "params": CONFIGS / "search_space_J_false_sharing.yaml",
        "out":    RESULTS / "results_J_false_sharing.csv",
        "desc":   "False sharing    (unpadded vs cache-line padded slots)",
    },
}

log = logging.getLogger("run_experiments")


def list_experiments() -> None:
    print("Available experiments:")
    for k, v in EXPERIMENTS.items():
        print(f"  {k} : {v['desc']}")
        print(f"        params -> {Path(v['params']).name}")
        print(f"        out    -> {Path(v['out']).name}")


def select_interactively() -> list[str]:
    list_experiments()
    raw = input("\nEnter experiments to run (e.g. 'A C' or 'all'): ").strip()
    if not raw or raw.lower() == "all":
        return list(EXPERIMENTS.keys())
    keys = [tok.upper() for tok in raw.replace(",", " ").split()]
    bad  = [k for k in keys if k not in EXPERIMENTS]
    if bad:
        raise SystemExit(f"unknown experiment(s): {bad}")
    return keys


def normalize_keys(tokens: list[str]) -> list[str]:
    if not tokens:
        return list(EXPERIMENTS.keys())
    out: list[str] = []
    for tok in tokens:
        key = tok.strip().upper()
        if key.lower() == "all":
            return list(EXPERIMENTS.keys())
        if key not in EXPERIMENTS:
            raise SystemExit(
                f"unknown experiment '{tok}'. "
                f"Valid: {', '.join(EXPERIMENTS)} or 'all'."
            )
        if key not in out:
            out.append(key)
    return out


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "experiments", nargs="*",
        help="Experiments to run (A B C D E F G H I J). Empty = all.",
    )
    p.add_argument("--list",        action="store_true",
                   help="List experiments and exit.")
    p.add_argument("--interactive", action="store_true",
                   help="Prompt for experiment selection.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Only enumerate combos, do not execute the binary.")
    p.add_argument("--fresh",       action="store_true",
                   help="Delete each experiment's output CSV before running.")
    p.add_argument("--no-build",    action="store_true",
                   help="Skip rebuilding the C binary.")
    p.add_argument("--no-prune",    action="store_true",
                   help="Disable canonicalize() pruning (full Cartesian grid).")
    smt = p.add_mutually_exclusive_group()
    smt.add_argument("--smt-on",     action="store_const", const="on", dest="smt",
                     help="Label this run as SMT on; write results under results/smt_on/.")
    smt.add_argument("--smt-off",    action="store_const", const="off", dest="smt",
                     help="Label this run as SMT off; write results under results/smt_off/.")
    p.set_defaults(smt="keep")
    p.add_argument("--limit",       type=int, default=None,
                   help="Cap number of combos per experiment (debugging).")
    p.add_argument("--timeout",     type=float, default=600.0,
                   help="Per-run timeout in seconds (default 600).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def smt_modes(args: argparse.Namespace) -> list[str | None]:
    if args.smt == "keep":
        return [None]
    return [args.smt]


def smt_output_path(base_path: Path, mode: str | None) -> Path:
    if mode is None:
        return base_path
    return base_path.parent / f"smt_{mode}" / base_path.name


def run_experiment(
    key: str,
    args: argparse.Namespace,
    no_build: bool,
    out_path: Path,
    smt_mode: str | None,
) -> int:
    info   = EXPERIMENTS[key]
    banner = f"Experiment {key}: {info['desc']}"
    bar    = "=" * len(banner)
    print(f"\n{bar}\n{banner}\n{bar}", flush=True)

    return grid_search.run(
        params_path=Path(info["params"]),
        out_path=out_path,
        limit=args.limit,
        timeout=args.timeout,
        dry_run=args.dry_run,
        prune=not args.no_prune,
        no_build=no_build,
        fresh=args.fresh,
        smt_mode=smt_mode,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    configure_logging(args.verbose)

    if args.list:
        list_experiments()
        return 0

    if args.interactive:
        selected = select_interactively()
    else:
        selected = normalize_keys(args.experiments)

    missing = [k for k in selected
               if not Path(EXPERIMENTS[k]["params"]).exists()]
    if missing:
        log.error("missing YAML for: %s", missing)
        return 2

    started  = time.perf_counter()
    rc       = 0
    no_build = args.no_build
    for mode in smt_modes(args):
        if mode is not None:
            print(f"\nSMT {mode}: results will be written under results/smt_{mode}/")

        for key in selected:
            out_path = smt_output_path(Path(EXPERIMENTS[key]["out"]), mode)
            ret = run_experiment(
                key,
                args,
                no_build=no_build,
                out_path=out_path,
                smt_mode=mode,
            )
            if ret != 0:
                rc = ret
                log.error("experiment %s returned %d", key, ret)
            # After the first run the binary is up-to-date; skip the build check
            # in subsequent experiments and SMT passes.
            no_build = True

    elapsed = time.perf_counter() - started
    print(f"\nAll selected experiments finished in {elapsed:.1f}s "
          f"(exit={rc}).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
