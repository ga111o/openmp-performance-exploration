#!/usr/bin/env python3
"""Summarize OpenMP experiment CSVs into analysis-ready statistics.

The script reads the CSV files produced by run_experiments.py and
run_prime_range.py.  It writes compact tables under results/statistics/ so the
main conclusions can be checked without reopening every raw CSV or figure.

Examples:
    python summarize_results.py
    python summarize_results.py --results-dir results --out-dir results/statistics
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "statistics"

SMT_ON_LABEL = "SMT on"
SMT_OFF_LABEL = "SMT off"
SINGLE_RUN_LABEL = "single run"

NS_TO_MS = 1.0e-6
CACHE_WALK_MIN_STEPS = 1 << 22
CACHE_WALK_MAX_STEPS = 1 << 26

CSV_FILES = {
    "A": "results_A_scaling.csv",
    "B": "results_B_schedule.csv",
    "C": "results_C_affinity.csv",
    "D": "results_D_runtime.csv",
    "E": "results_E_cache.csv",
    "F": "results_F_l3_ccd.csv",
    "G": "results_G_l1_l2_affinity.csv",
    "H": "results_H_bind_compute.csv",
    "I": "results_I_bind_cache.csv",
    "J": "results_J_false_sharing.csv",
}

PARAM_COLUMNS = [
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
    "cpu_set",
]

BASE_NUMERIC_COLUMNS = [
    "combo_id",
    "schedule_chunk",
    "task_chunk_size",
    "num_threads",
    "iterations",
    "num_threads_actual",
    "pool_create",
    "fork_join_min",
    "serial_min",
    "parallel_min",
    "pure_parallel",
    "speedup_total",
    "speedup_pure",
    "acc",
    "wall_time_sec",
]

CORE_METRICS = [
    "speedup_pure",
    "speedup_total",
    "efficiency_pure",
    "parallel_ms",
    "serial_ms",
    "pure_parallel_ms",
    "fork_join_ms",
    "pool_create_ms",
    "wall_time_sec",
    "cycles_per_load",
    "parallel_cycles_per_load",
    "cycles_per_update",
    "parallel_cycles_per_update",
]


@dataclass(frozen=True)
class ExperimentSpec:
    title: str
    context: str
    primary_metric: str
    larger_is_better: bool
    group_by: tuple[str, ...]


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "A": ExperimentSpec(
        "Strong scaling",
        "num_threads x iterations; compute-bound parallel_for scaling under controlled cpu_set",
        "speedup_pure",
        True,
        ("num_threads", "iterations"),
    ),
    "B": ExperimentSpec(
        "Schedule",
        "work_sharing, schedule_kind/chunk, and task_chunk_size at 12 threads",
        "speedup_pure",
        True,
        ("work_sharing", "schedule_kind", "schedule_chunk", "task_chunk_size"),
    ),
    "C": ExperimentSpec(
        "Affinity",
        "proc_bind x places effect at 12 threads",
        "speedup_pure",
        True,
        ("proc_bind", "places"),
    ),
    "D": ExperimentSpec(
        "Runtime policy",
        "OMP_WAIT_POLICY x OMP_DYNAMIC across iteration counts",
        "speedup_pure",
        True,
        ("wait_policy", "dynamic", "iterations"),
    ),
    "E": ExperimentSpec(
        "Cache capacity",
        "single-thread pointer-chase working-set size, reported as cycles per load",
        "cycles_per_load",
        False,
        ("working_set_class", "iterations"),
    ),
    "F": ExperimentSpec(
        "L3/CCD sharing",
        "shared pointer-chase, one CCD cpu_set versus split CCD cpu_set",
        "parallel_cycles_per_load",
        False,
        ("cpu_set", "iterations"),
    ),
    "G": ExperimentSpec(
        "L1/L2 locality",
        "private pointer-chase, proc_bind x places under 12/24 threads",
        "parallel_cycles_per_load",
        False,
        ("num_threads", "proc_bind", "places", "iterations"),
    ),
    "H": ExperimentSpec(
        "Bind compute",
        "compute-bound close versus spread placement for 2-6 threads",
        "speedup_pure",
        True,
        ("num_threads", "proc_bind"),
    ),
    "I": ExperimentSpec(
        "Bind cache",
        "private cache-walk close versus spread placement for several working sets",
        "parallel_cycles_per_load",
        False,
        ("num_threads", "proc_bind", "iterations"),
    ),
    "J": ExperimentSpec(
        "False sharing",
        "unpadded versus cache-line padded volatile slots",
        "parallel_cycles_per_update",
        False,
        ("work_sharing", "num_threads", "proc_bind", "iterations"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write analysis-ready statistics from result CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing results_*.csv or smt_on/smt_off subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where summary CSV files will be written.",
    )
    parser.add_argument(
        "--no-prime",
        action="store_true",
        help="Skip results/prime_once_range.csv even when it exists.",
    )
    return parser.parse_args()


def smt_label(path: Path) -> str:
    name = path.name.lower()
    if name == "smt_on":
        return SMT_ON_LABEL
    if name == "smt_off":
        return SMT_OFF_LABEL
    return SINGLE_RUN_LABEL


def result_sets(results_dir: Path) -> list[Path]:
    smt_dirs = sorted(
        path
        for path in results_dir.glob("smt_*")
        if path.is_dir() and any((path / filename).exists() for filename in CSV_FILES.values())
    )
    if smt_dirs:
        return smt_dirs
    return [results_dir]


def to_number(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def add_openmp_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    to_number(df, BASE_NUMERIC_COLUMNS)

    for src, dst in [
        ("parallel_min", "parallel_ms"),
        ("serial_min", "serial_ms"),
        ("pure_parallel", "pure_parallel_ms"),
        ("fork_join_min", "fork_join_ms"),
        ("pool_create", "pool_create_ms"),
    ]:
        if src in df.columns:
            df[dst] = df[src] * NS_TO_MS

    threads = df["num_threads_actual"].replace(0, np.nan)
    df["efficiency_pure"] = df["speedup_pure"] / threads
    df["working_set_kib"] = df["iterations"] * 8.0 / 1024.0
    df["working_set_mib"] = df["working_set_kib"] / 1024.0
    df["working_set_class"] = np.select(
        [
            df["working_set_kib"] <= 32.0,
            df["working_set_kib"] <= 512.0,
            df["working_set_mib"] <= 32.0,
        ],
        ["L1-sized", "L2-sized", "L3-sized"],
        default="beyond L3/DRAM",
    )

    cache_steps = np.clip(
        df["iterations"] * 4.0,
        CACHE_WALK_MIN_STEPS,
        CACHE_WALK_MAX_STEPS,
    )
    load_count = pd.Series(cache_steps, index=df.index, dtype="float64")
    if "work_sharing" in df.columns:
        private = df["work_sharing"].eq("cache_walk_private")
        load_count.loc[private] *= threads.loc[private].fillna(df.loc[private, "num_threads"])
    df["cycles_per_load"] = df["serial_min"] / load_count
    df["parallel_cycles_per_load"] = df["parallel_min"] / load_count

    updates = df["iterations"] * threads.fillna(df["num_threads"])
    df["cycles_per_update"] = df["serial_min"] / updates.replace(0, np.nan)
    df["parallel_cycles_per_update"] = df["parallel_min"] / updates.replace(0, np.nan)
    return df


def read_openmp_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []

    for set_dir in result_sets(results_dir):
        label = smt_label(set_dir)
        for key, filename in CSV_FILES.items():
            path = set_dir / filename
            if not path.exists():
                continue
            raw = pd.read_csv(path).replace({"NA": np.nan, "true": True, "false": False})
            raw["experiment"] = key
            raw["experiment_title"] = EXPERIMENTS[key].title
            raw["experiment_context"] = EXPERIMENTS[key].context
            raw["smt"] = label
            raw["result_set"] = set_dir.name
            raw["source_csv"] = str(path.relative_to(results_dir.parent))

            status = (
                raw.groupby(["experiment", "experiment_title", "result_set", "smt", "status"], dropna=False)
                .size()
                .reset_index(name="rows")
            )
            status_frames.append(status)

            ok = raw["status"].eq("ok") if "status" in raw.columns else pd.Series(True, index=raw.index)
            frames.append(add_openmp_derived_columns(raw.loc[ok].reset_index(drop=True)))

    if not frames:
        raise FileNotFoundError(f"No OpenMP result CSV files found under {results_dir}")

    return pd.concat(frames, ignore_index=True), pd.concat(status_frames, ignore_index=True)


def config_label(row: pd.Series) -> str:
    parts: list[str] = []
    for col in PARAM_COLUMNS:
        if col not in row or pd.isna(row[col]):
            continue
        value = row[col]
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        parts.append(f"{col}={value}")
    return "; ".join(parts)


def stat_record(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "p90": np.nan,
            "max": np.nan,
            "cv": np.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": int(values.size),
        "mean": mean,
        "median": float(values.median()),
        "std": std,
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
        "cv": std / abs(mean) if mean else np.nan,
    }


def summarize_metric_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in group.columns:
                continue
            record = {**base, "metric": metric}
            record.update(stat_record(group[metric]))
            rows.append(record)
    return pd.DataFrame(rows)


def experiment_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (experiment, result_set, smt), group in df.groupby(
        ["experiment", "result_set", "smt"], dropna=False
    ):
        spec = EXPERIMENTS[str(experiment)]
        metric = spec.primary_metric
        values = pd.to_numeric(group[metric], errors="coerce")
        best_idx = values.idxmax() if spec.larger_is_better else values.idxmin()
        worst_idx = values.idxmin() if spec.larger_is_better else values.idxmax()
        best = group.loc[best_idx]
        worst = group.loc[worst_idx]
        rows.append(
            {
                "experiment": experiment,
                "title": spec.title,
                "context": spec.context,
                "result_set": result_set,
                "smt": smt,
                "rows_ok": len(group),
                "primary_metric": metric,
                "larger_is_better": spec.larger_is_better,
                "primary_mean": values.mean(),
                "primary_median": values.median(),
                "primary_min": values.min(),
                "primary_max": values.max(),
                "best_value": best[metric],
                "best_config": config_label(best),
                "worst_value": worst[metric],
                "worst_config": config_label(worst),
                "num_threads_values": compact_values(group.get("num_threads")),
                "iteration_values": compact_values(group.get("iterations")),
            }
        )
    return pd.DataFrame(rows).sort_values(["experiment", "result_set", "smt"])


def compact_values(series: pd.Series | None, limit: int = 8) -> str:
    if series is None:
        return ""
    values = sorted(pd.unique(series.dropna()))
    rendered = [str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in values]
    if len(rendered) > limit:
        return ", ".join(rendered[:limit]) + f", ... ({len(rendered)} values)"
    return ", ".join(rendered)


def best_worst_configs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (experiment, result_set, smt), group in df.groupby(
        ["experiment", "result_set", "smt"], dropna=False
    ):
        spec = EXPERIMENTS[str(experiment)]
        metric = spec.primary_metric
        ranked = group.sort_values(metric, ascending=not spec.larger_is_better)
        for rank, (_, row) in enumerate(ranked.head(5).iterrows(), start=1):
            rows.append(
                {
                    "experiment": experiment,
                    "title": spec.title,
                    "result_set": result_set,
                    "smt": smt,
                    "kind": "best",
                    "rank": rank,
                    "metric": metric,
                    "value": row[metric],
                    "parallel_ms": row.get("parallel_ms"),
                    "speedup_pure": row.get("speedup_pure"),
                    "efficiency_pure": row.get("efficiency_pure"),
                    "config": config_label(row),
                }
            )
        for rank, (_, row) in enumerate(ranked.tail(5).iloc[::-1].iterrows(), start=1):
            rows.append(
                {
                    "experiment": experiment,
                    "title": spec.title,
                    "result_set": result_set,
                    "smt": smt,
                    "kind": "worst",
                    "rank": rank,
                    "metric": metric,
                    "value": row[metric],
                    "parallel_ms": row.get("parallel_ms"),
                    "speedup_pure": row.get("speedup_pure"),
                    "efficiency_pure": row.get("efficiency_pure"),
                    "config": config_label(row),
                }
            )
    return pd.DataFrame(rows)


def parameter_effects(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment, group in df.groupby("experiment", dropna=False):
        spec = EXPERIMENTS[str(experiment)]
        group_cols = ["experiment", "experiment_title", "result_set", "smt"]
        group_cols.extend(col for col in spec.group_by if col in group.columns)
        metrics = [spec.primary_metric, "speedup_pure", "parallel_ms", "efficiency_pure"]
        if spec.primary_metric not in metrics:
            metrics.insert(0, spec.primary_metric)
        summary = summarize_metric_groups(group, group_cols, list(dict.fromkeys(metrics)))
        summary["context"] = spec.context
        summary["primary_metric"] = spec.primary_metric
        summary["larger_is_better"] = spec.larger_is_better
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def row_key(row: pd.Series, cols: list[str]) -> str:
    return " | ".join(f"{col}={row[col]}" for col in cols if col in row and pd.notna(row[col]))


def numeric_value(row: pd.Series, metric: str) -> float:
    return pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]


def ratio_change(new_value: float, old_value: float) -> tuple[float, float]:
    if pd.isna(new_value) or pd.isna(old_value) or old_value == 0:
        return np.nan, np.nan
    ratio = new_value / old_value
    return ratio, (ratio - 1.0) * 100.0


def append_pair_metric_rows(
    rows: list[dict[str, object]],
    *,
    base: dict[str, object],
    metrics: list[str],
    available_columns: Iterable[str],
    first: pd.Series,
    second: pd.Series,
    first_value_name: str,
    second_value_name: str,
    ratio_name: str,
    pct_name: str,
) -> None:
    columns = set(available_columns)
    for metric in metrics:
        if metric not in columns:
            continue
        first_value = numeric_value(first, metric)
        second_value = numeric_value(second, metric)
        ratio, pct = ratio_change(second_value, first_value)
        rows.append(
            {
                **base,
                "metric": metric,
                first_value_name: first_value,
                second_value_name: second_value,
                ratio_name: ratio,
                pct_name: pct,
            }
        )


def add_pairwise_rows(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    *,
    experiment: str,
    comparison: str,
    match_cols: list[str],
    variant_col: str,
    baseline: object,
    challenger: object,
    metrics: list[str],
) -> None:
    part = df[df["experiment"].eq(experiment)].copy()
    if part.empty or variant_col not in part.columns:
        return
    for keys, group in part.groupby(["result_set", "smt", *match_cols], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(["result_set", "smt", *match_cols], keys))
        candidates = group[group[variant_col].isin([baseline, challenger])]
        if candidates[variant_col].nunique(dropna=False) < 2:
            continue
        a = candidates[candidates[variant_col].eq(baseline)].iloc[0]
        b = candidates[candidates[variant_col].eq(challenger)].iloc[0]
        append_pair_metric_rows(
            rows,
            base={
                "experiment": experiment,
                "title": EXPERIMENTS[experiment].title,
                "comparison": comparison,
                **key_values,
                "variant_column": variant_col,
                "baseline": baseline,
                "challenger": challenger,
                "match_key": row_key(a, match_cols),
            },
            metrics=metrics,
            available_columns=candidates.columns,
            first=a,
            second=b,
            first_value_name="baseline_value",
            second_value_name="challenger_value",
            ratio_name="ratio_challenger_over_baseline",
            pct_name="pct_change_challenger_vs_baseline",
        )


def key_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    add_pairwise_rows(
        rows,
        df,
        experiment="D",
        comparison="passive vs active wait policy",
        match_cols=["dynamic", "iterations"],
        variant_col="wait_policy",
        baseline="active",
        challenger="passive",
        metrics=["speedup_pure", "parallel_ms"],
    )
    add_pairwise_rows(
        rows,
        df,
        experiment="D",
        comparison="OMP_DYNAMIC true vs false",
        match_cols=["wait_policy", "iterations"],
        variant_col="dynamic",
        baseline=False,
        challenger=True,
        metrics=["speedup_pure", "parallel_ms"],
    )
    for experiment, metrics in [
        ("H", ["speedup_pure", "parallel_ms"]),
        ("I", ["parallel_cycles_per_load", "speedup_pure", "parallel_ms"]),
    ]:
        add_pairwise_rows(
            rows,
            df,
            experiment=experiment,
            comparison="spread vs close binding",
            match_cols=["num_threads", "iterations"],
            variant_col="proc_bind",
            baseline="close",
            challenger="spread",
            metrics=metrics,
        )
    add_pairwise_rows(
        rows,
        df,
        experiment="F",
        comparison="split CCD vs intra CCD",
        match_cols=["iterations"],
        variant_col="cpu_set",
        baseline="0-5",
        challenger="0-2,6-8",
        metrics=["parallel_cycles_per_load", "speedup_pure", "parallel_ms"],
    )
    add_pairwise_rows(
        rows,
        df,
        experiment="J",
        comparison="padded vs unpadded false-sharing slots",
        match_cols=["num_threads", "proc_bind", "iterations"],
        variant_col="work_sharing",
        baseline="false_sharing",
        challenger="false_sharing_padded",
        metrics=["parallel_cycles_per_update", "speedup_pure", "parallel_ms"],
    )
    return pd.DataFrame(rows)


def smt_rows_for_group(
    keys: object,
    key_cols: list[str],
    group: pd.DataFrame,
    metrics: list[str],
) -> list[dict[str, object]]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    by_smt = {str(row["smt"]): row for _, row in group.iterrows()}
    if SMT_OFF_LABEL not in by_smt or SMT_ON_LABEL not in by_smt:
        return []

    off = by_smt[SMT_OFF_LABEL]
    on = by_smt[SMT_ON_LABEL]
    key_values = dict(zip(key_cols, keys))
    rows: list[dict[str, object]] = []
    append_pair_metric_rows(
        rows,
        base={
            **key_values,
            "title": EXPERIMENTS[str(key_values["experiment"])].title,
            "config": config_label(off),
        },
        metrics=metrics,
        available_columns=group.columns,
        first=off,
        second=on,
        first_value_name="smt_off",
        second_value_name="smt_on",
        ratio_name="ratio_on_over_off",
        pct_name="pct_change_on_vs_off",
    )
    return [row for row in rows if not pd.isna(row["smt_off"]) and not pd.isna(row["smt_on"])]


def smt_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df["smt"].nunique(dropna=True) < 2:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    metrics = [
        "speedup_pure",
        "efficiency_pure",
        "parallel_ms",
        "parallel_cycles_per_load",
        "parallel_cycles_per_update",
    ]
    key_cols = ["experiment", *PARAM_COLUMNS]
    for keys, group in df.groupby(key_cols, dropna=False):
        rows.extend(smt_rows_for_group(keys, key_cols, group, metrics))
    return pd.DataFrame(rows)


def read_prime_range(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"].eq("ok")].copy()
    to_number(df, ["limit", "primes", "cycles", "seconds", "wall_seconds"])
    df = df.dropna(subset=["limit", "primes", "cycles", "seconds", "wall_seconds"])
    df = df.sort_values("limit").drop_duplicates(subset="limit", keep="last")
    df["prime_density"] = df["primes"] / df["limit"]
    df["pnt_estimate"] = df["limit"] / np.log(df["limit"])
    df["pnt_relative_error_pct"] = (df["primes"] / df["pnt_estimate"] - 1.0) * 100.0
    df["cycles_per_limit"] = df["cycles"] / df["limit"]
    df["ns_per_limit"] = df["seconds"] * 1.0e9 / df["limit"]
    df["measured_ms"] = df["seconds"] * 1.0e3
    df["wall_ms"] = df["wall_seconds"] * 1.0e3
    return df


def prime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in [
        "primes",
        "prime_density",
        "pnt_relative_error_pct",
        "cycles",
        "cycles_per_limit",
        "ns_per_limit",
        "measured_ms",
        "wall_ms",
    ]:
        row = {"metric": metric, "limit_min": df["limit"].min(), "limit_max": df["limit"].max()}
        row.update(stat_record(df[metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def prime_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 1e6, 1e7, 1e8, 1e9, np.inf]
    labels = ["<=1e6", "1e6-1e7", "1e7-1e8", "1e8-1e9", ">1e9"]
    bucketed = df.copy()
    bucketed["limit_bucket"] = pd.cut(bucketed["limit"], bins=bins, labels=labels, include_lowest=True)
    return summarize_metric_groups(
        bucketed,
        ["limit_bucket"],
        ["prime_density", "pnt_relative_error_pct", "cycles_per_limit", "ns_per_limit", "measured_ms"],
    )


def write_report(
    out_dir: Path,
    overview: pd.DataFrame,
    best_worst: pd.DataFrame,
    comparisons: pd.DataFrame,
    prime: pd.DataFrame | None,
) -> None:
    lines = [
        "# Result Statistics Report",
        "",
        "Generated from the raw CSV files in `results/`.",
        "",
        "## Experiment Overview",
        "",
    ]
    for _, row in overview.iterrows():
        better = "higher is better" if row["larger_is_better"] else "lower is better"
        lines.append(
            f"- {row['experiment']} {row['title']} ({row['smt']}): "
            f"{row['primary_metric']} median={row['primary_median']:.6g}, "
            f"best={row['best_value']:.6g} ({better})."
        )

    lines.extend(["", "## Best Configurations", ""])
    best = best_worst[best_worst["kind"].eq("best") & best_worst["rank"].eq(1)]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['experiment']} {row['title']} ({row['smt']}): "
            f"{row['metric']}={row['value']:.6g}; {row['config']}"
        )

    if not comparisons.empty:
        lines.extend(["", "## Key Comparisons", ""])
        for _, row in comparisons.head(20).iterrows():
            lines.append(
                f"- {row['experiment']} {row['comparison']} ({row['smt']}, {row['match_key']}): "
                f"{row['metric']} changes by {row['pct_change_challenger_vs_baseline']:.2f}%."
            )

    if prime is not None and not prime.empty:
        limits = f"{int(prime['limit'].min())}..{int(prime['limit'].max())}"
        lines.extend(["", "## Prime Sweep", ""])
        lines.append(f"- Prime sweep covers limit={limits}, rows={len(prime)}.")

    (out_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    openmp, status = read_openmp_results(results_dir)
    overview = experiment_overview(openmp)
    metric_summary = summarize_metric_groups(
        openmp,
        ["experiment", "experiment_title", "result_set", "smt"],
        CORE_METRICS,
    )
    effects = parameter_effects(openmp)
    best_worst = best_worst_configs(openmp)
    comparisons = key_comparisons(openmp)
    smt = smt_comparison(openmp)

    write_csv(status, out_dir / "status_summary.csv")
    write_csv(overview, out_dir / "experiment_overview.csv")
    write_csv(metric_summary, out_dir / "metric_summary.csv")
    write_csv(effects, out_dir / "parameter_effects.csv")
    write_csv(best_worst, out_dir / "best_worst_configs.csv")
    write_csv(comparisons, out_dir / "key_comparisons.csv")
    if not smt.empty:
        write_csv(smt, out_dir / "smt_comparison.csv")

    prime_df: pd.DataFrame | None = None
    prime_path = results_dir / "prime_once_range.csv"
    if not args.no_prime and prime_path.exists():
        prime_df = read_prime_range(prime_path)
        write_csv(prime_summary(prime_df), out_dir / "prime_summary.csv")
        write_csv(prime_buckets(prime_df), out_dir / "prime_bucket_summary.csv")

    write_report(out_dir, overview, best_worst, comparisons, prime_df)

    print(f"Wrote result statistics to {out_dir}")
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
