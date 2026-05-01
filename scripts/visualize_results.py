#!/usr/bin/env python3
"""Create publication-quality figures from the OpenMP experiment CSV files.

The script expects the CSV files produced by run_experiments.py:

    results/results_A_scaling.csv
    ...

or the SMT-split layout:

    results/smt_on/results_A_scaling.csv
    results/smt_off/results_A_scaling.csv
    ...

It writes vector figures for LaTeX inclusion and high-DPI PNG copies for quick
inspection.  Example:

    python visualize_results.py
    python visualize_results.py --formats pdf svg png --show
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

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

NS_TO_MS = 1.0e-6
CACHE_WALK_MIN_STEPS = 1 << 22
CACHE_WALK_MAX_STEPS = 1 << 26
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#6F6F6F",
    "light_gray": "#D9D9D9",
}


@dataclass(frozen=True)
class FigureSpec:
    name: str
    title: str
    fig: plt.Figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate journal-style figures from OpenMP result CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing results_*.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES,
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "svg", "png"],
        help="Output formats. Use pdf/svg for LaTeX-quality vector graphics.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after writing them.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def smt_label(path: Path) -> str:
    name = path.name.lower()
    if name == "smt_on":
        return "SMT on"
    if name == "smt_off":
        return "SMT off"
    return "single run"


def result_sets(results_dir: Path) -> list[Path]:
    smt_dirs = sorted(
        path
        for path in results_dir.glob("smt_*")
        if path.is_dir() and any((path / filename).exists() for filename in CSV_FILES.values())
    )
    if smt_dirs:
        return smt_dirs
    return [results_dir]


def read_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    sets = result_sets(results_dir)
    data: dict[str, list[pd.DataFrame]] = {key: [] for key in CSV_FILES}
    missing: list[str] = []

    for set_dir in sets:
        label = smt_label(set_dir)
        for key, filename in CSV_FILES.items():
            path = set_dir / filename
            if not path.exists():
                missing.append(str(path))
                continue
            frame = pd.read_csv(path)
            frame["smt"] = label
            frame["result_set"] = set_dir.name
            data[key].append(clean_frame(frame))

    if missing:
        raise FileNotFoundError("Missing result CSV(s): " + ", ".join(missing))

    return {key: pd.concat(frames, ignore_index=True) for key, frames in data.items()}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({"NA": np.nan, "true": True, "false": False})
    numeric_columns = [
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
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = df["status"].eq("ok") if "status" in df else pd.Series(True, index=df.index)
    df = df.loc[ok].reset_index(drop=True)
    df["parallel_ms"] = df["parallel_min"] * NS_TO_MS
    df["serial_ms"] = df["serial_min"] * NS_TO_MS
    df["fork_join_ms"] = df["fork_join_min"] * NS_TO_MS
    df["pool_create_ms"] = df["pool_create"] * NS_TO_MS
    df["efficiency_pure"] = df["speedup_pure"] / df["num_threads_actual"].replace(0, np.nan)
    df["working_set_kib"] = df["iterations"] * 8.0 / 1024.0
    df["cache_walk_steps"] = np.clip(
        df["iterations"] * 4.0,
        CACHE_WALK_MIN_STEPS,
        CACHE_WALK_MAX_STEPS,
    )
    load_count = df["cache_walk_steps"].copy()
    if "work_sharing" in df:
        private = df["work_sharing"].eq("cache_walk_private")
        load_count.loc[private] *= df.loc[private, "num_threads_actual"].fillna(
            df.loc[private, "num_threads"]
        )
    df["cycles_per_load"] = df["serial_min"] / load_count
    df["parallel_cycles_per_load"] = df["parallel_min"] / load_count
    update_count = df["iterations"] * df["num_threads_actual"].fillna(df["num_threads"])
    df["cycles_per_update"] = df["serial_min"] / update_count
    df["parallel_cycles_per_update"] = df["parallel_min"] / update_count
    return df


def human_int(value: float) -> str:
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def has_smt_split(df: pd.DataFrame) -> bool:
    return "smt" in df and df["smt"].nunique(dropna=True) > 1


def smt_order(values: pd.Series) -> list[str]:
    preferred = ["SMT off", "SMT on", "single run"]
    present = [str(value) for value in values.dropna().unique()]
    return [value for value in preferred if value in present] + sorted(
        value for value in present if value not in preferred
    )


def with_smt(value: object, smt: object, enabled: bool, sep: str = "\n") -> str:
    text = str(value)
    if not enabled:
        return text
    return f"{text}{sep}{smt}"


def line_style_for_smt(smt: object) -> str:
    return {"SMT off": "-", "SMT on": "--"}.get(str(smt), "-")


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=PALETTE["light_gray"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7, color=PALETTE["gray"])


def add_median_line(ax: plt.Axes, values: pd.Series, label: str = "median") -> None:
    median = float(values.median())
    ax.axhline(median, color=PALETTE["gray"], linestyle=":", linewidth=1.0)
    ax.annotate(
        f"{label}: {median:.2f}x",
        xy=(0.995, median),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["gray"],
    )


def plot_scaling(df: pd.DataFrame) -> FigureSpec:
    split = has_smt_split(df)
    iterations = sorted(df["iterations"].dropna().unique())
    smt_values = smt_order(df["smt"]) if "smt" in df else ["single run"]
    thread_ticks = sorted(df["num_threads"].dropna().unique())
    max_threads = int(df["num_threads"].max())
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(iterations)))
    color_by_iter = dict(zip(iterations, colors))

    panel_letters = iter("abcdefgh")

    if split:
        n_smt = len(smt_values)
        fig = plt.figure(figsize=(6.4 * n_smt, 14.5), constrained_layout=True)
        gs = fig.add_gridspec(3, n_smt)
        speedup_axes = [fig.add_subplot(gs[0, i]) for i in range(n_smt)]
        efficiency_axes = [fig.add_subplot(gs[1, i]) for i in range(n_smt)]
        ax_heatmap = fig.add_subplot(gs[2, 0])
        ax_overhead = fig.add_subplot(gs[2, 1]) if n_smt >= 2 else fig.add_subplot(gs[2, 0])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), constrained_layout=True)
        speedup_axes = [axes[0, 0]]
        efficiency_axes = [axes[0, 1]]
        ax_heatmap = axes[1, 0]
        ax_overhead = axes[1, 1]

    def panel(label: str) -> str:
        return f"({next(panel_letters)}) {label}"

    ideal_x = np.arange(1, max_threads + 1)
    for ax, smt in zip(speedup_axes, smt_values):
        for iteration in iterations:
            part = df[df["iterations"].eq(iteration)]
            if "smt" in part:
                part = part[part["smt"].eq(smt)]
            part = part.sort_values("num_threads")
            if part.empty:
                continue
            ax.plot(
                part["num_threads"],
                part["speedup_pure"],
                marker="o",
                color=color_by_iter[iteration],
                linestyle="-",
                label=human_int(iteration),
            )
        ax.plot(ideal_x, ideal_x, "--", color=PALETTE["gray"], linewidth=1.1, label="ideal")
        ax.axvline(6, color=PALETTE["red"], linestyle=":", linewidth=1.1)
        ax.text(6.3, ax.get_ylim()[1] * 0.92, "CCD boundary", fontsize=9, color=PALETTE["red"])
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("Pure parallel speedup")
        suffix = f" ({smt})" if split else ""
        ax.set_title(panel("Strong scaling") + suffix)
        ax.set_xticks(thread_ticks)
        ax.legend(title="N", ncols=2, loc="upper left", fontsize=9)
        style_axis(ax)

    for ax, smt in zip(efficiency_axes, smt_values):
        for iteration in iterations:
            part = df[df["iterations"].eq(iteration)]
            if "smt" in part:
                part = part[part["smt"].eq(smt)]
            part = part.sort_values("num_threads")
            if part.empty:
                continue
            ax.plot(
                part["num_threads"],
                part["efficiency_pure"],
                marker="o",
                color=color_by_iter[iteration],
                linestyle="-",
                label=human_int(iteration),
            )
        ax.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
        ax.axhline(0.5, color=PALETTE["gray"], linestyle=":", linewidth=0.9)
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("Parallel efficiency")
        suffix = f" ({smt})" if split else ""
        ax.set_title(panel("Efficiency") + suffix)
        ax.set_ylim(bottom=0)
        ax.set_xticks(thread_ticks)
        ax.legend(title="N", ncols=2, loc="upper right", fontsize=9)
        style_axis(ax)

    heat_columns = ["num_threads"]
    if split:
        heat_columns = ["smt", "num_threads"]
    heat = df.pivot_table(
        index="iterations",
        columns=heat_columns,
        values="speedup_pure",
        aggfunc="mean",
    ).sort_index()
    im = ax_heatmap.imshow(heat.values, aspect="auto", cmap="magma", origin="lower")
    if split:
        heat_labels = [f"{smt.replace('SMT ', '')}\n{int(threads)}" for smt, threads in heat.columns]
    else:
        heat_labels = [int(x) for x in heat.columns]
    ax_heatmap.set_xticks(np.arange(len(heat.columns)), heat_labels)
    ax_heatmap.set_yticks(np.arange(len(heat.index)), [human_int(x) for x in heat.index])
    ax_heatmap.set_xlabel("OpenMP threads")
    ax_heatmap.set_ylabel("Iterations")
    ax_heatmap.set_title(panel("Speedup landscape"))
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.02)
    cbar.set_label("Speedup")

    large = df[df["iterations"].eq(max(iterations))].sort_values(
        ["smt", "num_threads"] if "smt" in df else ["num_threads"]
    )
    overhead_index = large["num_threads"].astype(int).astype(str) + "T"
    if split:
        overhead_index = large["smt"].str.replace("SMT ", "", regex=False) + " / " + overhead_index
    overhead = pd.DataFrame(
        {
            "fork/join": large["fork_join_ms"].to_numpy(),
            "pool create": large["pool_create_ms"].to_numpy(),
        },
        index=overhead_index,
    )
    overhead.plot(
        kind="bar",
        stacked=True,
        color=[PALETTE["blue"], PALETTE["orange"]],
        width=0.75,
        ax=ax_overhead,
    )
    ax_overhead.set_xlabel("Configuration")
    ax_overhead.set_ylabel("Overhead (ms)")
    ax_overhead.set_title(panel(f"Runtime overhead at N={human_int(max(iterations))}"))
    ax_overhead.tick_params(axis="x", rotation=45)
    for tick in ax_overhead.get_xticklabels():
        tick.set_ha("right")
    ax_overhead.legend(loc="upper left")
    style_axis(ax_overhead)

    fig.suptitle("Experiment A: strong-scaling behavior across workload sizes")
    return FigureSpec("figure_A_scaling", "Experiment A", fig)


def schedule_label(row: pd.Series) -> str:
    work_sharing = str(row["work_sharing"])
    if work_sharing == "parallel_for":
        kind = str(row["schedule_kind"])
        chunk = row["schedule_chunk"]
        if kind == "auto":
            return "for:auto"
        if pd.isna(chunk) or int(chunk) == 0:
            return f"for:{kind}"
        return f"for:{kind},{int(chunk)}"
    if work_sharing == "tasks":
        chunk = row["task_chunk_size"]
        if pd.isna(chunk) or int(chunk) == 0:
            return "tasks:auto"
        return f"tasks:{int(chunk)}"
    return work_sharing


def plot_schedule(df: pd.DataFrame) -> FigureSpec:
    df = df.copy()
    split = has_smt_split(df)
    df["label"] = df.apply(schedule_label, axis=1)
    if split:
        df["label"] = df.apply(lambda row: with_smt(row["label"], row["smt"], True), axis=1)
    ordered = df.sort_values("speedup_pure", ascending=False).reset_index(drop=True)

    n_combos = len(ordered)
    rank_width = max(11.0, 0.32 * n_combos + 3.0)
    fig = plt.figure(figsize=(rank_width, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    ax_rank = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])

    colors = ordered["work_sharing"].map(
        {
            "parallel_for": PALETTE["blue"],
            "manual": PALETTE["green"],
            "tasks": PALETTE["orange"],
        }
    ).fillna(PALETTE["gray"])
    ax_rank.bar(
        np.arange(n_combos),
        ordered["speedup_pure"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    add_median_line(ax_rank, ordered["speedup_pure"])
    ax_rank.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax_rank.set_xticks(np.arange(n_combos), ordered["label"], rotation=70, ha="right", fontsize=9)
    ax_rank.set_xlim(-0.6, n_combos - 0.4)
    ax_rank.set_ylabel("Pure parallel speedup")
    ax_rank.set_title("(a) Schedule-combination distribution")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["blue"], label="parallel for"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["green"], label="manual"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["orange"], label="tasks"),
    ]
    ax_rank.legend(handles=legend_handles, title="work sharing", loc="upper right")
    style_axis(ax_rank)

    group_cols = ["schedule_kind"]
    if split:
        group_cols.append("smt")
    for group_key, part in df[df["work_sharing"].eq("parallel_for")].groupby(group_cols):
        kind = group_key[0] if isinstance(group_key, tuple) else group_key
        smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
        marker = {"static": "o", "dynamic": "s", "guided": "^", "auto": "D"}.get(kind, "o")
        x = part["schedule_chunk"].fillna(0).replace(0, 1)
        ax_scatter.plot(
            x,
            part["speedup_pure"],
            marker=marker,
            linestyle=line_style_for_smt(smt),
            label=with_smt(kind, smt, split, sep=" / "),
        )
    ax_scatter.set_xscale("log")
    ax_scatter.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax_scatter.set_xlabel("Schedule chunk size (0 shown at 1 on log axis)")
    ax_scatter.set_ylabel("Pure parallel speedup")
    ax_scatter.set_title("(b) Chunk-size sensitivity for parallel for")
    ax_scatter.legend(
        title="schedule",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    style_axis(ax_scatter)

    fig.suptitle("Experiment B: scheduling strategy and granularity effects", y=1.02)
    return FigureSpec("figure_B_schedule", "Experiment B", fig)


def affinity_label(row: pd.Series) -> str:
    bind = row["proc_bind"]
    places = row["places"]
    if pd.isna(bind) or bind is False:
        return "bind=false"
    return f"{bind}\n{places}"


def plot_affinity(df: pd.DataFrame) -> FigureSpec:
    df = df.copy()
    split = has_smt_split(df)
    df["label"] = df.apply(affinity_label, axis=1)
    if split:
        df["label"] = df.apply(lambda row: with_smt(row["label"], row["smt"], True), axis=1)
    df = df.sort_values("speedup_pure", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.4]},
        constrained_layout=True,
    )
    ax_bar, ax_time = axes

    indices = np.arange(len(df))
    ax_bar.bar(
        indices,
        df["speedup_pure"],
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.7,
    )
    add_median_line(ax_bar, df["speedup_pure"])
    ax_bar.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    bar_labels = [label.replace("\n", " / ") for label in df["label"]]
    ax_bar.set_xticks(indices, bar_labels, rotation=35, ha="right", fontsize=10)
    ax_bar.set_ylabel("Pure parallel speedup")
    ax_bar.set_title("(a) Affinity policy comparison")
    style_axis(ax_bar)

    palette = plt.cm.tab10(np.linspace(0, 1, max(len(df), 1)))
    for idx, (_, row) in enumerate(df.iterrows()):
        ax_time.scatter(
            row["parallel_ms"],
            row["speedup_pure"],
            s=110,
            color=palette[idx],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=f"{idx + 1}. {row['label'].replace(chr(10), ' / ')}",
        )
        ax_time.annotate(
            str(idx + 1),
            (row["parallel_ms"], row["speedup_pure"]),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=4,
        )
    ax_time.set_xlabel("Parallel region time (ms)")
    ax_time.set_ylabel("Pure parallel speedup")
    ax_time.set_title("(b) Time-to-speedup trade-off")
    ax_time.legend(
        title="configuration",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )
    style_axis(ax_time)

    fig.suptitle("Experiment C: thread affinity and placement")
    return FigureSpec("figure_C_affinity", "Experiment C", fig)


def runtime_label(row: pd.Series) -> str:
    dynamic = "dynamic" if bool(row["dynamic"]) else "fixed"
    label = f"{row['wait_policy']} / {dynamic}"
    return with_smt(label, row["smt"], "smt" in row and row.get("_smt_split", False), sep=" / ")


def plot_runtime(df: pd.DataFrame) -> FigureSpec:
    df = df.copy()
    df["_smt_split"] = has_smt_split(df)
    df["policy"] = df.apply(runtime_label, axis=1)
    policies = sorted(df["policy"].unique())
    colors = dict(zip(policies, plt.cm.Set2(np.linspace(0, 1, len(policies)))))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    ax_speedup, ax_wall, ax_threads, ax_overhead = axes.ravel()

    for policy, part in df.groupby("policy"):
        part = part.sort_values("iterations")
        color = colors[policy]
        ax_speedup.plot(
            part["iterations"],
            part["speedup_pure"],
            marker="o",
            color=color,
            label=policy,
        )
        ax_wall.plot(
            part["iterations"],
            part["wall_time_sec"],
            marker="o",
            color=color,
            label=policy,
        )
        ax_threads.plot(
            part["iterations"],
            part["num_threads_actual"],
            marker="o",
            color=color,
            label=policy,
        )

    ax_speedup.set_xscale("log")
    ax_speedup.set_xlabel("Iterations")
    ax_speedup.set_ylabel("Pure parallel speedup")
    ax_speedup.set_title("(a) Runtime policy speedup")
    ax_speedup.legend(
        ncols=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )
    style_axis(ax_speedup)

    ax_wall.set_xscale("log")
    ax_wall.set_yscale("log")
    ax_wall.set_xlabel("Iterations")
    ax_wall.set_ylabel("Wall time (s)")
    ax_wall.set_title("(b) End-to-end runtime")
    style_axis(ax_wall)

    ax_threads.set_xscale("log")
    ax_threads.set_xlabel("Iterations")
    ax_threads.set_ylabel("Actual worker threads")
    ax_threads.set_title("(c) Runtime-selected team size")
    style_axis(ax_threads)

    latest = df[df["iterations"].eq(df["iterations"].max())].copy()
    latest = latest.sort_values("speedup_pure", ascending=False).reset_index(drop=True)
    overhead = pd.DataFrame(
        {
            "fork/join": latest["fork_join_ms"].to_numpy(),
            "pool create": latest["pool_create_ms"].to_numpy(),
        },
        index=latest["policy"],
    )
    overhead.plot(
        kind="bar",
        stacked=True,
        width=0.72,
        color=[PALETTE["blue"], PALETTE["orange"]],
        ax=ax_overhead,
    )
    ax_overhead.set_xlabel("")
    ax_overhead.set_ylabel("Overhead (ms)")
    ax_overhead.set_title(f"(d) Overhead at N={human_int(df['iterations'].max())}")
    ax_overhead.tick_params(axis="x", rotation=35)
    ax_overhead.legend(loc="upper left")
    style_axis(ax_overhead)

    fig.suptitle("Experiment D: OpenMP runtime policy sensitivity", y=1.02)
    return FigureSpec("figure_D_runtime", "Experiment D", fig)


def plot_cache(df: pd.DataFrame) -> FigureSpec:
    df = df.sort_values("working_set_kib").copy()
    split = has_smt_split(df)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax_cycles, ax_time = axes

    for smt, part in df.groupby("smt" if "smt" in df else lambda _: "single run"):
        ax_cycles.plot(
            part["working_set_kib"],
            part["cycles_per_load"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=PALETTE["blue"] if str(smt) != "SMT on" else PALETTE["orange"],
            label=str(smt) if split else None,
        )
    for kib, label in [
        (32, "L1d 32 KiB"),
        (512, "L2 512 KiB"),
        (32 * 1024, "local L3 32 MiB"),
    ]:
        ax_cycles.axvline(kib, color=PALETTE["gray"], linestyle=":", linewidth=0.9)
        ax_cycles.annotate(
            label,
            xy=(kib, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            rotation=90,
            ha="left",
            va="top",
            fontsize=7.0,
            color=PALETTE["gray"],
        )
    ax_cycles.set_xscale("log", base=2)
    ax_cycles.set_xlabel("Pointer-chase working set (KiB)")
    ax_cycles.set_ylabel("Cycles per dependent load")
    ax_cycles.set_title("(a) Cache-capacity latency curve")
    style_axis(ax_cycles)

    if split:
        ax_cycles.legend(title="mode")
    for smt, part in df.groupby("smt" if "smt" in df else lambda _: "single run"):
        ax_time.plot(
            part["working_set_kib"],
            part["serial_ms"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=PALETTE["blue"] if str(smt) != "SMT on" else PALETTE["orange"],
            label=str(smt) if split else None,
        )
    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Pointer-chase working set (KiB)")
    ax_time.set_ylabel("Serial timed region (ms)")
    ax_time.set_title("(b) Absolute runtime")
    if split:
        ax_time.legend(title="mode")
    style_axis(ax_time)

    fig.suptitle("Experiment E: cache-capacity breakpoints via pointer chasing", y=1.03)
    return FigureSpec("figure_E_cache", "Experiment E", fig)


def cpu_set_label(value: object) -> str:
    text = str(value)
    if text == "0-5":
        return "intra CCD (0-5)"
    if text == "0-2,6-8":
        return "split CCD (0-2,6-8)"
    return text


def plot_l3_ccd(df: pd.DataFrame) -> FigureSpec:
    df = df.sort_values(["cpu_set", "working_set_kib"]).copy()
    split = has_smt_split(df)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax_cycles, ax_speedup = axes
    colors = {
        "0-5": PALETTE["blue"],
        "0-2,6-8": PALETTE["orange"],
    }

    group_cols = ["cpu_set", "smt"] if split else ["cpu_set"]
    for group_key, part in df.groupby(group_cols):
        cpu_set = group_key[0] if isinstance(group_key, tuple) else group_key
        smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
        label = cpu_set_label(cpu_set)
        label = with_smt(label, smt, split, sep=" / ")
        color = colors.get(str(cpu_set), None)
        ax_cycles.plot(
            part["working_set_kib"],
            part["parallel_cycles_per_load"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=color,
            label=label,
        )
        ax_speedup.plot(
            part["working_set_kib"],
            part["speedup_pure"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=color,
            label=label,
        )

    for ax in axes:
        ax.axvline(32 * 1024, color=PALETTE["gray"], linestyle=":", linewidth=0.9)
        ax.annotate(
            "local L3 32 MiB",
            xy=(32 * 1024, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            rotation=90,
            ha="left",
            va="top",
            fontsize=7.0,
            color=PALETTE["gray"],
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Shared pointer-chase working set (KiB)")
        style_axis(ax)

    ax_cycles.set_ylabel("Parallel cycles per dependent load")
    ax_cycles.set_title("(a) Shared L3 pressure")
    ax_cycles.legend()
    ax_speedup.set_ylabel("Pure parallel speedup")
    ax_speedup.set_title("(b) Intra-CCD vs split-CCD speedup")
    ax_speedup.legend()

    fig.suptitle("Experiment F: L3 sharing across CCD placement", y=1.03)
    return FigureSpec("figure_F_l3_ccd", "Experiment F", fig)


def locality_label(row: pd.Series) -> str:
    bind = row["proc_bind"]
    places = row["places"]
    threads = int(row["num_threads"])
    if pd.isna(bind) or bind is False:
        return f"{threads}T bind=false"
    return f"{threads}T {bind}/{places}"


def plot_l1_l2_affinity(df: pd.DataFrame) -> FigureSpec:
    df = df.sort_values(["num_threads", "proc_bind", "places", "working_set_kib"]).copy()
    split = has_smt_split(df)
    df["label"] = df.apply(locality_label, axis=1)
    if split:
        df["label"] = df.apply(lambda row: with_smt(row["label"], row["smt"], True), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), constrained_layout=True)
    ax_12, ax_24 = axes

    for tag, ax, threads in [("(a)", ax_12, 12), ("(b)", ax_24, 24)]:
        part = df[df["num_threads"].eq(threads)]
        for label, group in part.groupby("label"):
            ax.plot(
                group["working_set_kib"],
                group["parallel_cycles_per_load"],
                marker="o",
                label=label,
            )
        for kib, label in [(32, "L1d"), (512, "L2")]:
            ax.axvline(kib, color=PALETTE["gray"], linestyle=":", linewidth=0.9)
            ax.annotate(
                label,
                xy=(kib, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(3, -3),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=7.0,
                color=PALETTE["gray"],
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Per-thread private working set (KiB)")
        ax.set_ylabel("Parallel cycles per dependent load")
        ax.set_title(f"{tag} {threads} OpenMP threads")
        ax.legend(fontsize=6.8)
        style_axis(ax)

    fig.suptitle("Experiment G: L1/L2 locality under OpenMP affinity policies", y=1.03)
    return FigureSpec("figure_G_l1_l2_affinity", "Experiment G", fig)


def plot_bind_compute(df: pd.DataFrame) -> FigureSpec:
    df = df.sort_values(["proc_bind", "num_threads"]).copy()
    split = has_smt_split(df)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax_speedup, ax_time = axes
    colors = {"close": PALETTE["blue"], "spread": PALETTE["orange"]}

    group_cols = ["proc_bind", "smt"] if split else ["proc_bind"]
    for group_key, part in df.groupby(group_cols):
        bind = group_key[0] if isinstance(group_key, tuple) else group_key
        smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
        label = with_smt(bind, smt, split, sep=" / ")
        color = colors.get(str(bind))
        ax_speedup.plot(
            part["num_threads"],
            part["speedup_pure"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=color,
            label=label,
        )
        ax_time.plot(
            part["num_threads"],
            part["parallel_ms"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=color,
            label=label,
        )

    ax_speedup.set_xlabel("OpenMP threads")
    ax_speedup.set_ylabel("Pure parallel speedup")
    ax_speedup.set_title("(a) Compute-bound speedup")
    ax_speedup.legend(title="proc_bind")
    style_axis(ax_speedup)

    ax_time.set_xlabel("OpenMP threads")
    ax_time.set_ylabel("Parallel region time (ms)")
    ax_time.set_title("(b) Absolute runtime")
    ax_time.set_yscale("log")
    ax_time.legend(title="proc_bind")
    style_axis(ax_time)

    fig.suptitle("Experiment H: close vs spread on compute-bound kernel", y=1.03)
    return FigureSpec("figure_H_bind_compute", "Experiment H", fig)


def plot_bind_cache(df: pd.DataFrame) -> FigureSpec:
    df = df.sort_values(["iterations", "proc_bind", "num_threads"]).copy()
    split = has_smt_split(df)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
    colors = {"close": PALETTE["blue"], "spread": PALETTE["orange"]}
    sizes = sorted(df["working_set_kib"].dropna().unique())

    for ax, kib in zip(axes, sizes):
        part = df[df["working_set_kib"].eq(kib)]
        group_cols = ["proc_bind", "smt"] if split else ["proc_bind"]
        for group_key, group in part.groupby(group_cols):
            bind = group_key[0] if isinstance(group_key, tuple) else group_key
            smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
            label = with_smt(bind, smt, split, sep=" / ")
            ax.plot(
                group["num_threads"],
                group["parallel_cycles_per_load"],
                marker="o",
                linestyle=line_style_for_smt(smt),
                color=colors.get(str(bind)),
                label=label,
            )
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("Parallel cycles per load")
        ax.set_title(f"{human_int(kib)} KiB/thread")
        ax.legend(title="proc_bind")
        style_axis(ax)

    fig.suptitle("Experiment I: close vs spread on private cache working sets", y=1.03)
    return FigureSpec("figure_I_bind_cache", "Experiment I", fig)


def false_sharing_label(value: object) -> str:
    text = str(value)
    if text == "false_sharing":
        return "unpadded"
    if text == "false_sharing_padded":
        return "padded"
    return text


def plot_false_sharing(df: pd.DataFrame) -> FigureSpec:
    df = df.sort_values(["iterations", "proc_bind", "work_sharing", "num_threads"]).copy()
    split = has_smt_split(df)
    df["layout"] = df["work_sharing"].map(false_sharing_label)

    largest = df["iterations"].max()
    main = df[df["iterations"].eq(largest)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
    ax_cycles, ax_ratio, ax_bind = axes
    colors = {"unpadded": PALETTE["red"], "padded": PALETTE["green"]}

    close = main[main["proc_bind"].eq("close")]
    group_cols = ["layout", "smt"] if split else ["layout"]
    for group_key, part in close.groupby(group_cols):
        layout = group_key[0] if isinstance(group_key, tuple) else group_key
        smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
        ax_cycles.plot(
            part["num_threads"],
            part["parallel_cycles_per_update"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            color=colors.get(layout),
            label=with_smt(layout, smt, split, sep=" / "),
        )
    ax_cycles.set_xlabel("OpenMP threads")
    ax_cycles.set_ylabel("Parallel cycles per update")
    ax_cycles.set_title(f"(a) close, N={human_int(largest)}")
    ax_cycles.legend(title="slot layout")
    style_axis(ax_cycles)

    ratio_rows = []
    ratio_group_cols = ["proc_bind", "num_threads"] + (["smt"] if split else [])
    for group_key, part in main.groupby(ratio_group_cols):
        bind, threads = group_key[:2] if isinstance(group_key, tuple) else (group_key, None)
        smt = group_key[2] if isinstance(group_key, tuple) and len(group_key) > 2 else "single run"
        values = part.set_index("layout")["parallel_cycles_per_update"]
        if {"unpadded", "padded"}.issubset(values.index):
            ratio_rows.append(
                {
                    "proc_bind": bind,
                    "num_threads": threads,
                    "smt": smt,
                    "ratio": values["unpadded"] / values["padded"],
                }
            )
    ratio = pd.DataFrame(ratio_rows)
    group_cols = ["proc_bind", "smt"] if split else ["proc_bind"]
    for group_key, part in ratio.groupby(group_cols):
        bind = group_key[0] if isinstance(group_key, tuple) else group_key
        smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
        ax_ratio.plot(
            part["num_threads"],
            part["ratio"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            label=with_smt(bind, smt, split, sep=" / "),
        )
    ax_ratio.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax_ratio.set_xlabel("OpenMP threads")
    ax_ratio.set_ylabel("Unpadded / padded cycles")
    ax_ratio.set_title("(b) False-sharing penalty")
    ax_ratio.legend(title="proc_bind")
    style_axis(ax_ratio)

    unpadded = main[main["layout"].eq("unpadded")]
    group_cols = ["proc_bind", "smt"] if split else ["proc_bind"]
    for group_key, part in unpadded.groupby(group_cols):
        bind = group_key[0] if isinstance(group_key, tuple) else group_key
        smt = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else "single run"
        ax_bind.plot(
            part["num_threads"],
            part["parallel_cycles_per_update"],
            marker="o",
            linestyle=line_style_for_smt(smt),
            label=with_smt(bind, smt, split, sep=" / "),
        )
    ax_bind.set_xlabel("OpenMP threads")
    ax_bind.set_ylabel("Parallel cycles per update")
    ax_bind.set_title("(c) Unpadded close vs spread")
    ax_bind.legend(title="proc_bind")
    style_axis(ax_bind)

    fig.suptitle("Experiment J: intentionally induced false sharing", y=1.03)
    return FigureSpec("figure_J_false_sharing", "Experiment J", fig)


def plot_summary(data: dict[str, pd.DataFrame]) -> FigureSpec:
    rows = []
    for experiment, df in data.items():
        if experiment in {"E", "F", "G", "I", "J"}:
            continue
        columns = ["speedup_pure", "parallel_ms", "fork_join_ms", "pool_create_ms"]
        if "smt" in df:
            columns.append("smt")
        part = df[columns].copy()
        part["experiment"] = experiment
        part["series"] = part["experiment"]
        if "smt" in part and part["smt"].nunique(dropna=True) > 1:
            part["series"] = part["experiment"] + "\n" + part["smt"].str.replace("SMT ", "", regex=False)
        part["runtime_overhead_ms"] = part["fork_join_ms"] + part["pool_create_ms"]
        rows.append(part)
    summary = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax_dist, ax_trade = axes
    experiments = sorted(summary["series"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(experiments), 1)))
    distributions = [
        summary.loc[summary["series"].eq(experiment), "speedup_pure"].to_numpy()
        for experiment in experiments
    ]

    box = ax_dist.boxplot(
        distributions,
        labels=experiments,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": PALETTE["black"], "linewidth": 1.2},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    rng = np.random.default_rng(7)
    for idx, (experiment, color) in enumerate(zip(experiments, colors), start=1):
        values = summary.loc[summary["series"].eq(experiment), "speedup_pure"].to_numpy()
        jitter = rng.normal(loc=idx, scale=0.035, size=len(values))
        ax_dist.scatter(
            jitter,
            values,
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )
    ax_dist.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax_dist.set_xlabel("Experiment")
    ax_dist.set_ylabel("Pure parallel speedup")
    ax_dist.set_title("(a) Distribution of measured combinations")
    style_axis(ax_dist)

    for experiment, color in zip(experiments, colors):
        part = summary[summary["series"].eq(experiment)]
        ax_trade.scatter(
            part["parallel_ms"],
            part["speedup_pure"],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.85,
            label=experiment,
            zorder=3,
        )
    ax_trade.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax_trade.set_xscale("log")
    ax_trade.set_xlabel("Parallel region time (ms)")
    ax_trade.set_ylabel("Pure parallel speedup")
    ax_trade.set_title("(b) Time-speedup trade-off across combinations")
    ax_trade.legend(title="experiment")
    style_axis(ax_trade)

    fig.suptitle("Cross-experiment comparison of OpenMP configuration effects", y=1.03)
    return FigureSpec("figure_summary", "Summary", fig)


def save_figure(spec: FigureSpec, out_dir: Path, formats: list[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{spec.name}.{fmt}"
        save_kwargs = {"format": fmt}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        spec.fig.savefig(path, **save_kwargs)
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    data = read_results(args.results_dir)

    figures = [
        plot_scaling(data["A"]),
        plot_schedule(data["B"]),
        plot_affinity(data["C"]),
        plot_runtime(data["D"]),
        plot_cache(data["E"]),
        plot_l3_ccd(data["F"]),
        plot_l1_l2_affinity(data["G"]),
        plot_bind_compute(data["H"]),
        plot_bind_cache(data["I"]),
        plot_false_sharing(data["J"]),
        plot_summary(data),
    ]

    for spec in figures:
        save_figure(spec, args.out_dir, args.formats, args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
