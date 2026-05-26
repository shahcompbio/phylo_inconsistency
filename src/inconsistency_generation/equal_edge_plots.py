"""Plotting functions for the equal-edge-weight analysis (Figure A1)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from inconsistency_generation.equal_edge_analysis import (
    caterpillar_splits,
    nni_cherry_caterpillar_splits,
    three_leaf_subtree_splits,
)
from inconsistency_generation.plotting import draw_tstar_vertical
from inconsistency_generation.tree_utils import edge_dict_to_topology


def _finalize(fig, pdf=None, show: bool = True) -> None:
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Topology schematics ────────────────────────────────────────────────────────

def plot_example_topologies(n: int = 6, pdf=None, show: bool = True) -> None:
    """Three-panel vertical schematic of T1, T2, T3 for a given n."""
    example_specs = [
        (r"$T_1$", caterpillar_splits(n), "internal_4"),
        (r"$T_2$", nni_cherry_caterpillar_splits(n), "internal_3"),
        (r"$T_3$", three_leaf_subtree_splits(n), "internal_3"),
    ]
    leaf_labels = [chr(ord("A") + i) if n <= 26 else str(i) for i in range(n)]
    leaf_colors = {
        leaf_labels[n - 3]: "#2166AC",
        leaf_labels[n - 2]: "#B2182B",
        leaf_labels[n - 1]: "#8C00C8",
    }

    fig, axes = plt.subplots(3, 1, figsize=(1, 2.5))
    for ax, (label, edge_dict, dashed_edge) in zip(axes, example_specs):
        topology = edge_dict_to_topology(edge_dict)
        k = len(edge_dict)
        draw_tstar_vertical(
            ax, topology,
            {key: 1.0 / k for key in edge_dict},
            {key: 0.0 for key in edge_dict},
            color=False, widths=False, labels=False, leaf_labels=False,
            max_lw=1, min_lw=1, h_lw=1, leafsize=8, fontscale=0.55,
            dashed_edges={dashed_edge},
            leaf_colors=leaf_colors,
        )
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=10)

    plt.subplots_adjust(hspace=0.20)
    _finalize(fig, pdf=pdf, show=show)


# ── Heatmap panel ──────────────────────────────────────────────────────────────

def plot_heatmap_panel(
    analysis_results: dict,
    analyses: list[dict],
    selected_n_values: list[int],
    figsize: tuple[float, float],
    pdf=None,
    show: bool = True,
) -> None:
    """Contourf heatmap of Δ(candidate, true) over (α, β) space for each n."""
    font_size = 8
    global_vmax = max(
        np.nanmax(np.abs(diff))
        for _, _, _, diffs in analysis_results.values()
        for n, diff in diffs.items()
        if n in selected_n_values
    )
    norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0.0, vmax=global_vmax)
    levels = np.linspace(-global_vmax, global_vmax, 31)

    n_rows = len(analyses)
    n_cols = len(selected_n_values) + 1
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(
        n_rows, n_cols,
        width_ratios=[4] * len(selected_n_values) + [0.25],
        wspace=0.1, hspace=0.25,
    )

    for row, analysis in enumerate(analyses):
        alpha_vals, beta_vals, _, diffs = analysis_results[analysis["name"]]
        for col, n in enumerate(selected_n_values):
            ax = fig.add_subplot(grid[row, col])
            diff = diffs[n]
            pct = 100.0 * np.nanmean(diff < 0)
            ax.contourf(beta_vals, alpha_vals, diff, levels=levels, cmap="RdBu_r", norm=norm)
            ax.contour(beta_vals, alpha_vals, diff, levels=[0], colors="#000077", linewidths=0.8)
            ax.text(
                analysis["annotation_beta"], analysis["annotation_alpha"],
                f"{pct:.1f}%", ha="center", va="center", fontsize=6,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "none",
                      "alpha": 0.85, "edgecolor": "none"},
            )
            if row < n_rows - 1:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(f"$n={n}$", fontsize=font_size, pad=2)
            if col > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=font_size)

        cbar_ax = fig.add_subplot(grid[row, -1])
        mappable = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=cbar_ax)
        cb.set_label(analysis["colorbar_label"], fontsize=font_size, labelpad=0)
        cb.ax.tick_params(labelsize=font_size, pad=0)

    fig.supxlabel(r"FP rate $\beta$", fontsize=font_size, y=-0.05)
    fig.supylabel(r"FN rate $\alpha$", fontsize=font_size, x=0.035)
    _finalize(fig, pdf=pdf, show=show)


# ── Area-by-n line plot (T1/T2) ────────────────────────────────────────────────

def plot_area_by_n_figure(
    analysis_results: dict,
    analyses: list[dict],
    n_values: list[int],
    pdf=None,
    show: bool = True,
) -> None:
    """Fraction of (α, β) space where each direction is inconsistent, by n (T1/T2 only)."""
    colors = ["#7A3E9D", "#1B9E77"]
    markers = ["o", "s"]

    area_by_name = {}
    for analysis in analyses:
        _, _, _, diffs = analysis_results[analysis["name"]]
        area_by_name[analysis["name"]] = [
            float(100.0 * np.nanmean(diffs[n] < 0)) for n in n_values
        ]

    sum_area = [
        area_by_name[analyses[0]["name"]][i] + area_by_name[analyses[1]["name"]][i]
        for i in range(len(n_values))
    ]

    fig, ax = plt.subplots(figsize=(2, 2))
    for analysis, color, marker in zip(analyses, colors, markers):
        ax.plot(n_values, area_by_name[analysis["name"]],
                f"{marker}:", linewidth=1, markersize=3, color=color,
                label=f"{analysis['colorbar_label']} area")
    ax.plot(n_values, sum_area, "^:", linewidth=1, markersize=3.5, color="#4D4D4D", label="Union")

    ax.set_ylim(0, 100)
    ax.set_xlabel(r"$n$", fontsize=7)
    ax.set_ylabel("Inconsistent area (%)", fontsize=7, labelpad=0)
    ax.set_xticks(n_values, n_values, fontsize=7)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _finalize(fig, pdf=pdf, show=show)


# ── All-directions area line plot ──────────────────────────────────────────────

def plot_all_directions_area(
    analysis_results: dict,
    t2_t3_results: dict,
    analyses: list[dict],
    analyses_t2_t3: list[dict],
    n_values: list[int],
    n_values_t3: list[int],
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    pdf=None,
    show: bool = True,
) -> None:
    """Fraction of (α, β) space inconsistent in each of the four directions, plus union."""
    all_specs = [
        (analyses[0]["colorbar_label"],      analysis_results, analyses[0]["name"],      n_values,    "#7A3E9D", "o"),
        (analyses[1]["colorbar_label"],      analysis_results, analyses[1]["name"],      n_values,    "#1B9E77", "s"),
        (analyses_t2_t3[0]["colorbar_label"], t2_t3_results,   analyses_t2_t3[0]["name"], n_values_t3, "#E6AB02", "^"),
        (analyses_t2_t3[1]["colorbar_label"], t2_t3_results,   analyses_t2_t3[1]["name"], n_values_t3, "#D95F02", "D"),
    ]

    n_union = sorted(set(n_values) | set(n_values_t3))
    union_pct = []
    for n in n_union:
        any_neg = np.zeros((len(alpha_vals), len(beta_vals)), dtype=bool)
        for _, results_dict, name, nv, _, _ in all_specs:
            _, _, _, diffs = results_dict[name]
            if n in diffs:
                any_neg |= (diffs[n] < 0)
        union_pct.append(float(100.0 * np.nanmean(any_neg)))

    fig, ax = plt.subplots(figsize=(2, 2.5))
    for label, results_dict, name, nv, color, marker in all_specs:
        _, _, _, diffs = results_dict[name]
        pcts = [float(100.0 * np.nanmean(diffs[n] < 0)) for n in nv]
        ax.plot(nv, pcts, f"{marker}:", linewidth=1, markersize=3, alpha=0.8,
                color=color, label=label, clip_on=False)
    ax.plot(n_union, union_pct, "P:", linewidth=1.2, markersize=4, alpha=0.8,
            color="#4D4D4D", label="Union")

    ax.set_ylim(-5, 100)
    ax.axhline(0, color="k", linewidth=1, linestyle="-", alpha=0.4, zorder=-1)
    ax.set_xlabel(r"$n$", fontsize=8)
    ax.set_ylabel("Inconsistent area (%)", fontsize=8, labelpad=0)
    ax.set_xticks(n_union, n_union, fontsize=7)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], fontsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _finalize(fig, pdf=pdf, show=show)


# ── Combined heatmap (paper figure) ───────────────────────────────────────────

def plot_combined_heatmap(
    analysis_results: dict,
    t2_t3_results: dict,
    alpha_vals: np.ndarray,
    selected_n: list[int] | None = None,
    pdf=None,
    show: bool = True,
) -> None:
    """Combined heatmap: the interesting direction from each NNI pair, n ∈ {5, 7, 9}."""
    if selected_n is None:
        selected_n = [5, 7, 9]

    interesting = [
        ("cherry_cat_minus_caterpillar", analysis_results, r"$\Delta(T_2, T_1)$", 0.16, 0.40),
        ("three_leaf_minus_cherry_cat",  t2_t3_results,    r"$\Delta(T_3, T_2)$", 0.35, 0.25),
    ]

    font_size = 8
    global_vmax = max(
        np.nanmax(np.abs(results_dict[name][3][n]))
        for name, results_dict, _, _, _ in interesting
        for n in selected_n
    )
    levels = np.linspace(-global_vmax, global_vmax, 31)
    cb_ticks = [-global_vmax, 0.0, global_vmax]
    vmax_str = f"{global_vmax:.1f}"
    cb_ticklabels = [f"-{vmax_str}", "0", vmax_str]

    n_rows = len(interesting)
    n_cols = len(selected_n) + 1
    fig = plt.figure(figsize=(3.5, 2.5))
    grid = gridspec.GridSpec(
        n_rows, n_cols,
        width_ratios=[4] * len(selected_n) + [0.25],
        wspace=0.12, hspace=0.4,
    )

    for row, (name, results_dict, colorbar_label, ann_alpha, ann_beta) in enumerate(interesting):
        _, beta_vals, _, diffs = results_dict[name]
        av = alpha_vals
        for col, n in enumerate(selected_n):
            ax = fig.add_subplot(grid[row, col])
            diff = diffs[n]
            norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0.0, vmax=global_vmax)
            ax.contourf(beta_vals, av, diff, levels=levels, cmap="RdBu_r", norm=norm)
            ax.contour(beta_vals, av, diff, levels=[0], colors="#000077", linewidths=0.8)
            if row < n_rows - 1:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(f"$n={n}$", fontsize=font_size, pad=2)
            if col > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=font_size)

        cbar_ax = fig.add_subplot(grid[row, -1])
        norm_cb = TwoSlopeNorm(vmin=-global_vmax, vcenter=0.0, vmax=global_vmax)
        mappable = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_cb)
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=cbar_ax)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(cb_ticklabels)
        cb.ax.tick_params(labelsize=font_size, pad=1)
        cb.ax.set_title(colorbar_label, fontsize=font_size, pad=6)

    fig.supxlabel(r"FP rate $\beta$", fontsize=font_size, y=-0.05)
    fig.supylabel(r"FN rate $\alpha$", fontsize=font_size, x=0.01)
    _finalize(fig, pdf=pdf, show=show)
