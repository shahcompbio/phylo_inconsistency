"""Plotting functions for the counterexample generator notebook."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from .plotting import draw_gray_topology, draw_tstar_vertical

_LEGEND_HANDLES = [
    Line2D([0], [0], color="#d62728", lw=3, label="Δ ≪ 0  (favor candidate)"),
    Line2D([0], [0], color="#aaaaaa", lw=3, label="Δ ≈ 0  (neutral)"),
    Line2D([0], [0], color="#1f77b4", lw=3, label="Δ ≫ 0  (favor T*)"),
]


def _draw_nni_row(
    axes_row, case, results, row_label="",
    gamma=0.15, max_lw=2.0, min_lw=0.5,
    lw_h=4.0, lw_v=4.0, lw_hl=6.0,
):
    """Fill one row of axes: gray candidate topology + T* panels for each prior."""
    topo_star      = case["topo_star"]
    topo_nbr       = case["topo_nbr"]
    deltas         = case["deltas"]
    p_eq           = case["p_eq"]
    diff_eq        = case["diff_eq"]
    m              = case["m"]
    edge_key       = case["edge_key"]
    tree_star_plot = case["tree_star"]

    draw_gray_topology(
        axes_row[0], topo_nbr, n_edges=m, highlight_edge=edge_key,
        lw_h=lw_h, lw_v=lw_v, lw_hl=lw_hl,
    )
    axes_row[0].set_title(f"Candidate\n(NNI on '{edge_key}')", fontsize=8, fontweight="bold")

    draw_tstar_vertical(
        axes_row[1], topo_star, p_eq, deltas,
        gamma=gamma, labels=False, max_lw=max_lw, min_lw=min_lw,
    )
    axes_row[1].set_title(
        f"T* with Δ_e\n(equal,  Σ pΔ = {diff_eq:+.4f})",
        fontsize=8, fontweight="bold",
    )

    nudged_panels = [
        (r[0], r[3], sum(r[3][k] * deltas[k] for k in tree_star_plot))
        for r in results if len(r) == 9 and r[-1]
    ]
    for col_idx, (lbl, p_inc, diff_inc) in enumerate(nudged_panels):
        ax = axes_row[2 + col_idx]
        draw_tstar_vertical(
            ax, topo_star, p_inc, deltas,
            gamma=gamma, labels=False, leaf_labels=False,
            max_lw=max_lw, min_lw=min_lw,
        )
        inc_label = "  ✓" if diff_inc < 0 else ""
        ax.set_title(
            f"{lbl}\nΣ p_e Δ_e = {diff_inc:+.4f}{inc_label}",
            fontsize=8, fontweight="bold",
            color="#d62728" if diff_inc < 0 else "black",
        )

    if row_label:
        axes_row[0].set_ylabel(row_label, fontsize=9, fontweight="bold", labelpad=8)


def plot_nni_panel(
    cases, all_results, n, alpha, beta, seed,
    row_height=6.5, gamma=0.15, max_lw=2.0, min_lw=0.5,
    pdf=None, show=True,
):
    """Draw a grid of NNI cases: one row per case, five columns.

    Columns: (1) gray candidate topology, (2) T* with equal weights,
    (3–5) T* with KL-nudged distributions for each of three priors.

    Parameters
    ----------
    cases       : list of dicts from counterexample_analysis.make_nni_case
    all_results : list of phase-3 result lists from counterexample_analysis.run_phase3
    n           : int   number of leaves (for the figure title)
    alpha, beta : float noise rates
    seed        : int   RNG seed
    row_height  : float figure height per row in inches
    gamma       : float color saturation exponent for draw_tstar_vertical
    max_lw, min_lw : float line-width bounds
    pdf         : PdfPages or None
    show        : bool
    """
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, row_height * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, (case, res) in enumerate(zip(cases, all_results)):
        _draw_nni_row(
            axes[i], case, res, row_label=f"NNI neighbor {i + 1}",
            gamma=gamma, max_lw=max_lw, min_lw=min_lw,
        )

    fig.legend(handles=_LEGEND_HANDLES, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"n={n}  |  α={alpha}, β={beta}  |  seed={seed}\n"
        "branch length ∝ p_e  ·  width ∝ |Δ_e|  ·  colour = sign(Δ_e)",
        fontsize=10,
    )
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_nni_subset(
    cases, all_results, subset_idx, n, alpha, beta, seed,
    gamma=0.15,
    pdf=None, show=True,
):
    """Compact paper figure matching the figure_v3 layout.

    Layout: (n_prior + 1) rows × (n_nni + 1) cols.
    - Top-left:        T* topology (gray, leaf labels)
    - Top row cols 1+: candidate tree topologies (gray, leaf labels)
    - Left col rows 1+: T* with canonical prior (gray, no leaf labels, prior label)
    - Body:            T* with KL-nudged distribution (colored + widths)

    Parameters
    ----------
    subset_idx : list of int  0-based indices into cases / all_results
    gamma      : float  color compression exponent
    pdf        : PdfPages or None
    show       : bool
    """
    subset_cases   = [cases[i]       for i in subset_idx]
    subset_results = [all_results[i] for i in subset_idx]

    feasible = [r for r in subset_results[0] if r[-1]]
    prior_labels = [r[0] for r in feasible]
    n_nni   = len(subset_idx)
    n_prior = len(prior_labels)

    height = (n_prior + 1) * 0.7
    width  = (n_nni + 1) * 1.2

    fig = plt.figure(figsize=(width, height))
    gs  = gridspec.GridSpec(
        n_prior + 1, n_nni + 1, figure=fig,
        width_ratios=[1] + [1] * n_nni,
        height_ratios=[1.2] + [1] * n_prior,
        hspace=0.35, wspace=0.2,
    )

    def _compress_x(ax, factor=1.2):
        xl = ax.get_xlim()
        cx = (xl[0] + xl[1]) / 2
        hw = (xl[1] - xl[0]) / 2
        ax.set_xlim(cx - hw * factor, cx + hw * factor)

    # ── Top-left: T* (gray) ──────────────────────────────────────────────────
    ref_case   = subset_cases[0]
    ref_topo   = ref_case["topo_star"]
    ref_tree   = ref_case["tree_star"]
    ref_deltas = ref_case["deltas"]
    m          = ref_case["m"]
    p_unif     = {k: 1 / m for k in ref_tree}

    ax_tstar = fig.add_subplot(gs[0, 0])
    draw_tstar_vertical(
        ax_tstar, ref_topo, p_unif, ref_deltas,
        gamma=gamma, labels=False, leaf_labels=True,
        max_lw=2, min_lw=0.5, h_lw=0.5, fontscale=0.4, leafsize=1,
        color=False, widths=False,
    )
    ax_tstar.set_title("$T^*$", fontsize=7, pad=0)

    # ── Top row: candidate trees (gray) ─────────────────────────────────────
    for col, case in enumerate(subset_cases):
        ax = fig.add_subplot(gs[0, col + 1])
        draw_tstar_vertical(
            ax, case["topo_nbr"], case["p_eq"], case["deltas"],
            gamma=gamma, labels=False, leaf_labels=True,
            max_lw=2, min_lw=0.5, h_lw=0.5, fontscale=0.4, leafsize=1,
            color=False, widths=False,
        )
        ax.set_title(f"$T_{col + 1}$", fontsize=6, pad=0)

    # ── Left column: canonical priors on T* (gray) ──────────────────────────
    canonical_priors = [(r[0], r[1]) for r in subset_results[0] if r[-1]]
    for row, (lbl, prior) in enumerate(canonical_priors):
        ax = fig.add_subplot(gs[row + 1, 0])
        draw_tstar_vertical(
            ax, ref_topo, prior, ref_deltas,
            gamma=gamma, labels=False, leaf_labels=False,
            max_lw=6, min_lw=0.5, h_lw=0.4, leafsize=0.8,
            color=False, widths=False,
        )
        _compress_x(ax, factor=1.2)
        ax.set_title(lbl, fontsize=5, loc="left", pad=-2)

    # ── Body: nudged distributions (colored) ────────────────────────────────
    for col, (case, res) in enumerate(zip(subset_cases, subset_results)):
        case_topo   = case["topo_star"]
        case_deltas = case["deltas"]
        nudged = [r[3] for r in res if len(r) == 9 and r[-1]]
        for row, p_inc in enumerate(nudged):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            draw_tstar_vertical(
                ax, case_topo, p_inc, case_deltas,
                gamma=gamma, labels=False, leaf_labels=False,
                max_lw=6, min_lw=0.5, h_lw=0.4, leafsize=0.8,
                color=True, widths=True,
            )
            _compress_x(ax, factor=1.2)

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
