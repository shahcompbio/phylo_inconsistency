"""Plotting functions for the NNI margin across-n analysis (notebook 08)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from inconsistency_generation.nni_margin_analysis import WINNING_NEIGHBOR_FEATURE_LABELS

# ── Display-name overrides for prior keys ─────────────────────────────────────
PRIOR_DISPLAY_NAMES: dict[str, str] = {
    "Flat": "Non-informative",
    "Near equal": "Balanced",
}

FEATURE_LABEL_MAP: dict[str, str] = {
    "max_split_size": "Max split size",
    "n_cherries": "# Cherries",
    "depth": "Depth",
    "top_balance": "Top balance",
    "colless_index": "Colless index",
    "root_has_cherry": "Root cherry",
    "root_has_singleton": "Root singleton",
    "n_full_internal": "# Full internal",
}

_FOCUS_PRIORS = ["Flat", "Heavy trunk", "Clade weighted"]
_PRIOR_COLORS = {
    "Flat": "#576872",
    "Heavy trunk": "#a4c116",
    "Clade weighted": "#2e7e08",
}


def _n_colors(n_leaves_list: list[int]) -> dict[int, tuple]:
    cmap = plt.cm.magma
    return {n: cmap(i / (len(n_leaves_list) + 1)) for i, n in enumerate(n_leaves_list)}


def finalize_figure(fig, pdf=None, show: bool = True) -> None:
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_margin_across_n_figure(
    results_by_n: dict,
    win_rates_by_n: dict,
    n_leaves_list: list[int],
    prior_display_names: dict[str, str] | None = None,
    pdf=None,
    show: bool = True,
) -> None:
    """Margin distribution histograms (top) and per-tree loss-rate scatter (bottom)."""
    if prior_display_names is None:
        prior_display_names = PRIOR_DISPLAY_NAMES
    colors = _n_colors(n_leaves_list)
    prior_list = list(next(iter(results_by_n.values())).keys())

    fig = plt.figure(figsize=(10, 4.2))
    gs = mgridspec.GridSpec(2, len(prior_list), figure=fig, hspace=0.32, wspace=0.3)
    axes = np.empty((2, len(prior_list)), dtype=object)
    for col in range(len(prior_list)):
        axes[0, col] = fig.add_subplot(gs[0, col])
        axes[1, col] = fig.add_subplot(gs[1, col])

    all_vals = np.concatenate([results_by_n[n][p] for n in n_leaves_list for p in prior_list])
    bins = np.linspace(np.percentile(all_vals, 0.1), np.percentile(all_vals, 99.9), 80)

    for i, prior_name in enumerate(prior_list):
        ax = axes[0, i]
        leg_handles = []
        for n in n_leaves_list:
            vals = results_by_n[n][prior_name]
            frac_neg = (vals < 0).mean()
            ax.hist(vals, bins=bins, color=colors[n], lw=0, density=True, alpha=0.05, zorder=n, clip_on=False)
            ax.hist(vals, bins=bins, color=colors[n], histtype="step", lw=1.1, density=True, alpha=0.8, zorder=n, clip_on=False)
            leg_handles.append(plt.Line2D([0], [0], color=colors[n], lw=1.1, label=f"n={n}  ({frac_neg:.1%})"))
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.8, zorder=20)
        ax.set_title(prior_display_names.get(prior_name, prior_name), fontsize=7, pad=2)
        ax.set_xlabel(r"$\min_T\; \Delta_{\mathbf{p}}(T, T^*)$", fontsize=7, labelpad=1)
        ax.set_ylabel("density" if i == 0 else "", fontsize=7, labelpad=1)
        if i > 0:
            ax.set_yticklabels([])
        ax.set_xlim(-0.12, 0.25)
        ax.tick_params(labelsize=7)
        ax.legend(handles=leg_handles, fontsize=7, frameon=False, handlelength=1.0,
                  title="# leaves (% ≤ 0)", title_fontsize=7, loc=(0.6, 0.3))

    rng = np.random.default_rng(1)
    for i, prior_name in enumerate(prior_list):
        ax = axes[1, i]
        for j, n in enumerate(n_leaves_list):
            lr = 1 - win_rates_by_n[n][prior_name]
            jitter = rng.uniform(-0.18, 0.18, size=len(lr))
            ax.scatter(np.full(len(lr), j) + jitter, lr, color=colors[n], s=2, lw=0,
                       zorder=3, alpha=0.8, edgecolor="none", clip_on=False)
        ax.set_xticks(range(len(n_leaves_list)))
        ax.set_xticklabels([f"n={n}" for n in n_leaves_list], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("True tree loss rate" if i == 0 else "", fontsize=7, labelpad=1)
        if i > 0:
            ax.set_yticklabels([])
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.tick_params(labelsize=7)

    sns.despine(fig=fig)
    fig.tight_layout()
    finalize_figure(fig, pdf=pdf, show=show)


def plot_topology_figure(
    topology_analysis: dict,
    win_rates_by_n: dict,
    n_leaves_list: list[int],
    prior_display_names: dict[str, str] | None = None,
    pdf=None,
    show: bool = True,
) -> None:
    """Pairwise loss-rate scatter, top-balance feature plot, and feature importance panel."""
    if prior_display_names is None:
        prior_display_names = PRIOR_DISPLAY_NAMES
    colors = _n_colors(n_leaves_list)
    focus_n = topology_analysis["focus_n"]
    X = topology_analysis["X"]
    feature_names = topology_analysis["feature_names"]
    focus_targets = topology_analysis["focus_targets"]
    importances = topology_analysis["importances"]
    r2_scores = topology_analysis["r2_scores"]

    feature_group_order = list(reversed([
        "top_balance", "root_has_singleton", "root_has_cherry",
        "depth", "max_split_size",
        "n_cherries", "n_full_internal", "colless_index",
    ]))
    fn_list = list(feature_names)
    display_order = [fn_list.index(f) for f in feature_group_order if f in fn_list]

    scatter_pairs = [
        ("Flat", "Heavy trunk"),
        ("Flat", "Clade weighted"),
        ("Heavy trunk", "Clade weighted"),
    ]
    prior_order_imp = ["Clade weighted", "Heavy trunk", "Flat"]

    fig = plt.figure(figsize=(8, 3.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.1, 1],
                          height_ratios=[1, 0.6], hspace=0.3, wspace=0.45)
    ax_scat = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_feat = [fig.add_subplot(gs[1, c]) for c in range(3)]
    ax_imp = fig.add_subplot(gs[:, 4])
    fig.add_subplot(gs[:, 3]).axis("off")

    for ax, (px, py) in zip(ax_scat, scatter_pairs):
        for n in n_leaves_list:
            ax.scatter(1 - win_rates_by_n[n][px], 1 - win_rates_by_n[n][py],
                       color=colors[n], s=2, alpha=0.6, lw=0, edgecolors="none", zorder=n)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(prior_display_names.get(px, px), fontsize=7, labelpad=1)
        ax.set_ylabel(prior_display_names.get(py, py), fontsize=7, labelpad=1)
        ax.tick_params(labelsize=7)
        ax.set_yticks([0, 0.5, 1]) if ax is ax_scat[0] else ax.set_yticks([0, 0.5, 1], [])
        ax.axhline(0, color="gray", lw=0.5, ls=":", alpha=0.5, zorder=20)
        ax.axvline(0, color="gray", lw=0.5, ls=":", alpha=0.5, zorder=20)

    handles_n = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[n], markersize=5, label=f"n={n}")
        for n in n_leaves_list
    ]
    ax_scat[0].legend(handles=handles_n, fontsize=6, frameon=False,
                      loc="upper right", title="# leaves", title_fontsize=6)

    for ax, target_name in zip(ax_feat, _FOCUS_PRIORS):
        x_vals = X["top_balance"].values
        y_vals = 1 - focus_targets[target_name]
        ax.scatter(x_vals, y_vals, s=4, alpha=0.6, lw=0, color=_PRIOR_COLORS[target_name])
        smoothed = lowess(y_vals, x_vals, frac=0.5, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1, alpha=0.5)
        ax.set_xlabel("Top balance", fontsize=7, labelpad=1)
        ax.set_ylabel("$T^*$ Loss rate" if ax is ax_feat[0] else "", fontsize=7, labelpad=0)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1 / (2 * focus_n), 1 / 2)
        ax.set_yticks([0, 0.5, 1]) if ax is ax_feat[0] else ax.set_yticks([0, 0.5, 1], [])
        ax.set_xticks(
            [i / focus_n for i in range(1, 1 + focus_n // 2)],
            [f"{i}|{focus_n - i}" for i in range(1, 1 + focus_n // 2)],
        )
        ax.axhline(0, color="gray", lw=0.5, ls=":", alpha=0.5, zorder=20)

    bar_h = 0.2
    y_pos = np.arange(len(display_order))
    for gi, target_name in enumerate(prior_order_imp):
        vals = [importances[target_name][i] for i in display_order]
        offset = (gi - (len(prior_order_imp) - 1) / 2) * bar_h
        ax_imp.barh(y_pos + offset, vals, height=bar_h * 0.9,
                    color=_PRIOR_COLORS[target_name], alpha=0.85, lw=0,
                    label=f"{prior_display_names.get(target_name, target_name)}  (R²={r2_scores[target_name]:.2f})")

    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels([FEATURE_LABEL_MAP.get(feature_names[i], feature_names[i]) for i in display_order], fontsize=7)
    ax_imp.set_ylim(-0.7, len(display_order) + 1.8)
    ax_imp.vlines(0, -0.5, len(display_order) - 0.5, color="black", lw=0.8, ls="--", alpha=0.6)
    ax_imp.set_xlabel("Feature importance", fontsize=7)
    ax_imp.tick_params(labelsize=7)
    handles_imp, labels_imp = ax_imp.get_legend_handles_labels()
    ax_imp.legend(handles_imp[::-1], labels_imp[::-1], fontsize=7, frameon=False, loc=(-0.32, 0.8))

    sns.despine(fig=fig)
    ax_imp.spines["left"].set_bounds(-0.5, len(display_order) - 0.5)
    finalize_figure(fig, pdf=pdf, show=show)


def plot_all_topology_features(
    topology_analysis: dict,
    prior_display_names: dict[str, str] | None = None,
    pdf=None,
    show: bool = True,
) -> None:
    """Grid of every topology feature vs. true-tree loss rate for each prior."""
    if prior_display_names is None:
        prior_display_names = PRIOR_DISPLAY_NAMES
    focus_n = topology_analysis["focus_n"]
    X = topology_analysis["X"]
    feature_names = topology_analysis["feature_names"]
    focus_targets = topology_analysis["focus_targets"]
    all_feature_names = list(feature_names)

    fig, axes = plt.subplots(
        len(all_feature_names), len(_FOCUS_PRIORS),
        figsize=(8.5, 1.55 * len(all_feature_names)),
        squeeze=False,
    )
    rng = np.random.default_rng(7)

    for row, feature_name in enumerate(all_feature_names):
        x_vals_raw = X[feature_name].to_numpy()
        unique_vals = np.unique(x_vals_raw)
        if len(unique_vals) <= 10:
            x_span = max(unique_vals.max() - unique_vals.min(), 1.0)
            x_vals_plot = x_vals_raw + rng.uniform(-0.025 * x_span, 0.025 * x_span, size=len(x_vals_raw))
        else:
            x_vals_plot = x_vals_raw

        for col, target_name in enumerate(_FOCUS_PRIORS):
            ax = axes[row, col]
            y_vals = 1 - focus_targets[target_name]
            ax.scatter(x_vals_plot, y_vals, s=2, alpha=0.8, lw=0, color=_PRIOR_COLORS[target_name])
            smoothed = lowess(y_vals, x_vals_raw, frac=0.5, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1, alpha=0.55)
            ax.set_ylim(0, 1.05)
            ax.axhline(0, color="gray", lw=0.5, ls=":", alpha=0.5, zorder=20)
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(prior_display_names.get(target_name, target_name), fontsize=7, pad=2)
            if col == 0:
                ax.set_ylabel("$T^*$ loss rate", fontsize=7, labelpad=1)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(FEATURE_LABEL_MAP.get(feature_name, feature_name) if col == 1 else "",
                          fontsize=7, labelpad=1)

            if feature_name == "top_balance":
                ax.set_xlim(1 / (2 * focus_n), 1 / 2)
                ax.set_xticks(
                    [i / focus_n for i in range(1, 1 + focus_n // 2)],
                    [f"{i}|{focus_n - i}" for i in range(1, 1 + focus_n // 2)],
                )
            elif len(unique_vals) <= 6:
                ax.set_xticks(unique_vals, [f"{v:g}" for v in unique_vals])

    sns.despine(fig=fig)
    fig.tight_layout()
    finalize_figure(fig, pdf=pdf, show=show)


def plot_winning_neighbor_heatmap(
    winning_neighbor_dfs: dict,
    n_leaves_list: list[int],
    prior_display_names: dict[str, str] | None = None,
    pdf=None,
    show: bool = True,
) -> None:
    """Heatmap of mean topological feature differences for winning neighbors vs. true tree."""
    if prior_display_names is None:
        prior_display_names = PRIOR_DISPLAY_NAMES
    all_features = [
        "n_cherries", "colless_index", "top_balance", "depth",
        "n_full_internal", "root_has_cherry", "root_has_singleton", "max_split_size",
    ]
    # Sort features from most positive to most negative pooled mean
    pooled_means = {
        feat: np.mean([winning_neighbor_dfs[p][feat].mean() for p in _FOCUS_PRIORS])
        for feat in all_features
    }
    heatmap_features = sorted(all_features, key=lambda f: -pooled_means[f])

    fig, axes = plt.subplots(
        1, len(_FOCUS_PRIORS) + 1,
        figsize=(1.2 * len(_FOCUS_PRIORS) + 0.5, 2),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.12]},
    )
    heatmap_axes, cbar_ax = axes[:-1], axes[-1]

    for ax, prior_name in zip(heatmap_axes, _FOCUS_PRIORS):
        medians = winning_neighbor_dfs[prior_name].groupby("n")[heatmap_features].mean()
        im = ax.imshow(medians.T, aspect="auto", cmap="PRGn", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(n_leaves_list)))
        ax.set_xticklabels([str(n) for n in n_leaves_list], fontsize=7)
        ax.set_yticks(range(len(heatmap_features)))
        if ax is heatmap_axes[0]:
            ax.set_yticklabels([WINNING_NEIGHBOR_FEATURE_LABELS[name] for name in heatmap_features], fontsize=7)
        else:
            ax.set_yticklabels([])
        display_name = prior_display_names.get(prior_name, prior_name)
        ax.set_title(display_name.replace(" ", "\n"), fontsize=7)
        ax.set_xticks(np.arange(-0.5, len(n_leaves_list), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(heatmap_features), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    heatmap_axes[1].set_xlabel("n", fontsize=7)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("Change\n(winner-true)", fontsize=6, pad=6)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    finalize_figure(fig, pdf=pdf, show=show)


def plot_caterpillar_win_vs_rf(
    caterpillar_dfs: dict,
    n_leaves_list: list[int],
    prior_display_names: dict[str, str] | None = None,
    pdf=None,
    show: bool = True,
) -> None:
    """Line plot of caterpillar loss rate vs. rooted RF distance, by prior and n."""
    if prior_display_names is None:
        prior_display_names = PRIOR_DISPLAY_NAMES
    colors = _n_colors(n_leaves_list)

    fig, axes = plt.subplots(
        1, len(_FOCUS_PRIORS),
        figsize=(1.2 * len(_FOCUS_PRIORS), 2),
        sharey=False,
    )

    for ax, prior_name in zip(axes, _FOCUS_PRIORS):
        df = caterpillar_dfs[prior_name].copy()
        # Treat margin ≈ 0 as a caterpillar win (ties count as optimal)
        df["cat_optimal"] = df["margin"] <= 1e-12
        grouped = df.groupby(["n", "rf_dist"])["cat_optimal"].mean().reset_index()

        for n in n_leaves_list:
            sub = grouped[grouped["n"] == n].sort_values("rf_dist")
            ax.plot(sub["rf_dist"], sub["cat_optimal"],
                    color=colors[n], alpha=0.5, lw=0.9, zorder=n)
            ax.scatter(sub["rf_dist"], sub["cat_optimal"],
                       color=colors[n], s=10, alpha=0.8, lw=0, zorder=n + 0.5)

        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
        display_name = prior_display_names.get(prior_name, prior_name)
        ax.set_title(display_name.replace(" ", "\n"), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(0, int(df["rf_dist"].max()) + 1, 2))
        ax.set_ylim(-0.05, 1.05)
        if prior_name != _FOCUS_PRIORS[0]:
            ax.set_yticklabels([])
        if prior_name == _FOCUS_PRIORS[1]:
            ax.set_xlabel("Rooted RF distance", fontsize=8)

    axes[0].set_ylabel("Avg. loss rate\nto caterpillar", fontsize=8)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[n],
                   markersize=5, label=f"n={n}")
        for n in n_leaves_list
    ]
    axes[0].legend(handles=handles, fontsize=7, frameon=False,
                   title="# leaves", title_fontsize=7, loc="upper right")
    sns.despine(fig=fig)
    fig.tight_layout()
    finalize_figure(fig, pdf=pdf, show=show)
