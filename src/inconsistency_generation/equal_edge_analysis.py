"""Tree builders and heatmap computation for the equal-edge-weight analysis (Figure A1)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from inconsistency_generation.costs import min_cost_tree, noise_prob


# ── Tree builders ──────────────────────────────────────────────────────────────

def caterpillar_splits(n: int) -> dict[str, tuple]:
    """T1: canonical left-spine caterpillar."""
    splits = {}
    for i in range(n):
        s = [0] * n
        s[i] = 1
        splits[f"pendant_{i}"] = tuple(s)
    for k in range(2, n):
        splits[f"internal_{k}"] = tuple([1] * k + [0] * (n - k))
    splits["trunk"] = tuple([1] * n)
    return splits


def nni_cherry_caterpillar_splits(n: int) -> dict[str, tuple]:
    """T2: one NNI move from T1 — the two rightmost leaves form a cherry."""
    if n < 4:
        raise ValueError("NNI cherry-caterpillar requires n >= 4")
    splits = {}
    for i in range(n):
        s = [0] * n
        s[i] = 1
        splits[f"pendant_{i}"] = tuple(s)
    for k in range(2, n - 1):
        splits[f"internal_{k}"] = tuple([1] * k + [0] * (n - k))
    splits["terminal_cherry"] = tuple([0] * (n - 2) + [1, 1])
    splits["trunk"] = tuple([1] * n)
    return splits


def three_leaf_subtree_splits(n: int) -> dict[str, tuple]:
    """T3: one NNI move from T2 — the three rightmost leaves form a subtree."""
    if n < 5:
        raise ValueError("Three-leaf subtree tree requires n >= 5")
    splits = {}
    for i in range(n):
        s = [0] * n
        s[i] = 1
        splits[f"pendant_{i}"] = tuple(s)
    for k in range(2, n - 2):
        splits[f"internal_{k}"] = tuple([1] * k + [0] * (n - k))
    splits["terminal_cherry"] = tuple([0] * (n - 2) + [1, 1])
    splits["triple"] = tuple([0] * (n - 3) + [1, 1, 1])
    splits["trunk"] = tuple([1] * n)
    return splits


TREE_BUILDERS: dict[str, callable] = {
    "caterpillar": caterpillar_splits,
    "cherry_cat": nni_cherry_caterpillar_splits,
    "three_leaf": three_leaf_subtree_splits,
}


# ── Cost functions ─────────────────────────────────────────────────────────────

def expected_cost_given_edge(
    candidate_splits: list[tuple],
    generating_edge: tuple,
    alpha: float,
    beta: float,
    w_pos: float,
    w_neg: float,
) -> float:
    """E[C(candidate; x_hat) | phi* = generating_edge]."""
    n = len(generating_edge)
    total = 0.0
    for obs in itertools.product((0, 1), repeat=n):
        p = noise_prob(obs, generating_edge, alpha, beta)
        total += p * min_cost_tree(candidate_splits, obs, w_pos, w_neg)
    return total


def expected_tree_cost(
    candidate_splits: list[tuple],
    true_tree_edges: list[tuple],
    edge_probabilities: list[float],
    alpha: float,
    beta: float,
    w_pos: float,
    w_neg: float,
) -> float:
    """E[C(candidate; x_hat)] under equal edge-weight prior."""
    assert abs(sum(edge_probabilities) - 1.0) < 1e-9
    return sum(
        p * expected_cost_given_edge(candidate_splits, edge, alpha, beta, w_pos, w_neg)
        for edge, p in zip(true_tree_edges, edge_probabilities)
    )


# ── Heatmap computation ────────────────────────────────────────────────────────

def compute_heatmaps(
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    n_values: list[int],
    candidate_name: str,
    true_tree_name: str,
) -> dict[int, np.ndarray]:
    """Compute Δ(candidate, true_tree) over the (alpha, beta) grid for each n."""
    candidate_builder = TREE_BUILDERS[candidate_name]
    true_tree_builder = TREE_BUILDERS[true_tree_name]
    diffs: dict[int, np.ndarray] = {}

    for n in n_values:
        candidate_tree = candidate_builder(n)
        true_tree = true_tree_builder(n)
        true_tree_edges = list(true_tree.values())
        candidate_splits = list(candidate_tree.values())
        edge_probabilities = [1.0 / len(true_tree_edges)] * len(true_tree_edges)

        diff = np.zeros((len(alpha_vals), len(beta_vals)))
        for ai, alpha in enumerate(alpha_vals):
            for bi, beta in enumerate(beta_vals):
                w_pos = np.log((1 - beta) / alpha)
                w_neg = np.log((1 - alpha) / beta)
                if w_pos <= 0 or w_neg <= 0:
                    diff[ai, bi] = np.nan
                    continue
                true_cost = expected_tree_cost(
                    true_tree_edges, true_tree_edges, edge_probabilities,
                    alpha, beta, w_pos, w_neg,
                )
                cand_cost = expected_tree_cost(
                    candidate_splits, true_tree_edges, edge_probabilities,
                    alpha, beta, w_pos, w_neg,
                )
                diff[ai, bi] = cand_cost - true_cost

        diffs[n] = diff
        print(
            f"  {candidate_name} − {true_tree_name}, n={n}: "
            f"inconsistent fraction = {np.nanmean(diff < 0):.3f}"
        )

    return diffs


def load_or_compute_heatmaps(
    analysis: dict,
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    n_values: list[int],
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, np.ndarray]]:
    """Load heatmaps from cache or compute and cache them."""
    cache_path = Path(analysis["cache_path"])
    if cache_path.exists() and not force_recompute:
        cache = np.load(cache_path)
        cached_n = [int(v) for v in cache["n_values"]]
        diffs = {n: cache[f"diff_{n}"] for n in cached_n}
        print(f"Loaded {analysis['name']} from {cache_path.name}")
        return cache["alpha_vals"], cache["beta_vals"], cached_n, diffs

    print(f"Computing {analysis['name']} ...")
    diffs = compute_heatmaps(
        alpha_vals, beta_vals, n_values,
        candidate_name=analysis["candidate"],
        true_tree_name=analysis["true_tree"],
    )
    np.savez_compressed(
        cache_path,
        alpha_vals=alpha_vals,
        beta_vals=beta_vals,
        n_values=np.array(n_values, dtype=int),
        **{f"diff_{n}": diffs[n] for n in n_values},
    )
    print(f"  Saved to {cache_path.name}")
    return alpha_vals, beta_vals, n_values, diffs


# ── Analysis spec factories ────────────────────────────────────────────────────

def default_analyses(cache_dir: Path | str) -> list[dict]:
    """Return the two T1/T2 analysis specs with cache paths resolved."""
    cache_dir = Path(cache_dir)
    return [
        {
            "name": "caterpillar_minus_cherry_cat",
            "candidate": "caterpillar",
            "true_tree": "cherry_cat",
            "colorbar_label": r"$\Delta(T_1, T_2)$",
            "annotation_alpha": 0.45,
            "annotation_beta": 0.32,
            "cache_path": cache_dir / "equal_edges_matched_weights_caterpillar_minus_cherry_cat_data.npz",
        },
        {
            "name": "cherry_cat_minus_caterpillar",
            "candidate": "cherry_cat",
            "true_tree": "caterpillar",
            "colorbar_label": r"$\Delta(T_2, T_1)$",
            "annotation_alpha": 0.16,
            "annotation_beta": 0.40,
            "cache_path": cache_dir / "equal_edges_matched_weights_cherry_cat_minus_caterpillar_data.npz",
        },
    ]


def default_t2_t3_analyses(cache_dir: Path | str) -> list[dict]:
    """Return the two T2/T3 analysis specs with cache paths resolved."""
    cache_dir = Path(cache_dir)
    return [
        {
            "name": "cherry_cat_minus_three_leaf",
            "candidate": "cherry_cat",
            "true_tree": "three_leaf",
            "colorbar_label": r"$\Delta(T_2, T_3)$",
            "annotation_alpha": 0.35,
            "annotation_beta": 0.25,
            "cache_path": cache_dir / "equal_edges_matched_weights_cherry_cat_minus_three_leaf_data.npz",
        },
        {
            "name": "three_leaf_minus_cherry_cat",
            "candidate": "three_leaf",
            "true_tree": "cherry_cat",
            "colorbar_label": r"$\Delta(T_3, T_2)$",
            "annotation_alpha": 0.35,
            "annotation_beta": 0.25,
            "cache_path": cache_dir / "equal_edges_matched_weights_three_leaf_minus_cherry_cat_data.npz",
        },
    ]
