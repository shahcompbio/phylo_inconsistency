"""Caterpillar comparison analysis — topology utilities, computation, and caching."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from inconsistency_generation.costs import compute_all_deltas
from inconsistency_generation.tree_generation import random_binary_tree_splits


# ── Topology utilities ─────────────────────────────────────────────────────────

def internal_splits(tree: dict) -> set[frozenset]:
    """Return all non-trivial internal clade sets (2 ≤ |S| ≤ n−1)."""
    n = len(next(iter(tree.values())))
    result = set()
    for vec in tree.values():
        s = sum(vec)
        if 2 <= s <= n - 1:
            result.add(frozenset(i for i, v in enumerate(vec) if v == 1))
    return result


def rf_distance(tree1: dict, tree2: dict) -> int:
    """Rooted Robinson-Foulds distance: |S1 △ S2| for internal split sets."""
    return len(internal_splits(tree1).symmetric_difference(internal_splits(tree2)))


def caterpillar_from_spine(spine: list[int], n: int) -> dict:
    """Build a split-dict caterpillar from an ordered leaf spine."""
    edges: dict[str, tuple] = {
        "trunk": tuple([1] * n),
        **{f"pendant_{i}": tuple(1 if j == i else 0 for j in range(n)) for i in range(n)},
    }
    prefix: set[int] = set()
    for k, leaf in enumerate(spine[:-1]):
        prefix.add(leaf)
        edges[f"int_{k}"] = tuple(1 if i in prefix else 0 for i in range(n))
    return edges


def caterpillar_spine(tree_star: dict) -> list[int]:
    """Order leaves by a depth-prioritized DFS of the rooted clade hierarchy.

    Visits children in decreasing subtree-depth order (ties broken by min leaf
    index). The resulting sequence is the spine of the closest caterpillar.
    """
    n = len(next(iter(tree_star.values())))
    all_leaves = frozenset(range(n))

    clades: set[frozenset] = set()
    for vec in tree_star.values():
        s = sum(vec)
        if 1 <= s <= n:
            clades.add(frozenset(i for i, v in enumerate(vec) if v == 1))

    children_cache: dict = {}
    depth_cache: dict = {}

    def children(clade: frozenset) -> list[frozenset]:
        if clade not in children_cache:
            subsets = [sub for sub in clades if sub < clade]
            children_cache[clade] = [
                sub for sub in subsets
                if not any(sub < mid < clade for mid in subsets)
            ]
        return children_cache[clade]

    def subtree_depth(clade: frozenset) -> int:
        if clade not in depth_cache:
            ch = children(clade)
            depth_cache[clade] = 0 if not ch else 1 + max(subtree_depth(c) for c in ch)
        return depth_cache[clade]

    def depth_order(clade: frozenset) -> list[int]:
        if len(clade) == 1:
            return [next(iter(clade))]
        ordered = sorted(children(clade), key=lambda c: (-subtree_depth(c), min(c)))
        result: list[int] = []
        for child in ordered:
            result.extend(depth_order(child))
        return result

    return depth_order(all_leaves)


def closest_caterpillar(tree_star: dict) -> tuple[dict, int]:
    """Return (caterpillar_dict, rf_dist) for the depth-heuristic closest caterpillar."""
    n = len(next(iter(tree_star.values())))
    spine = caterpillar_spine(tree_star)
    cat = caterpillar_from_spine(spine, n)
    return cat, rf_distance(tree_star, cat)


# ── Computation ────────────────────────────────────────────────────────────────

def compute_caterpillar_results(
    n_leaves_list: list[int],
    n_trees: int,
    priors: dict,
    n_samples: int,
    alpha: float,
    beta: float,
    w_pos: float,
    w_neg: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate random trees, find each one's closest caterpillar, and sample margins.

    For each tree, the closest caterpillar (depth-heuristic) and its per-edge
    delta values are computed once, independently of the edge-length distribution.
    Edge-length weights are then sampled *n_samples* times per tree under each prior.

    Returns
    -------
    dict mapping prior name → DataFrame with columns:
        n, rf_dist, margin, cat_wins
    """
    rng = np.random.default_rng(seed)

    # Pre-compute caterpillar deltas per tree (independent of edge-length draws)
    cat_entries: dict[int, list[dict]] = {}
    for n in n_leaves_list:
        entries = []
        for _ in range(n_trees):
            tree_star = random_binary_tree_splits(n, rng)
            cat, rf = closest_caterpillar(tree_star)
            deltas = compute_all_deltas(tree_star, cat, alpha, beta, w_pos, w_neg)
            entries.append({"tree_star": tree_star, "deltas": deltas, "rf_dist": rf})
        cat_entries[n] = entries

    # Sample edge-length draws and compute margins
    records: dict[str, list[dict]] = {nm: [] for nm in priors}
    for n, entries in cat_entries.items():
        for prior_name, prior_fn in priors.items():
            for entry in entries:
                tree_star = entry["tree_star"]
                deltas = entry["deltas"]
                rf = entry["rf_dist"]
                for _ in range(n_samples):
                    p_dict = prior_fn(tree_star, rng)
                    margin = sum(p_dict[e] * deltas[e] for e in p_dict)
                    records[prior_name].append(
                        {"n": n, "rf_dist": rf, "margin": margin, "cat_wins": margin < 0}
                    )

    return {nm: pd.DataFrame(records[nm]) for nm in priors}


# ── Caching helpers ────────────────────────────────────────────────────────────

def default_caterpillar_cache_path(
    cache_dir: Path | str,
    n_leaves_list: list[int],
    n_trees: int,
    n_samples: int,
    alpha: float,
    beta: float,
    seed: int,
) -> Path:
    n_label = "-".join(str(n) for n in n_leaves_list)
    return Path(cache_dir) / (
        f"caterpillar_results_n{n_label}_trees{n_trees}_"
        f"samples{n_samples}_a{alpha:.3f}_b{beta:.3f}_seed{seed}.pkl"
    )


def load_or_compute_caterpillar_results(
    n_leaves_list: list[int],
    n_trees: int,
    priors: dict,
    n_samples: int,
    alpha: float,
    beta: float,
    w_pos: float,
    w_neg: float,
    seed: int = 42,
    cache_path: Path | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load cached caterpillar results or compute and cache them."""
    if cache_path is not None and not force:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                print(f"Loaded caterpillar results from {cache_path.name}")
                return pickle.load(fh)

    print("Computing caterpillar results ... ", end="", flush=True)
    dfs = compute_caterpillar_results(
        n_leaves_list=n_leaves_list,
        n_trees=n_trees,
        priors=priors,
        n_samples=n_samples,
        alpha=alpha,
        beta=beta,
        w_pos=w_pos,
        w_neg=w_neg,
        seed=seed,
    )
    print("done.")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(dfs, fh)
        print(f"Cached to {cache_path.name}")

    return dfs
