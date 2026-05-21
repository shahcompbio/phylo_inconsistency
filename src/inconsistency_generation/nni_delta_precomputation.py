import pickle
import time
from pathlib import Path

import numpy as np

from inconsistency_generation.costs import compute_all_deltas
from inconsistency_generation.tree_generation import (
    get_internal_edge_keys,
    nni_neighbors,
    random_binary_tree_splits,
)


def default_flip_weights(alpha: float, beta: float) -> tuple[float, float]:
    return np.log((1 - alpha) / alpha), np.log((1 - beta) / beta)


def precompute_nni_deltas_for_tree(
    tree_star: dict[str, tuple[int, ...]],
    alpha: float,
    beta: float,
    w_pos: float,
    w_neg: float,
) -> list[dict[str, object]]:
    """Compute NNI delta vectors for every neighbor of a single tree."""
    records: list[dict[str, object]] = []
    n_leaves = len(next(iter(tree_star.values())))

    for edge_key in get_internal_edge_keys(tree_star):
        nni1, nni2, info = nni_neighbors(tree_star, edge_key)
        split_size = min(len(info["S"]), n_leaves - len(info["S"]))
        abc_sizes = (len(info["A"]), len(info["B"]), len(info["C"]))

        for nni_index, neighbor in enumerate((nni1, nni2)):
            deltas = compute_all_deltas(tree_star, neighbor, alpha, beta, w_pos, w_neg)
            records.append(
                {
                    "nni_edge": edge_key,
                    "nni_index": nni_index,
                    "split_size": split_size,
                    "abc_sizes": abc_sizes,
                    "deltas": deltas,
                }
            )

    return records


def compute_nni_delta_bundle(
    n_leaves_list: list[int],
    n_trees: int,
    alpha: float,
    beta: float,
    seed: int = 42,
    w_pos: float | None = None,
    w_neg: float | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Generate and precompute NNI delta data for all requested leaf counts."""
    if w_pos is None or w_neg is None:
        inferred_w_pos, inferred_w_neg = default_flip_weights(alpha, beta)
        if w_pos is None:
            w_pos = inferred_w_pos
        if w_neg is None:
            w_neg = inferred_w_neg

    results_by_n: dict[int, list[dict[str, object]]] = {}

    for n_leaves in n_leaves_list:
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()
        tree_records: list[dict[str, object]] = []

        if verbose:
            print(f"[n={n_leaves}] Generating {n_trees} trees and precomputing NNI deltas...")

        for tree_index in range(n_trees):
            tree_star = random_binary_tree_splits(n_leaves, rng=rng)
            nni_data = precompute_nni_deltas_for_tree(tree_star, alpha, beta, w_pos, w_neg)
            tree_records.append(
                {
                    "tree_star": tree_star,
                    "nni_data": nni_data,
                }
            )

            if verbose and (tree_index + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  {tree_index + 1}/{n_trees} trees ({elapsed:.1f}s elapsed)")

        results_by_n[n_leaves] = tree_records

        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"  Finished n={n_leaves} in {elapsed:.1f}s")

    return {
        "params": {
            "n_leaves_list": list(n_leaves_list),
            "n_trees": n_trees,
            "alpha": alpha,
            "beta": beta,
            "w_pos": w_pos,
            "w_neg": w_neg,
            "seed": seed,
        },
        "results_by_n": results_by_n,
    }


def default_nni_delta_cache_path(
    cache_dir: Path | str,
    n_leaves_list: list[int],
    n_trees: int,
    alpha: float,
    beta: float,
    seed: int,
    w_pos: float | None = None,
    w_neg: float | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    n_label = "-".join(str(value) for value in n_leaves_list)
    file_name = (
        f"nni_deltas_n{n_label}_trees{n_trees}_"
        f"a{alpha:.3f}_b{beta:.3f}_seed{seed}.pkl"
    )

    if w_pos is not None or w_neg is not None:
        default_w_pos, default_w_neg = default_flip_weights(alpha, beta)
        if w_pos is None:
            w_pos = default_w_pos
        if w_neg is None:
            w_neg = default_w_neg

        if not (np.isclose(w_pos, default_w_pos) and np.isclose(w_neg, default_w_neg)):
            file_name = file_name.removesuffix(".pkl") + f"_wp{w_pos:.3f}_wn{w_neg:.3f}.pkl"

    return cache_dir / file_name


def save_nni_delta_bundle(bundle: dict[str, object], cache_path: Path | str) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path


def load_nni_delta_bundle(cache_path: Path | str) -> dict[str, object]:
    cache_path = Path(cache_path)
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def load_or_compute_nni_delta_bundle(
    cache_path: Path | str,
    n_leaves_list: list[int],
    n_trees: int,
    alpha: float,
    beta: float,
    seed: int = 42,
    w_pos: float | None = None,
    w_neg: float | None = None,
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        if verbose:
            print(f"Loaded NNI delta bundle from {cache_path}")
        return load_nni_delta_bundle(cache_path)

    bundle = compute_nni_delta_bundle(
        n_leaves_list=n_leaves_list,
        n_trees=n_trees,
        alpha=alpha,
        beta=beta,
        seed=seed,
        w_pos=w_pos,
        w_neg=w_neg,
        verbose=verbose,
    )
    save_nni_delta_bundle(bundle, cache_path)

    if verbose:
        print(f"Saved NNI delta bundle to {cache_path}")

    return bundle