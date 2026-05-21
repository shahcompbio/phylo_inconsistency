import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from inconsistency_generation.tree_generation import nni_neighbors


def prior_dirichlet_uniform(tree_star, rng):
    keys = list(tree_star.keys())
    weights = rng.dirichlet(np.ones(len(keys)))
    return dict(zip(keys, weights))


def prior_near_equal(concentration: float = 20.0):
    def _prior(tree_star, rng):
        keys = list(tree_star.keys())
        weights = rng.dirichlet(np.full(len(keys), concentration))
        return dict(zip(keys, weights))

    _prior.__name__ = f"NearEqual(c={concentration})"
    return _prior


def prior_heavy_trunk(trunk_weight: float = 5.0):
    def _prior(tree_star, rng):
        keys = list(tree_star.keys())
        alpha = np.array([trunk_weight if key == "trunk" else 1.0 for key in keys])
        weights = rng.dirichlet(alpha)
        return dict(zip(keys, weights))

    _prior.__name__ = f"HeavyTrunk(w={trunk_weight})"
    return _prior


def prior_clade_size_weighted(tree_star, rng):
    keys = list(tree_star.keys())
    sizes = np.array([float(sum(tree_star[key])) for key in keys])
    weights = rng.dirichlet(sizes**2)
    return dict(zip(keys, weights))


def default_priors() -> dict[str, object]:
    return {
        "Near equal": prior_near_equal(concentration=5),
        "Flat": prior_dirichlet_uniform,
        "Heavy trunk": prior_heavy_trunk(trunk_weight=5.0),
        "Clade weighted": prior_clade_size_weighted,
    }


def default_margin_cache_path(
    cache_dir: Path | str,
    n_leaves_list: list[int],
    n_trees: int,
    n_samples: int,
    alpha: float,
    beta: float,
    seed: int,
) -> Path:
    cache_dir = Path(cache_dir)
    n_label = "-".join(str(n_leaves) for n_leaves in n_leaves_list)
    return cache_dir / (
        f"nni_margin_across_n_n{n_label}_trees{n_trees}_"
        f"samples{n_samples}_a{alpha:.3f}_b{beta:.3f}_seed{seed}.pkl"
    )


def default_winning_neighbor_cache_path(
    cache_dir: Path | str,
    n_leaves_list: list[int],
    n_trees: int,
    n_samples: int,
    alpha: float,
    beta: float,
    seed: int,
) -> Path:
    cache_dir = Path(cache_dir)
    n_label = "-".join(str(n_leaves) for n_leaves in n_leaves_list)
    return cache_dir / (
        f"winning_neighbor_topology_n{n_label}_trees{n_trees}_"
        f"samples{n_samples}_a{alpha:.3f}_b{beta:.3f}_seed{seed}.pkl"
    )


WINNING_NEIGHBOR_FEATURE_NAMES = [
    "max_split_size",
    "n_cherries",
    "depth",
    "top_balance",
    "colless_index",
    "root_has_cherry",
    "root_has_singleton",
    "n_full_internal",
]


WINNING_NEIGHBOR_FEATURE_LABELS = {
    "n_cherries": "Delta cherries",
    "colless_index": "Delta Colless",
    "top_balance": "Delta top balance",
    "depth": "Delta depth",
    "n_full_internal": "Delta full internal",
    "root_has_cherry": "Delta root cherry",
    "root_has_singleton": "Delta root singleton",
    "max_split_size": "Delta max split size",
}


def eval_margins(nni_data, p_dict):
    return [sum(p_dict[edge] * record["deltas"][edge] for edge in p_dict) for record in nni_data]


def compute_margin_across_n_results(bundle, priors, n_samples: int, seed: int):
    results_by_n = {}
    win_rates_by_n = {}

    for n_leaves in sorted(bundle["results_by_n"]):
        tree_cache = bundle["results_by_n"][n_leaves]
        print(f"Sampling margins for n={n_leaves} across {len(tree_cache)} trees...")

        rng = np.random.default_rng(seed)
        per_prior_mins = {name: [] for name in priors}
        per_prior_wins = {name: [] for name in priors}

        for entry in tree_cache:
            tree_star = entry["tree_star"]
            nni_data = entry["nni_data"]
            for prior_name, prior_fn in priors.items():
                mins = []
                for _ in range(n_samples):
                    p_dict = prior_fn(tree_star, rng)
                    mins.append(min(eval_margins(nni_data, p_dict)))
                mins_arr = np.array(mins)
                per_prior_mins[prior_name].append(mins_arr)
                per_prior_wins[prior_name].append((mins_arr > 0).mean())

        results_by_n[n_leaves] = {
            name: np.concatenate(per_prior_mins[name]) for name in priors
        }
        win_rates_by_n[n_leaves] = {
            name: np.array(per_prior_wins[name]) for name in priors
        }

    return results_by_n, win_rates_by_n


def load_or_compute_margin_across_n_results(
    cache_path: Path | str,
    bundle,
    priors,
    n_samples: int,
    seed: int,
    force_recompute: bool = False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        print(f"Loaded across-n margin cache from {cache_path}")
        return payload["results_by_n"], payload["win_rates_by_n"]

    results_by_n, win_rates_by_n = compute_margin_across_n_results(
        bundle=bundle,
        priors=priors,
        n_samples=n_samples,
        seed=seed,
    )

    params = bundle.get("params", {})
    payload = {
        "params": {
            "n_leaves_list": list(params.get("n_leaves_list", [])),
            "n_trees": params.get("n_trees"),
            "alpha": params.get("alpha"),
            "beta": params.get("beta"),
            "n_samples": n_samples,
            "seed": seed,
        },
        "results_by_n": results_by_n,
        "win_rates_by_n": win_rates_by_n,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved across-n margin cache to {cache_path}")
    return results_by_n, win_rates_by_n


def tree_star_children(tree_star):
    n_leaves = len(next(iter(tree_star.values())))
    leaf_sets = {}
    for key, vec in tree_star.items():
        if key == "trunk":
            leaf_sets[key] = frozenset(range(n_leaves))
        else:
            leaf_sets[key] = frozenset(i for i, value in enumerate(vec) if value == 1)

    sorted_keys = sorted(tree_star.keys(), key=lambda key: len(leaf_sets[key]), reverse=True)
    children = {key: set() for key in tree_star}
    processed = []

    for key in sorted_keys:
        for parent in reversed(processed):
            if leaf_sets[key] < leaf_sets[parent]:
                children[parent].add(key)
                break
        processed.append(key)

    return children


def _split_of(vec):
    return frozenset(i for i, value in enumerate(vec) if value == 1)


def _leaf_sets(tree_star):
    n_leaves = len(next(iter(tree_star.values())))
    leaf_sets = {}
    for key, vec in tree_star.items():
        if key == "trunk":
            leaf_sets[key] = frozenset(range(n_leaves))
        else:
            leaf_sets[key] = _split_of(vec)
    return leaf_sets


def _direct_leaf_count(key, children, leaf_sets):
    covered = set().union(*(leaf_sets[child] for child in children[key])) if children[key] else set()
    return len(leaf_sets[key] - covered)


def is_near_cherry_edge(key, children, leaf_sets):
    return len(children[key]) == 1 and _direct_leaf_count(key, children, leaf_sets) == 1


def is_full_internal_edge(key, children, leaf_sets):
    return len(children[key]) == 2 and all(len(leaf_sets[child]) > 1 for child in children[key])


def tree_shape_stats(tree_star):
    n_leaves = len(next(iter(tree_star.values())))
    leaf_sets = _leaf_sets(tree_star)
    children = tree_star_children(tree_star)

    proper_keys = [key for key in tree_star if key != "trunk"]
    root_children = list(children["trunk"])

    max_split_size = max(
        (min(len(leaf_sets[key]), n_leaves - len(leaf_sets[key])) for key in proper_keys),
        default=0,
    )
    n_cherries = sum(1 for key in proper_keys if len(leaf_sets[key]) == 2)

    def depth_of(leaf):
        return sum(1 for key in proper_keys if leaf in leaf_sets[key])

    depth = max((depth_of(leaf) for leaf in range(n_leaves)), default=0)

    if len(root_children) == 2:
        top_balance = min(len(leaf_sets[root_children[0]]), len(leaf_sets[root_children[1]])) / n_leaves
    else:
        top_balance = 0.0

    colless = 0
    for key, kids in children.items():
        if len(kids) == 2:
            kid_list = list(kids)
            colless += abs(len(leaf_sets[kid_list[0]]) - len(leaf_sets[kid_list[1]]))

    root_has_cherry = int(any(len(leaf_sets[child]) == 2 for child in root_children))
    root_has_singleton = int(any(len(leaf_sets[child]) == 1 for child in root_children))

    near_cherry_keys = frozenset(
        key for key in proper_keys if is_near_cherry_edge(key, children, leaf_sets)
    )
    full_internal_keys = frozenset(
        key for key in tree_star if is_full_internal_edge(key, children, leaf_sets)
    )

    return {
        "max_split_size": max_split_size,
        "n_cherries": n_cherries,
        "depth": depth,
        "top_balance": top_balance,
        "colless_index": colless,
        "root_has_cherry": root_has_cherry,
        "root_has_singleton": root_has_singleton,
        "n_near_cherries": len(near_cherry_keys),
        "near_cherry_keys": near_cherry_keys,
        "n_full_internal": len(full_internal_keys),
        "full_internal_keys": full_internal_keys,
    }


def prior_mean_weights(tree_star, prior_name: str):
    keys = list(tree_star.keys())
    if prior_name in ("Near equal", "Flat"):
        alpha = np.ones(len(keys))
    elif prior_name == "Heavy trunk":
        alpha = np.array([5.0 if key == "trunk" else 1.0 for key in keys])
    elif prior_name == "Clade weighted":
        sizes = np.array([float(sum(tree_star[key])) for key in keys])
        alpha = sizes**2
    else:
        alpha = np.ones(len(keys))
    return dict(zip(keys, alpha / alpha.sum()))


def compute_topology_feature_analysis(
    bundle,
    priors,
    win_rates_by_n,
    n_samples: int,
    seed: int,
    focus_n: int | None = None,
    focus_prior_names: list[str] | None = None,
):
    if focus_n is None:
        focus_n = max(bundle["results_by_n"])
    if focus_prior_names is None:
        focus_prior_names = ["Flat", "Heavy trunk", "Clade weighted"]

    tree_cache_focus = bundle["results_by_n"][focus_n]
    feat_rows = []
    rng = np.random.default_rng(seed + 1)

    for entry in tree_cache_focus:
        tree_star = entry["tree_star"]
        nni_data = entry["nni_data"]
        n_leaves = len(next(iter(tree_star.values())))

        stats = tree_shape_stats(tree_star)
        near_cherry_keys = stats["near_cherry_keys"]
        full_internal_keys = stats["full_internal_keys"]

        def clade_size(key):
            if key == "trunk":
                return n_leaves
            count = sum(tree_star[key])
            return min(count, n_leaves - count)

        row = {
            "max_split_size": stats["max_split_size"],
            "n_cherries": stats["n_cherries"],
            "depth": stats["depth"],
            "top_balance": stats["top_balance"],
            "colless_index": stats["colless_index"],
            "root_has_cherry": stats["root_has_cherry"],
            "root_has_singleton": stats["root_has_singleton"],
            "n_near_cherries": stats["n_near_cherries"],
            "n_full_internal": stats["n_full_internal"],
        }

        neighbor_stats = []
        for record in nni_data:
            edge_key = record["nni_edge"]
            nni1, nni2, _ = nni_neighbors(tree_star, edge_key)
            neighbor = nni1 if record["nni_index"] == 0 else nni2
            neighbor_stats.append(
                (tree_shape_stats(neighbor), record, edge_key in near_cherry_keys)
            )

        for prior_name in focus_prior_names:
            prior_fn = priors[prior_name]
            prefix = prior_name.lower().replace(" ", "_")
            mean_p = prior_mean_weights(tree_star, prior_name)

            row[f"weighted_nc ({prefix})"] = sum(mean_p.get(key, 0.0) for key in near_cherry_keys)
            row[f"weighted_fi_lin ({prefix})"] = sum(
                mean_p.get(key, 0.0) * clade_size(key) for key in full_internal_keys
            )
            row[f"weighted_fi_quad ({prefix})"] = sum(
                mean_p.get(key, 0.0) * clade_size(key) ** 2 for key in full_internal_keys
            )

            total = 0
            near_cherry_negative_total = 0
            sum_colless = 0.0
            sum_cherries = 0.0
            sum_depth = 0.0
            sum_top_balance = 0.0
            worst_margin = 0.0

            for _ in range(n_samples):
                p_dict = prior_fn(tree_star, rng)
                for neighbor_stat, record, is_near_cherry in neighbor_stats:
                    margin = sum(p_dict[key] * record["deltas"][key] for key in p_dict)
                    if margin < 0:
                        sum_colless += neighbor_stat["colless_index"] - stats["colless_index"]
                        sum_cherries += neighbor_stat["n_cherries"] - stats["n_cherries"]
                        sum_depth += neighbor_stat["depth"] - stats["depth"]
                        sum_top_balance += neighbor_stat["top_balance"] - stats["top_balance"]
                        worst_margin = min(worst_margin, margin)
                        if is_near_cherry:
                            near_cherry_negative_total += 1
                        total += 1

            if total > 0:
                row[f"Δcolless ({prefix})"] = sum_colless / total
                row[f"Δcherries ({prefix})"] = sum_cherries / total
                row[f"Δdepth ({prefix})"] = sum_depth / total
                row[f"Δtop_balance ({prefix})"] = sum_top_balance / total
                row[f"worst_margin ({prefix})"] = worst_margin
                row[f"frac_neg_nc ({prefix})"] = near_cherry_negative_total / total
            else:
                row[f"Δcolless ({prefix})"] = 0.0
                row[f"Δcherries ({prefix})"] = 0.0
                row[f"Δdepth ({prefix})"] = 0.0
                row[f"Δtop_balance ({prefix})"] = 0.0
                row[f"worst_margin ({prefix})"] = 0.0
                row[f"frac_neg_nc ({prefix})"] = 0.0

        feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows)
    drop_columns = [
        col
        for col in feat_df.columns
        if col.startswith("worst_margin")
        or col.startswith("Δ")
        or col.endswith("trunk)")
        or col.endswith("clade_weighted)")
        or col.endswith("flat)")
    ] + ["n_near_cherries"]
    X = feat_df.drop(columns=drop_columns)
    feature_names = list(X.columns)
    focus_targets = {prior_name: win_rates_by_n[focus_n][prior_name] for prior_name in focus_prior_names}

    importances = {}
    r2_scores = {}
    cv_folds = min(5, len(X))
    for target_name, y_vals in focus_targets.items():
        forest = RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        forest.fit(X, y_vals)
        if cv_folds >= 2:
            r2_scores[target_name] = cross_val_score(
                forest,
                X,
                y_vals,
                cv=cv_folds,
                scoring="r2",
            ).mean()
        else:
            r2_scores[target_name] = np.nan
        perm = permutation_importance(
            forest,
            X,
            y_vals,
            n_repeats=20,
            random_state=42,
            n_jobs=-1,
        )
        importances[target_name] = perm.importances_mean

    mean_importance = np.mean([importances[target] for target in focus_targets], axis=0)
    order_importance = np.argsort(mean_importance)

    return {
        "focus_n": focus_n,
        "feat_df": feat_df,
        "X": X,
        "feature_names": feature_names,
        "focus_targets": focus_targets,
        "importances": importances,
        "r2_scores": r2_scores,
        "order_imp": order_importance,
    }


def compute_winning_neighbor_feature_diffs(
    bundle,
    priors,
    n_samples: int,
    seed: int,
    feature_names: list[str] | None = None,
):
    if feature_names is None:
        feature_names = WINNING_NEIGHBOR_FEATURE_NAMES

    records_by_prior = {name: [] for name in priors}
    zero_diff = {feature_name: 0 for feature_name in feature_names}

    for n_leaves in sorted(bundle["results_by_n"]):
        rng = np.random.default_rng(seed)
        for entry in bundle["results_by_n"][n_leaves]:
            tree_star = entry["tree_star"]
            true_stats = tree_shape_stats(tree_star)
            neighbors = []
            for record in entry["nni_data"]:
                edge_key = record["nni_edge"]
                nni1, nni2, _ = nni_neighbors(tree_star, edge_key)
                neighbor = nni1 if record["nni_index"] == 0 else nni2
                neighbors.append((record, tree_shape_stats(neighbor)))

            for prior_name, prior_fn in priors.items():
                for _ in range(n_samples):
                    p_dict = prior_fn(tree_star, rng)
                    min_margin = float("inf")
                    winning_stats = None
                    for record, neighbor_stats in neighbors:
                        margin = sum(p_dict[edge] * record["deltas"][edge] for edge in p_dict)
                        if margin < min_margin:
                            min_margin = margin
                            winning_stats = neighbor_stats

                    if min_margin < 0 and winning_stats is not None:
                        diff = {
                            feature_name: winning_stats[feature_name] - true_stats[feature_name]
                            for feature_name in feature_names
                        }
                    else:
                        diff = dict(zero_diff)

                    diff["n"] = n_leaves
                    diff["min_margin"] = min_margin
                    diff["inconsistent"] = min_margin < 0
                    records_by_prior[prior_name].append(diff)

    return {
        prior_name: pd.DataFrame(records)
        for prior_name, records in records_by_prior.items()
    }


def load_or_compute_winning_neighbor_feature_diffs(
    cache_path: Path | str,
    bundle,
    priors,
    n_samples: int,
    seed: int,
    feature_names: list[str] | None = None,
    force_recompute: bool = False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        print(f"Loaded winning-neighbor cache from {cache_path}")
        return {
            prior_name: pd.DataFrame.from_records(records)
            for prior_name, records in payload["records_by_prior"].items()
        }

    dfs = compute_winning_neighbor_feature_diffs(
        bundle=bundle,
        priors=priors,
        n_samples=n_samples,
        seed=seed,
        feature_names=feature_names,
    )

    params = bundle.get("params", {})
    payload = {
        "params": {
            "n_leaves_list": list(params.get("n_leaves_list", [])),
            "n_trees": params.get("n_trees"),
            "alpha": params.get("alpha"),
            "beta": params.get("beta"),
            "n_samples": n_samples,
            "seed": seed,
            "feature_names": list(feature_names or WINNING_NEIGHBOR_FEATURE_NAMES),
        },
        "records_by_prior": {
            prior_name: df.to_dict(orient="records")
            for prior_name, df in dfs.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved winning-neighbor cache to {cache_path}")
    return dfs