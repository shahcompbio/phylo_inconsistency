"""Analysis helpers for the counterexample generator.

Given a true tree T* and a randomly chosen NNI neighbor, these routines
compute the per-edge deltas and find minimum-KL distributions that sit at
(or just past) the balance point for each of several priors.
"""

from .costs import compute_all_deltas, kingman_prior, min_kl_balanced_distribution, kl_nudge
from .tree_utils import edge_dict_to_topology

DEFAULT_MIN_EDGE_P = 0.005
DEFAULT_TRUNK_BOOST = 8.0
DEFAULT_EPS_LAMBDA = 0.5


def make_nni_case(tree, neighbor, edge_key, alpha, beta, w_p, w_n):
    """Compute deltas and equal-weight distribution for a true/NNI pair.

    Automatically chooses the orientation (tree_star vs candidate) so that at
    least some per-edge deltas are negative (i.e. the candidate is cheaper for
    those edges), making the case interesting for inconsistency analysis.

    Parameters
    ----------
    tree, neighbor : dict  {label: split_tuple}
    edge_key       : str   the internal edge on which the NNI was performed
    alpha, beta    : float  noise rates
    w_p, w_n       : float  flip weights

    Returns
    -------
    dict with keys:
        tree_star, nbr, topo_star, topo_nbr,
        deltas, p_eq, diff_eq, m, edge_key
    """
    deltas = compute_all_deltas(tree, neighbor, alpha, beta, w_p, w_n)
    tstar, nbr = tree, neighbor
    if all(d >= -1e-12 for d in deltas.values()):
        deltas = compute_all_deltas(neighbor, tree, alpha, beta, w_p, w_n)
        tstar, nbr = neighbor, tree
    topo_star = edge_dict_to_topology(tstar)
    topo_nbr  = edge_dict_to_topology(nbr)
    m         = len(tstar)
    p_eq      = {k: 1 / m for k in tstar}
    diff_eq   = sum(p_eq[k] * deltas[k] for k in tstar)
    return dict(
        tree_star=tstar, nbr=nbr,
        topo_star=topo_star, topo_nbr=topo_nbr,
        deltas=deltas, p_eq=p_eq, diff_eq=diff_eq,
        m=m, edge_key=edge_key,
    )


def run_phase3(
    case,
    min_edge_p=DEFAULT_MIN_EDGE_P,
    trunk_boost=DEFAULT_TRUNK_BOOST,
    eps_lambda=DEFAULT_EPS_LAMBDA,
):
    """For each of three priors find the min-KL balance point then nudge past it.

    Priors tested:
        (a) equal / uniform
        (b) long-trunk  (trunk edge gets trunk_boost × more mass)
        (c) Kingman coalescent

    Parameters
    ----------
    case        : dict   output of make_nni_case
    min_edge_p  : float  minimum probability mass per edge
    trunk_boost : float  multiplier for trunk edge mass in the long-trunk prior
    eps_lambda  : float  how far past the balance lambda to nudge

    Returns
    -------
    list of 9-tuples (or 7-tuples when infeasible):
        (label, prior, p_bal, p_inc, lam_star,
         diff_prior, diff_bal, diff_inc, feasible)
        infeasible entries have None for p_bal/p_inc/lam_star/diff values.
    """
    tree_star_plot = case["tree_star"]
    deltas         = case["deltas"]
    m              = case["m"]
    tree_keys      = list(tree_star_plot.keys())

    p_prior_eq = {k: 1 / m for k in tree_keys}

    p_raw_trunk = {k: trunk_boost / m if k == "trunk" else 1 / m for k in tree_keys}
    _tot        = sum(p_raw_trunk.values())
    p_prior_trunk = {k: v / _tot for k, v in p_raw_trunk.items()}

    p_prior_king = kingman_prior(tree_star_plot)

    priors = [
        ("Balanced",       p_prior_eq),
        ("Heavy trunk",    p_prior_trunk),
        ("Clade weighted", p_prior_king),
    ]

    results = []
    for label, prior in priors:
        p_bal, lam_star, ok = min_kl_balanced_distribution(
            deltas, prior, min_p=min_edge_p
        )
        if not ok:
            results.append((label, prior, None, None, None, None, False))
            continue
        p_inc, _ = kl_nudge(
            deltas, prior, lam_star, eps_lambda=eps_lambda, min_p=min_edge_p
        )
        diff_prior = sum(prior[k] * deltas[k] for k in tree_star_plot)
        diff_bal   = sum(p_bal[k] * deltas[k]  for k in tree_star_plot)
        diff_inc   = sum(p_inc[k] * deltas[k]  for k in tree_star_plot)
        results.append(
            (label, prior, p_bal, p_inc, lam_star, diff_prior, diff_bal, diff_inc, True)
        )
    return results
