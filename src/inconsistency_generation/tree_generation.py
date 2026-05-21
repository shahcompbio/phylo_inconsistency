import numpy as np


def random_binary_tree_splits(n, rng=None):
    """Generate a uniformly random bifurcating rooted tree on n labeled leaves.

    Uses the random-join (coalescent) algorithm: start with n singleton leaf
    sets; repeatedly pick two uniformly at random, merge them into a new
    internal node, until two sets remain (their union is the trunk).

    Parameters
    ----------
    n   : int  number of leaves (>= 2)
    rng : numpy.random.Generator or None

    Returns
    -------
    edges : dict  {label: split_tuple}
        'trunk'      — all-ones tuple
        'pendant_i'  — singleton tuple for leaf i
        'int_k'      — internal edges, k = 0, 1, ... (n-3 of them for n≥3)
    """
    if rng is None:
        rng = np.random.default_rng()

    edges = {}
    edges["trunk"] = tuple([1] * n)
    for i in range(n):
        e = [0] * n
        e[i] = 1
        edges[f"pendant_{i}"] = tuple(e)

    # pool: list of frozensets of leaf indices
    pool = [frozenset([i]) for i in range(n)]
    int_idx = 0

    # each iteration merges 2 → 1; loop runs n-2 times (stops when 2 remain)
    while len(pool) > 2:
        i, j = rng.choice(len(pool), size=2, replace=False)
        merged = pool[i] | pool[j]
        split = tuple(1 if k in merged else 0 for k in range(n))
        edges[f"int_{int_idx}"] = split
        int_idx += 1
        pool = [pool[k] for k in range(len(pool)) if k not in (i, j)]
        pool.append(merged)

    # The two remaining sets are the two direct children of the root.
    # Their union = all leaves = trunk, which is already in edges.
    # No further edge to add.
    return edges


def get_internal_edge_keys(edge_dict):
    """Return the keys of all internal (non-trunk, non-pendant) edges.

    An internal edge has 2 ≤ |split| ≤ n-2.

    Parameters
    ----------
    edge_dict : dict  {label: split_tuple}

    Returns
    -------
    list of str
    """
    n = len(next(iter(edge_dict.values())))
    return [k for k, v in edge_dict.items()
            if 2 <= sum(v) <= n - 2]


def nni_neighbors(edge_dict, edge_key):
    """Return both NNI neighbors obtained by rearranging `edge_key`.

    For internal edge e with subtree S = A ∪ B (A, B = the two child subtrees
    of the node at the bottom of e) and sibling subtree C:
      NNI-1: new split = B ∪ C  (swap A ↔ C)
      NNI-2: new split = A ∪ C  (swap B ↔ C)
    Only the split of `edge_key` changes; all other splits are identical.

    Parameters
    ----------
    edge_dict : dict  {label: split_tuple}
    edge_key  : str   must be an internal edge (2 ≤ |split| ≤ n-2)

    Returns
    -------
    nni1, nni2 : dict, dict  — two new edge dicts
    info       : dict  with keys A, B, C, S (frozensets of leaf indices),
                       nni1_split, nni2_split (tuples)
    """
    n = len(next(iter(edge_dict.values())))
    all_leaves = frozenset(range(n))

    def ones(t):
        return frozenset(i for i, v in enumerate(t) if v == 1)

    def to_tuple(fs):
        return tuple(1 if i in fs else 0 for i in range(n))

    S = ones(edge_dict[edge_key])
    assert 2 <= len(S) <= n - 2, (
        f"'{edge_key}' is not an internal edge (|split|={len(S)}, need 2..{n-2})"
    )

    other_splits = {k: ones(v) for k, v in edge_dict.items() if k != edge_key}

    # child subtrees A and B: maximal proper non-empty subsets of S
    within_S = [fs for fs in other_splits.values() if fs and fs < S]
    A_B = [fs for fs in within_S if not any(fs < other for other in within_S)]
    assert len(A_B) == 2, (
        f"Expected 2 child subtrees within S={set(S)}, got {len(A_B)}: "
        f"{[set(x) for x in A_B]}"
    )
    A, B = sorted(A_B, key=min)  # deterministic order: A has the smaller min leaf

    # sibling subtree C: parent_subtree minus S
    proper_supersets = [fs for fs in other_splits.values() if S < fs < all_leaves]
    parent = min(proper_supersets, key=len) if proper_supersets else all_leaves
    C = parent - S
    assert C, "Sibling subtree C is empty — is this really an internal edge?"

    nni1_split = to_tuple(B | C)   # swap A ↔ C
    nni2_split = to_tuple(A | C)   # swap B ↔ C

    def make_neighbor(new_split):
        return {k: (new_split if k == edge_key else v)
                for k, v in edge_dict.items()}

    info = {"A": A, "B": B, "C": C, "S": S,
            "nni1_split": nni1_split, "nni2_split": nni2_split}
    return make_neighbor(nni1_split), make_neighbor(nni2_split), info


def random_nni(edge_dict, rng=None):
    """Pick a random internal edge and return one random NNI neighbor.

    Parameters
    ----------
    edge_dict : dict  {label: split_tuple}
    rng       : numpy.random.Generator or None

    Returns
    -------
    neighbor  : dict  the NNI neighbor edge dict
    edge_key  : str   the internal edge that was rearranged
    nni_index : int   0 for NNI-1 (swap A↔C), 1 for NNI-2 (swap B↔C)
    info      : dict  as returned by nni_neighbors()
    """
    if rng is None:
        rng = np.random.default_rng()

    internal_keys = get_internal_edge_keys(edge_dict)
    assert internal_keys, "Tree has no internal edges (n too small?)"

    edge_key = internal_keys[rng.integers(len(internal_keys))]
    nni1, nni2, info = nni_neighbors(edge_dict, edge_key)
    nni_index = int(rng.integers(2))
    neighbor = nni1 if nni_index == 0 else nni2
    return neighbor, edge_key, nni_index, info

