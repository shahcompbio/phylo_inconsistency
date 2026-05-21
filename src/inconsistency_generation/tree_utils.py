def edge_dict_to_topology(edge_dict):
    """Convert a split-dict tree to a topology list for draw_tstar_vertical.

    Topology list entries: (parent_name, child_name, edge_key, leaf_x | None).
    Leaf x-positions are assigned 0, 1, 2, ... in DFS left-to-right order
    (children sorted by minimum leaf index for a consistent layout).
    """
    n = len(next(iter(edge_dict.values())))
    leaf_labels = [chr(ord("A") + i) if n <= 26 else str(i) for i in range(n)]
    all_leaves  = frozenset(range(n))

    def ones(t):
        return frozenset(i for i, v in enumerate(t) if v == 1)

    splits = {k: ones(v) for k, v in edge_dict.items()}

    def node_name(fs):
        if len(fs) == 1:
            return leaf_labels[next(iter(fs))]
        if fs == all_leaves:
            return "root"
        return "n_" + "_".join(str(i) for i in sorted(fs))

    topo = []
    for k, fs in splits.items():
        if fs == all_leaves:
            continue   # trunk drawn as root stub
        proper_supersets = [s for s in splits.values() if fs < s]
        parent_fs   = min(proper_supersets, key=len) if proper_supersets else all_leaves
        parent_name = node_name(parent_fs)
        child_name  = node_name(fs)
        topo.append((parent_name, child_name, k, None))

    # sort children: larger subtree on the left, ties broken by min leaf index
    children = {}
    for parent, child, _, _ in topo:
        children.setdefault(parent, []).append(child)

    def min_leaf(node):
        if node in leaf_labels:
            return leaf_labels.index(node)
        return min(min_leaf(c) for c in children.get(node, []))

    def subtree_size(node):
        if node in leaf_labels:
            return 1
        return sum(subtree_size(c) for c in children.get(node, []))

    for node in children:
        children[node].sort(key=lambda c: (-subtree_size(c), min_leaf(c)))

    leaf_x_map, counter = {}, [0]
    def dfs(node):
        if node in leaf_labels:
            leaf_x_map[node] = float(counter[0])
            counter[0] += 1
        else:
            for child in children.get(node, []):
                dfs(child)
    dfs("root")

    return [(p, c, k, leaf_x_map.get(c)) for p, c, k, _ in topo]

