import numpy as np


def draw_gray_topology(ax, topology, n_edges, trunk_key="trunk", highlight_edge=None,
                       dashed_edges=None, lw_h = 4.0, lw_v = 4.0, lw_hl = 6.0):
    """Draw a rooted binary tree in uniform gray with equal branch lengths.

    highlight_edge: optional edge key to draw in orange at increased linewidth.
    """
    p_eq = 1 / n_edges

    children_g, leaf_x_g = {}, {}
    for parent, child, key, x in topology:
        children_g.setdefault(parent, []).append(child)
        if x is not None:
            leaf_x_g[child] = float(x)

    root = ({p for p,_,_,_ in topology} - {c for _,c,_,_ in topology}).pop()
    node_y_g, node_x_g = {root: 1.0}, {}

    def assign(node, y):
        node_y_g[node] = y
        if node in leaf_x_g:
            node_x_g[node] = leaf_x_g[node]
            return leaf_x_g[node]
        xs = [assign(kid, y - p_eq) for kid in children_g.get(node, [])]
        node_x_g[node] = sum(xs) / len(xs)
        return node_x_g[node]
    assign(root, 1.0)

    gray = "#888888"
    hl_color = "#e07800"   # orange for the highlighted edge
    dashed_edges = set() if dashed_edges is None else set(dashed_edges)
    for parent, child, key, _ in topology:
        px, py = node_x_g[parent], node_y_g[parent]
        cx, cy = node_x_g[child],  node_y_g[child]
        is_hl = (key == highlight_edge)
        if abs(px - cx) > 1e-9:
            ax.plot([px, cx], [py, py], color=gray, lw=lw_h, solid_capstyle="butt")
        v_color = hl_color if is_hl else gray
        v_lw    = lw_hl if is_hl else lw_v
        ax.plot(
            [cx, cx], [py, cy],
            color=v_color,
            lw=v_lw,
            linestyle="densely dotted" if key in dashed_edges else "solid",
            solid_capstyle="butt",
        )

    rx, ry = node_x_g[root], node_y_g[root]
    ax.plot([rx, rx], [ry, ry + p_eq], color=gray, lw=lw_h, solid_capstyle="butt")

    y_vals = list(node_y_g.values())
    y_range = max(y_vals) - min(y_vals) or 1.0
    for node, x in leaf_x_g.items():
        y = node_y_g[node]
        ax.scatter(x, y, s=25, color="black", zorder=5)
        ax.text(x, y - 0.03 * y_range, node, ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.08 * y_range)
    ax.axis("off")


def _delta_colormap():
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "delta_cmap",
        [
            (0.0,  "#da0000"),   # strong red   (most negative)
            (0.35, "#d54747"),   # muted red
            (0.45, "#af6d6d"),   # edge of red
            (0.5,  "#666666"),   # gray center
            (0.55, "#6D7998"),   # edge of blue
            (0.65, "#5a77c6"),   # muted blue
            (1.0,  "#4D8EF0"),   # strong blue  (most positive)
        ],
    )
    return cmap


def _delta_colormap_gray():
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "delta_cmap",
        [
            (0,    "#666666"),
            (0.5,  "#666666"),
            (1.0,  "#666666"),
        ],
    )
    return cmap

_DELTA_CMAP = _delta_colormap()

def draw_tstar_vertical(ax, topology, p_dict, d_by_label,
                        trunk_p_key="trunk", max_lw=16.0, min_lw=2.0, h_lw=2.0,
                        labels=True, leaf_labels=True, gamma=0.4, fontscale=1.0,
                        leafsize=10, color=True, widths=False, dashed_edges=None):
    """Vertical rectilinear phylogeny colored and sized by Δ_e and p_e.

    - Branch length ∝ p_e
    - Line width    ∝ |Δ_e|  (when widths=True)
    - Color: diverging colormap (red → gray → blue) with signed power transform
      (when color=True); uniform gray otherwise.
      gamma < 1 compresses the center so near-zero values map quickly to color.
    """
    if not color:
        active_cmap = _delta_colormap_gray()
    else:
        active_cmap = _delta_colormap()
    dashed_edges = set() if dashed_edges is None else set(dashed_edges)

    max_p = max(p_dict.values())
    d_max = max(abs(v) for v in d_by_label.values()) + 1e-12

    def blen(key): return p_dict[key] / max_p
    if widths:
        def blw(key): return min(min_lw + (max_lw - min_lw) * abs(d_by_label[key]) / (d_max / 2), max_lw)
    else:
        def blw(key): return min_lw
    def bcol(key):
        d = d_by_label[key]
        x = d / d_max
        x_compressed = float(np.sign(x) * (abs(x) ** gamma))
        return active_cmap((x_compressed + 1.0) / 2.0)

    children, edge_of, leaf_x = {}, {}, {}
    for parent, child, key, x in topology:
        children.setdefault(parent, []).append(child)
        edge_of[(parent, child)] = key
        if x is not None:
            leaf_x[child] = float(x)

    root = ({p for p,_,_,_ in topology} - {c for _,c,_,_ in topology}).pop()
    node_y = {root: 1.0}
    node_x = {}

    def assign(node, y):
        node_y[node] = y
        if node in leaf_x:
            node_x[node] = leaf_x[node]
            return leaf_x[node]
        xs = [assign(kid, y - blen(edge_of[(node, kid)]))
              for kid in children.get(node, [])]
        node_x[node] = sum(xs) / len(xs)
        return node_x[node]
    assign(root, 1.0)

    neg_plot, pos_plot = [False], [False]

    def draw_branch(px, py, cx, cy, key):
        if abs(px - cx) > 1e-9:
            ax.plot([px, cx], [py, py], color="#222222", lw=h_lw,
                    solid_capstyle="round", zorder=10)
        ax.plot(
            [cx, cx], [py, cy],
            color=bcol(key),
            lw=blw(key),
            linestyle= (0, (1, 1)) if key in dashed_edges else "solid",
            solid_capstyle="butt",
        )
        if not labels:
            return
        my   = (py + cy) / 2
        dmag = abs(d_by_label[key])
        dval = d_by_label[key]
        if dmag > 0.05:
            ax.text(cx - 0.08, my, f"Δ={d_by_label[key]:+.2f}", ha="right", va="center",
                    fontsize=7 * fontscale, color=bcol(key))
            ax.text(cx + 0.08, my, f"p={p_dict[key]:.2f}", ha="left", va="center",
                    fontsize=7 * fontscale, color=bcol(key))
        elif dval < 0 and not neg_plot[0]:
            ax.text(cx + 0.08, my, f"p={p_dict[key]:.2f}", ha="left", va="center",
                    fontsize=7 * fontscale, color=bcol(key))
            neg_plot[0] = True
        elif dval > 0 and not pos_plot[0]:
            ax.text(cx + 0.08, my, f"p={p_dict[key]:.2f}", ha="left", va="center",
                    fontsize=7 * fontscale, color=bcol(key))
            pos_plot[0] = True

    # trunk stub above root
    tl = blen(trunk_p_key)
    rx, ry = node_x[root], node_y[root]
    ax.plot([rx, rx], [ry, ry + tl], color=bcol(trunk_p_key), lw=blw(trunk_p_key),
            solid_capstyle="butt")
    if labels:
        ax.text(rx + 0.08, ry + tl / 2, f"p={p_dict[trunk_p_key]:.2f}",
                ha="left",  va="center", fontsize=7 * fontscale, color=bcol(trunk_p_key))
        ax.text(rx - 0.08, ry + tl / 2, f"Δ={d_by_label[trunk_p_key]:+.2f}",
                ha="right", va="center", fontsize=7 * fontscale, color=bcol(trunk_p_key))

    for parent, child, key, _ in topology:
        px, py = node_x[parent], node_y[parent]
        cx, cy = node_x[child],  node_y[child]
        draw_branch(px, py, cx, cy, key)

    y_vals = list(node_y.values())
    y_range = max(y_vals) - min(y_vals) or 1.0
    for node, x in leaf_x.items():
        y = node_y[node]
        ax.scatter(x, y, s=leafsize, color="#222222", zorder=15, edgecolors="none")
        if leaf_labels:
            ax.text(x, y - 0.05 * y_range, node, ha="center", va="top",
                    fontsize=11 * fontscale)

    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.08 * y_range)
    ax.axis("off")
