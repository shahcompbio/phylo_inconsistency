from itertools import product
import numpy as np


def noise_prob(x_hat, x_e, alpha, beta):
    """Pr(x_hat | x_e) under independent per-leaf noise.

    alpha : false negative rate  Pr(obs=0 | true=1)
    beta  : false positive rate  Pr(obs=1 | true=0)
    """
    prob = 1.0
    for xi, xi_hat in zip(x_e, x_hat):
        if xi == 1:
            prob *= (1 - alpha) if xi_hat == 1 else alpha
        else:
            prob *= beta if xi_hat == 1 else (1 - beta)
    return prob


def flip_cost(x_e, x_hat, w_P, w_N):
    """Cost of editing x_hat into x_e.

    0→1 flip (hat=0, edge=1): costs w_P
    1→0 flip (hat=1, edge=0): costs w_N
    """
    cost = 0.0
    for xi, xi_hat in zip(x_e, x_hat):
        if xi == 1 and xi_hat == 0:
            cost += w_P
        elif xi == 0 and xi_hat == 1:
            cost += w_N
    return cost


def min_cost_tree(tree_splits, x_hat, w_P, w_N):
    """Minimum flip cost over all edges of the tree for a given observed pattern.

    Parameters
    ----------
    tree_splits : iterable of split tuples  (values of the edge dict)
    x_hat       : tuple of observed 0/1 characters
    w_P, w_N    : flip weights
    """
    return min(flip_cost(x_e, x_hat, w_P, w_N) for x_e in tree_splits)


def delta_e(gen_edge, tree_T_splits, tree_Tstar_splits, alpha, beta, w_P, w_N):
    """E[C(T; x_hat) - C(T*; x_hat) | phi* = gen_edge].

    Positive => T* is cheaper (correct preference for that edge).
    Negative => T  is cheaper (MCP would prefer T for mutations on this edge).

    Parameters
    ----------
    gen_edge          : tuple  the split where the true mutation occurred
    tree_T_splits     : list   split tuples for candidate tree T
    tree_Tstar_splits : list   split tuples for true tree T*
    alpha, beta       : float  noise rates
    w_P, w_N          : float  flip weights
    """
    n = len(gen_edge)
    total = 0.0
    for x_hat in product((0, 1), repeat=n):
        p = noise_prob(x_hat, gen_edge, alpha, beta)
        cost_T     = min_cost_tree(tree_T_splits,     x_hat, w_P, w_N)
        cost_Tstar = min_cost_tree(tree_Tstar_splits, x_hat, w_P, w_N)
        total += p * (cost_T - cost_Tstar)
    return total


def compute_all_deltas(tree_star, tree_T, alpha, beta, w_P, w_N):
    """Compute Delta_e(T*, T) for every edge e in T*.

    Parameters
    ----------
    tree_star, tree_T : dict  {label: split_tuple}
    alpha, beta, w_P, w_N : float

    Returns
    -------
    dict  {label: Delta_e value}
    """
    Tstar_splits = list(tree_star.values())
    T_splits     = list(tree_T.values())
    return {
        label: delta_e(gen_edge, T_splits, Tstar_splits, alpha, beta, w_P, w_N)
        for label, gen_edge in tree_star.items()
    }


def min_l2_balanced_distribution(deltas, min_p=0.0):
    """Minimum L2 distance from uniform subject to sum(p_e * Delta_e) = 0.

    Closed-form solution (projection of uniform onto the balancing hyperplane):
        p_e* = 1/m - (Delta_bar / ||Delta||^2) * Delta_e
    where Delta_bar = mean(Delta_e) and ||Delta||^2 = sum(Delta_e^2).

    If any p_e* < 0 the unconstrained solution exits the simplex. In that case
    the negative-weight edges are clamped to 0 and the remainder is re-solved
    on the reduced edge set (iteratively until all weights are non-negative).

    Parameters
    ----------
    deltas : dict  {label: Delta_e}
    min_p  : float  minimum probability mass per edge (default 0.0)

    Returns
    -------
    p      : dict  {label: probability}  — sums to 1, all >= min_p, sum(p_e Delta_e) = 0
    feasible : bool  — False if no distribution in the simplex achieves balance
               (i.e. all Delta_e have the same sign, or min_p floor is infeasible)
    """
    from scipy.optimize import minimize

    keys   = list(deltas.keys())
    d      = np.array([deltas[k] for k in keys], dtype=float)
    m_full = len(keys)

    # Quick feasibility check: need both positive and negative Delta_e
    if np.all(d >= 0) or np.all(d <= 0):
        return {k: 1.0 / m_full for k in keys}, False

    if min_p <= 0:
        # Original analytical iterative-projection path
        active = np.ones(m_full, dtype=bool)
        for _ in range(m_full):
            d_active  = d[active]
            m_act     = active.sum()
            delta_bar = d_active.mean()
            delta_sq  = (d_active ** 2).sum()
            if delta_sq < 1e-15:
                break
            p_active = 1.0 / m_act - (delta_bar / delta_sq) * d_active
            if np.all(p_active >= -1e-12):
                p_active = np.maximum(p_active, 0.0)
                p_active /= p_active.sum()
                p = np.zeros(m_full)
                p[active] = p_active
                return {k: float(p[i]) for i, k in enumerate(keys)}, True
            idx_in_active = np.argmin(p_active)
            global_idx    = np.where(active)[0][idx_in_active]
            active[global_idx] = False
        p = np.zeros(m_full)
        p[active] = 1.0 / active.sum()
        return {k: float(p[i]) for i, k in enumerate(keys)}, True
    else:
        # With min_p floor: use constrained QP via scipy SLSQP
        if m_full * min_p > 1.0:
            return {k: 1.0 / m_full for k in keys}, False
        p0 = np.ones(m_full) / m_full
        result = minimize(
            fun=lambda p: float(np.sum((p - 1.0 / m_full) ** 2)),
            jac=lambda p: 2.0 * (p - 1.0 / m_full),
            x0=p0,
            method="SLSQP",
            bounds=[(min_p, 1.0)] * m_full,
            constraints=[
                {"type": "eq", "fun": lambda p: float(p.sum() - 1.0),
                 "jac": lambda p: np.ones(m_full)},
                {"type": "eq", "fun": lambda p: float(p @ d),
                 "jac": lambda p: d},
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success:
            p_vals = np.maximum(result.x, min_p)
            p_vals /= p_vals.sum()
            return {k: float(p_vals[i]) for i, k in enumerate(keys)}, True
        # Retry without floor
        return min_l2_balanced_distribution(deltas, min_p=0.0)


def min_range_balanced_distribution(deltas, trunk_key="trunk", trunk_discount=5.0, min_p=0.0):
    """Minimum range subject to sum(p_e * Delta_e) = 0, with trunk discounted.

    Minimises  t_max - t_min  where each edge contributes its *effective* value
    to the range:

        effective_e = p_e / trunk_discount   if e == trunk_key
        effective_e = p_e                    otherwise

    This means the trunk is allowed to be up to `trunk_discount` times larger
    than the other edges without inflating the range penalty — the LP will
    naturally set a long trunk while keeping the non-trunk edges as equal as
    possible.

    Still a valid LP: the only change from the uniform-weight version is that
    the trunk's constraint coefficients are divided by trunk_discount.

    Parameters
    ----------
    deltas         : dict  {label: Delta_e}
    trunk_key      : str   key of the trunk edge (default "trunk")
    trunk_discount : float  how much less the trunk counts in the range (default 5)
    min_p          : float  minimum probability mass per edge (default 0.0)

    Returns
    -------
    p        : dict  {label: probability}
    feasible : bool  — False if all Delta_e have the same sign or min_p floor is infeasible
    """
    from scipy.optimize import linprog

    keys = list(deltas.keys())
    m    = len(keys)
    d    = np.array([deltas[k] for k in keys], dtype=float)

    if np.all(d >= 0) or np.all(d <= 0):
        return {k: 1.0 / m for k in keys}, False

    if m * min_p > 1.0:
        return {k: 1.0 / m for k in keys}, False

    # Discount factors: 1/trunk_discount for trunk, 1.0 for all other edges
    discount = np.array(
        [1.0 / trunk_discount if k == trunk_key else 1.0 for k in keys]
    )

    # Variables: x = [p_0, ..., p_{m-1}, t_max, t_min]
    n_vars = m + 2
    idx_tmax, idx_tmin = m, m + 1

    # Objective: minimise t_max - t_min
    c = np.zeros(n_vars)
    c[idx_tmax] =  1.0
    c[idx_tmin] = -1.0

    # Equality constraints
    A_eq = np.zeros((2, n_vars))
    A_eq[0, :m] = d      # sum(p_e * Delta_e) = 0
    A_eq[1, :m] = 1.0    # sum(p_e) = 1
    b_eq = np.array([0.0, 1.0])

    # Inequality constraints (A_ub x <= b_ub)
    # discount_e * p_e - t_max <= 0   (effective value <= t_max)
    # t_min - discount_e * p_e <= 0   (t_min <= effective value)
    A_ub = np.zeros((2 * m, n_vars))
    for i in range(m):
        A_ub[i,     i]        =  discount[i]   #  discount * p_e
        A_ub[i,     idx_tmax] = -1.0            # -t_max
        A_ub[m + i, i]        = -discount[i]   # -discount * p_e
        A_ub[m + i, idx_tmin] =  1.0            # +t_min
    b_ub = np.zeros(2 * m)

    # Bounds: p_e >= min_p; t_max, t_min in [0, 1]
    bounds = [(min_p, None)] * m + [(0, 1), (0, 1)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        if min_p > 0:
            # Retry without the floor
            return min_range_balanced_distribution(
                deltas, trunk_key=trunk_key, trunk_discount=trunk_discount, min_p=0.0
            )
        return min_l2_balanced_distribution(deltas)

    p_vals = np.maximum(result.x[:m], min_p)
    p_vals /= p_vals.sum()
    return {k: float(p_vals[i]) for i, k in enumerate(keys)}, True


def kingman_prior(tree):
    """Kingman-coalescent edge weights.

    Under the Kingman coalescent the expected waiting time when k lineages are
    present is 2 / (k(k-1)).  An edge with descendant clade of size s_e is
    active during a coalescent interval involving roughly s_e lineages, giving
    expected length ∝ 1 / (s_e(s_e - 1)) ≈ 1/s_e^2 for large s_e.

    To use as a *prior* (large s_e = longer edge = more mass), we invert and
    use weights ∝ s_e^2.  This gives pendants weight 1, deep internal edges
    weight s_e^2, and the trunk weight n^2 — a stronger bias toward long deep
    branches than the linear-clade-size approximation.

    Parameters
    ----------
    tree : dict  {label: split_tuple}

    Returns
    -------
    dict  {label: probability}  — sums to 1, all > 0
    """
    sizes   = {k: float(sum(v)) for k, v in tree.items()}
    weights = {k: s ** 2 for k, s in sizes.items()}
    total   = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def min_kl_balanced_distribution(deltas, prior, min_p=0.0, tol=1e-10, max_iter=200):
    """Minimum KL(p || prior) subject to sum(p_e * Delta_e) = 0.

    The solution is an exponential tilt of the prior:

        p*_e  ∝  prior_e * exp(lambda * Delta_e)

    where lambda is chosen so that the weighted-average Delta is zero.
    This treats relative deviations equally (doubling 0.25→0.5 costs the same
    as doubling 0.5→1.0), unlike L2 which penalises absolute differences.

    lambda is found by bisection on

        f(lambda) = sum_e  p*_e(lambda) * Delta_e  =  0

    f is strictly decreasing in lambda (negative tilt concentrates mass on
    negative-Delta edges), so bisection is well-defined whenever the problem
    is feasible (i.e. deltas have both signs).

    Returns the *balance-point* distribution (f = 0).  To obtain a strictly
    inconsistent distribution, nudge lambda by -eps past lambda*.

    Parameters
    ----------
    deltas  : dict  {label: Delta_e}
    prior   : dict  {label: probability}  — must be strictly positive
    min_p   : float  minimum probability per edge after normalisation (default 0)
    tol     : float  bisection convergence tolerance on f(lambda)
    max_iter: int    maximum bisection iterations

    Returns
    -------
    p        : dict  {label: probability}
    lambda_  : float  the tilt parameter at the balance point
    feasible : bool   False if all Delta_e have the same sign
    """
    keys = list(deltas.keys())
    d    = np.array([deltas[k] for k in keys], dtype=float)
    q    = np.array([prior[k]  for k in keys], dtype=float)

    if np.all(d >= 0) or np.all(d <= 0):
        return {k: prior[k] for k in keys}, 0.0, False

    def tilted(lam):
        log_w = lam * d + np.log(q)
        log_w -= log_w.max()          # numerical stability
        w = np.exp(log_w)
        w /= w.sum()
        return w

    def f(lam):
        return float(tilted(lam) @ d)

    # f(0) = sum(prior_e * Delta_e) — may be positive or negative
    # bracket: push lambda until f changes sign
    lo, hi = 0.0, 0.0
    f0 = f(0.0)
    step = 1.0
    if f0 > 0:          # need lambda < 0 to tilt toward negative-Delta edges
        lo = -step
        while f(lo) > 0 and abs(lo) < 1e6:
            lo *= 2
        hi = 0.0
    else:               # f0 <= 0, lambda* >= 0 (or already balanced)
        hi = step
        while f(hi) < 0 and hi < 1e6:
            hi *= 2
        lo = 0.0

    # Bisect
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        fm  = f(mid)
        if abs(fm) < tol:
            break
        if fm > 0:
            hi = mid
        else:
            lo = mid

    lam_star = (lo + hi) / 2.0
    p_vals   = tilted(lam_star)

    if min_p > 0:
        p_vals = np.maximum(p_vals, min_p)
        p_vals /= p_vals.sum()

    return {k: float(p_vals[i]) for i, k in enumerate(keys)}, lam_star, True


def kl_nudge(deltas, prior, lambda_star, eps_lambda=0.5, min_p=0.0):
    """Return the exponential tilt at lambda_star - eps_lambda (inconsistent side).

    Parameters
    ----------
    deltas       : dict  {label: Delta_e}
    prior        : dict  {label: probability}
    lambda_star  : float  balance-point tilt from min_kl_balanced_distribution
    eps_lambda   : float  how far past balance to go (default 0.5)
    min_p        : float  minimum probability per edge

    Returns
    -------
    p   : dict  {label: probability}
    lam : float  the tilt used
    """
    keys = list(deltas.keys())
    d    = np.array([deltas[k] for k in keys], dtype=float)
    q    = np.array([prior[k]  for k in keys], dtype=float)

    def _eval(lam):
        log_w = lam * d + np.log(q)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        if min_p > 0:
            w = np.maximum(w, min_p)
            w /= w.sum()
        return w

    # Keep doubling the nudge until the clamped distribution is genuinely
    # inconsistent (Σ p Δ < 0).  The natural tilt is always inconsistent but
    # clamping to min_p can reverse the sign.
    eps = eps_lambda
    for _ in range(20):
        lam = lambda_star - eps
        w   = _eval(lam)
        if float(w @ d) < 0:
            break
        eps *= 2.0

    return {k: float(w[i]) for i, k in enumerate(keys)}, lam
