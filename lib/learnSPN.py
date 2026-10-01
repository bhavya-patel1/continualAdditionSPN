
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2_contingency

from lib.customspn import LeafNode, ProductNode, SumNode


# 1. Leaf distribution
def create_leaf(var_id: int, data: np.ndarray, alpha: float = 1.0) -> LeafNode:
    """
    Univariate categorical distribution with Dirichlet (Laplace) smoothing.

    Parameters
    ----------
    var_id : int
        Column index of the variable.
    data : np.ndarray
        Full data matrix (instances × variables). Only column ``var_id`` is used.
    alpha : float
        Dirichlet pseudo-count (1.0 = Laplace smoothing).

    Returns
    -------
    LeafNode with MAP-estimated categorical probabilities.
    """
    values = data[:, var_id].astype(int)
    n_categories = int(values.max()) + 1  # domain {0, ..., max_val}
    counts = np.bincount(values, minlength=n_categories).astype(float)
    probs = (counts + alpha) / (counts.sum() + alpha * n_categories)
    return LeafNode(scope=var_id, params=probs)


# 2. Variable splitting - BH-corrected pairwise G-test
def _pairwise_g_test(x: np.ndarray, y: np.ndarray):
    """G-test of independence between two discrete variables.

    Returns (g_statistic, p_value).  If either variable is constant the pair
    is declared independent (p = 1).
    """
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return 0.0, 1.0

    # Build contingency table via cross-tabulation
    x_int, y_int = x.astype(int), y.astype(int)
    x_max, y_max = x_int.max() + 1, y_int.max() + 1
    table = np.zeros((x_max, y_max), dtype=float)
    np.add.at(table, (x_int, y_int), 1)

    # Drop rows/cols that are entirely zero (sparse categories)
    row_mask = table.sum(axis=1) > 0
    col_mask = table.sum(axis=0) > 0
    table = table[np.ix_(row_mask, col_mask)]

    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0

    try:
        stat, p_val, _, _ = chi2_contingency(table, lambda_="log-likelihood")
        return stat, p_val
    except ValueError:
        return 0.0, 1.0


def split_variables(data: np.ndarray, scope: list,
                    p_threshold: float = 0.05):
    """
    Split variables into approximately independent subsets.

    Performs pairwise G-tests on all variable pairs within *scope*, applies
    the Benjamini-Hochberg procedure to control the false discovery rate at
    level ``p_threshold``, builds a dependency graph from the significant
    pairs, and returns the connected components.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (instances × all variables).
    scope : list[int]
        Column indices of the variables under consideration.
    p_threshold : float
        FDR level for the BH procedure.

    Returns
    -------
    list[list[int]] | None
        A list of variable-index lists (one per independent component), or
        ``None`` if all variables fall into a single component (i.e. the
        variable split fails).
    """
    n_vars = len(scope)
    if n_vars <= 1:
        return None

    # Collect all pairwise p-values
    pair_indices = []
    p_values = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            _, p_val = _pairwise_g_test(data[:, scope[i]], data[:, scope[j]])
            pair_indices.append((i, j))
            p_values.append(p_val)

    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    #Benjamini-Hochberg correction 
    sorted_idx = np.argsort(p_values)
    rejected = np.zeros(n_tests, dtype=bool)
    for rank, idx in enumerate(sorted_idx, start=1):
        # BH threshold for this rank
        bh_threshold = (rank / n_tests) * p_threshold
        if p_values[idx] <= bh_threshold:
            rejected[idx] = True
        else:
            # Once we fail, all later (larger) p-values also fail
            break
    # All indices up to (and including) the last rejection are rejected
    if rejected.any():
        last_rejected_rank = np.max(np.where(rejected[sorted_idx])[0]) + 1
        for rank_i in range(last_rejected_rank):
            rejected[sorted_idx[rank_i]] = True

    # Build adjacency matrix from rejected (dependent) pairs
    adj = np.zeros((n_vars, n_vars), dtype=int)
    for k, (i, j) in enumerate(pair_indices):
        if rejected[k]:
            adj[i, j] = 1
            adj[j, i] = 1

    n_components, labels = connected_components(adj, directed=False,
                                                return_labels=True)

    if n_components <= 1:
        return None  # variable split failed

    components = []
    for c in range(n_components):
        comp_scope = [scope[i] for i in range(n_vars) if labels[i] == c]
        components.append(comp_scope)
    return components


# 3. Instance splitting — Naive Bayes hard EM
def _nb_log_prob(data: np.ndarray, cluster_params: list,
                 log_weights: np.ndarray) -> np.ndarray:
    """
    Compute log P(x | cluster) + log P(cluster) for every instance × cluster.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Data restricted to the current scope.
    cluster_params : list[list[np.ndarray]]
        ``cluster_params[c][j]`` = categorical probability vector for
        variable *j* in cluster *c*.
    log_weights : np.ndarray, shape (k,)
        Log mixing proportions.

    Returns
    -------
    np.ndarray, shape (n, k)
        Log-probability matrix.
    """
    n, d = data.shape
    k = len(cluster_params)
    log_probs = np.tile(log_weights, (n, 1))  # (n, k)

    for c in range(k):
        for j in range(d):
            vals = data[:, j].astype(int)
            params_cj = cluster_params[c][j]
            # Clip indices to valid range (safety)
            safe_vals = np.clip(vals, 0, len(params_cj) - 1)
            log_probs[:, c] += np.log(params_cj[safe_vals] + 1e-300)

    return log_probs


def _em_fit(data: np.ndarray, k: int, alpha: float = 1.0,
            max_iter: int = 50, random_state: int = 42):
    """
    Fit a Naive Bayes mixture with *k* components using hard EM.

    Returns
    -------
    labels : np.ndarray, shape (n,)
        Hard cluster assignments.
    weights : np.ndarray, shape (k,)
        Mixing proportions.
    ll : float
        Final total log-likelihood.
    """
    rng = np.random.RandomState(random_state)
    n, d = data.shape

    # Domain sizes per variable (within the scope slice)
    domain_sizes = [int(data[:, j].max()) + 1 for j in range(d)]

    # Initialise: random hard assignments 
    labels = rng.randint(0, k, size=n)

    for iteration in range(max_iter):
        # M-step: estimate parameters from current assignments
        cluster_params = []
        weights = np.zeros(k)

        for c in range(k):
            mask = labels == c
            n_c = mask.sum()
            weights[c] = n_c
            params_c = []
            for j in range(d):
                counts = np.bincount(data[mask, j].astype(int),
                                     minlength=domain_sizes[j]).astype(float)
                # Dirichlet smoothing
                probs = (counts + alpha) / (n_c + alpha * domain_sizes[j])
                params_c.append(probs)
            cluster_params.append(params_c)

        # Smoothed mixing weights (add alpha to prevent zero-weight clusters)
        weights = (weights + alpha) / (n + alpha * k)
        log_weights = np.log(weights + 1e-300)

        # E-step: hard assignment
        log_probs = _nb_log_prob(data, cluster_params, log_weights)
        new_labels = np.argmax(log_probs, axis=1)

        # Check convergence
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    # Final log-likelihood
    log_probs = _nb_log_prob(data, cluster_params, log_weights)
    # For hard EM: LL = sum of log P(x_i, c_i) = sum of max over c
    ll = log_probs[np.arange(n), labels].sum()

    return labels, weights, ll


def split_instances(data: np.ndarray, scope: list,
                    max_k: int = 5, lam: float = 1.0,
                    alpha: float = 1.0, max_iter: int = 50,
                    random_state: int = 42):
    """
    Cluster instances using a penalised Naive Bayes mixture (hard EM).

    Parameters
    ----------
    data : np.ndarray
        Full data matrix (instances × all variables).
    scope : list[int]
        Column indices of variables in the current sub-problem.
    max_k, lam, alpha, max_iter, random_state : hyperparameters

    Returns
    -------
    labels : np.ndarray, shape (n,)
        Cluster assignments.
    weights : np.ndarray, shape (best_k,)
        Mixing proportions |T_i| / |T|.
    best_k : int
        Selected number of clusters.
    """
    data_scope = data[:, scope]
    n = len(data_scope)
    d = len(scope)

    best_labels = None
    best_weights = None
    best_score = -np.inf
    best_k = 2

    for k in range(2, max_k + 1):
        if n < k * 2:
            break  # not enough instances

        labels, weights, ll = _em_fit(data_scope, k, alpha=alpha,
                                      max_iter=max_iter,
                                      random_state=random_state + k)
        penalty = lam * k * d  # log of exponential prior
        score = ll - penalty

        if score > best_score:
            best_score = score
            best_labels = labels.copy()
            best_weights = weights.copy()
            best_k = k

    # Fallback if nothing ran, do k=2 unconditionally
    if best_labels is None:
        best_labels, best_weights, _ = _em_fit(data_scope, 2, alpha=alpha,
                                               max_iter=max_iter,
                                               random_state=random_state)
        best_k = 2

    # Re-compute empirical weights from final labels
    empirical_weights = np.zeros(best_k)
    for c in range(best_k):
        empirical_weights[c] = (best_labels == c).sum()
    empirical_weights /= empirical_weights.sum()

    return best_labels, empirical_weights, best_k


# 4. Main recursive algorithm
def learn_spn(data: np.ndarray, scope: list,
              p_threshold: float = 0.01,
              min_instances: int = 30,
              max_k: int = 5,
              lam: float = 1.0,
              alpha: float = 1.0,
              max_depth: int = 50,
              max_iter: int = 50,
              random_state: int = 42,
              _depth: int = 0) -> 'Node':
    """
    Recursively learns an SPN structure following Algorithm 1 of Gens &
    Domingos (2013).

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_instances, n_total_variables).
    scope : list[int]
        Column indices of the variables in this sub-problem.
    p_threshold : float
        FDR level for the Benjamini-Hochberg independence test.
    min_instances : int
        Minimum number of instances before forcing a naive factorisation.
    max_k : int
        Maximum number of clusters to try during instance splitting.
    lam : float
        Penalty coefficient for the exponential prior on cluster count.
    alpha : float
        Dirichlet pseudo-count for smoothing categoricals.
    max_depth : int
        Maximum recursion depth before forcing factorisation.
    max_iter : int
        Maximum EM iterations per clustering attempt.
    random_state : int
        Random seed for reproducibility.
    _depth : int
        Internal recursion depth counter (do not set manually).

    Returns
    -------
    Node
        Root of the learned SPN sub-tree.
    """
    n = len(data)

    # Base case - single variable
    if len(scope) == 1:
        return create_leaf(scope[0], data, alpha=alpha)

    # Too few instances or too deep - naive factorisation 
    if n < min_instances or _depth >= max_depth:
        leaves = [create_leaf(v, data, alpha=alpha) for v in scope]
        return ProductNode(children=leaves)

    # All rows identical - factorize 
    data_scope = data[:, scope]
    if np.all(data_scope == data_scope[0]):
        leaves = [create_leaf(v, data, alpha=alpha) for v in scope]
        return ProductNode(children=leaves)

    # Step 1: Try variable split 
    components = split_variables(data, scope, p_threshold=p_threshold)

    if components is not None and len(components) > 1:
        children = [
            learn_spn(data, comp,
                      p_threshold=p_threshold,
                      min_instances=min_instances,
                      max_k=max_k, lam=lam, alpha=alpha,
                      max_depth=max_depth, max_iter=max_iter,
                      random_state=random_state,
                      _depth=_depth + 1)
            for comp in components
        ]
        return ProductNode(children=children)

    # Step 2: Instance split via Naive Bayes EM
    labels, weights, best_k = split_instances(
        data, scope, max_k=max_k, lam=lam, alpha=alpha,
        max_iter=max_iter, random_state=random_state
    )

    # Filter out clusters that are too small
    valid_clusters = []
    for c in range(best_k):
        mask = labels == c
        if mask.sum() >= min_instances:
            valid_clusters.append(c)

    # If fewer than 2 valid clusters, fall back to factorisation
    if len(valid_clusters) < 2:
        leaves = [create_leaf(v, data, alpha=alpha) for v in scope]
        return ProductNode(children=leaves)

    children = []
    child_weights = []
    for c in valid_clusters:
        mask = labels == c
        data_c = data[mask]
        children.append(
            learn_spn(data_c, scope,
                      p_threshold=p_threshold,
                      min_instances=min_instances,
                      max_k=max_k, lam=lam, alpha=alpha,
                      max_depth=max_depth, max_iter=max_iter,
                      random_state=random_state,
                      _depth=_depth + 1)
        )
        child_weights.append(len(data_c))

    # Weights = fraction of instances in each subset
    child_weights = np.array(child_weights, dtype=float)
    child_weights /= child_weights.sum()

    return SumNode(children=children, weights=child_weights)
