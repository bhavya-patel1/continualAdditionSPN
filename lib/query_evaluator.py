
import numpy as np
import torch
from pgmpy.inference import VariableElimination
from lib.customspn import SPN, Node, LeafNode, SumNode, ProductNode
from simple_einet.einet import Einet


# Helpers

def _l1_l2(p_model: np.ndarray, p_ref: np.ndarray):
    """L1 and L2 distance between two probability vectors."""
    diff = np.abs(p_model - p_ref)
    return float(np.mean(diff)), float(np.sqrt(np.mean(diff ** 2)))


def _get_cardinality(var_id: int, data: np.ndarray) -> int:
    return int(data[:, var_id].max()) + 1


# Custom SPN query helpers
def _spn_marginal_probs(root: Node, var_id: int, cardinality: int,
                        all_var_ids: list, num_total_vars: int) -> np.ndarray:
    """
    Compute P(X_var_id = v) for v in 0..cardinality-1 from a custom SPN tree
    by marginalising all other variables.
    Uses one dummy data row per cardinality value with only var_id set,
    and the SPN evaluates the joint; we sum/average over remaining vars.

    Simpler approach: for each value v, create N test rows where var_id = v,
    evaluate log P(x_1, ..., x_{var_id}=v, ..., x_D), and use log-sum-exp
    over the joint to approximate the marginal.
    """
    # Build a grid of all observed values for non-target vars
    # We approximate P(X_var=v) ≈ mean_x P(x, X_var=v) over test data is not
    # straightforward here. Instead we enumerate properly by setting var_id=v
    # and marginalising rest using the model's own marginal.
    # Since our SPN is a joint model P(x_1,...,x_D), we compute:
    # P(X_var=v) = sum_{x_{-var}} P(x_1,...,X_var=v,...,x_D)
    # For a discrete model with small cardinality this is tractable.
    # We approximate by summing over all combinations — too expensive for D>5.
    # Practical approximation: average over test data with var_id fixed to v.
    raise NotImplementedError("Use data-based approximation in eval_marginal_l1l2 instead.")


def _spn_conditional_probs(spn: SPN, var_id: int, evidence_var: int,
                            evidence_val: int, cardinality: int,
                            test_data: np.ndarray) -> np.ndarray:
    """
    Approximate P(X_var_id | X_evidence_var = evidence_val) from the SPN.
    Method: fix evidence, evaluate joint for each value of var_id,
    normalise.
    Uses rows from test_data where evidence_var = evidence_val.
    """
    mask = test_data[:, evidence_var] == evidence_val
    rows = test_data[mask]
    if len(rows) == 0:
        return None
    probs = np.zeros(cardinality)
    for v in range(cardinality):
        rows_v = rows.copy()
        rows_v[:, var_id] = v
        log_p = spn.eval(rows_v)
        probs[v] = np.exp(np.mean(log_p))
    s = probs.sum()
    return probs / s if s > 0 else np.ones(cardinality) / cardinality



# Einet query helpers
def _einet_conditional_probs(einet: Einet, scope: list,
                              target_local: int, evidence_local: int,
                              evidence_val: int, cardinality: int,
                              test_data: np.ndarray,
                              device: str = "cpu") -> np.ndarray:
    """
    P(X_target | X_evidence = evidence_val) from Einet.
    Same approach: fix evidence, vary target, evaluate joint, normalise.
    """
    evidence_global = scope[evidence_local]
    mask = test_data[:, evidence_global] == evidence_val
    rows = test_data[mask]
    if len(rows) == 0:
        return None
    local_rows = rows[:, scope].astype(np.int64)
    probs = np.zeros(cardinality)
    einet = einet.to(device)
    einet.eval()
    with torch.no_grad():
        for v in range(cardinality):
            local_rows_v = local_rows.copy()
            local_rows_v[:, target_local] = v
            t = torch.tensor(local_rows_v, dtype=torch.long, device=device)
            lp = einet(t).squeeze(-1).cpu().numpy()
            probs[v] = np.exp(np.mean(lp))
    s = probs.sum()
    return probs / s if s > 0 else np.ones(cardinality) / cardinality



# BN ground-truth query helper
def _bn_marginal(ve: VariableElimination, var_id: int,
                  var_names: list = None) -> np.ndarray:
    """P(X_var_id) from BN via variable elimination."""
    node = var_names[var_id] if var_names else var_id
    result = ve.query([node], show_progress=False)
    return result.values


def _bn_conditional(ve: VariableElimination, target_var: int,
                    evidence_var: int, evidence_val: int,
                    var_names: list = None) -> np.ndarray:
    """P(X_target | X_evidence = evidence_val) from BN."""
    target_node   = var_names[target_var]   if var_names else target_var
    evidence_node = var_names[evidence_var] if var_names else evidence_var
    try:
        result = ve.query([target_node],
                          evidence={evidence_node: evidence_val},
                          show_progress=False)
        return result.values
    except Exception:
        return None


def _find_most_dependent_var(target_var: int, scope: list,
                              data: np.ndarray) -> int:
    """
    Find the variable in scope most correlated with target_var
    (by Cramer's V approximation — chi-sq / N).
    """
    from scipy.stats import chi2_contingency
    best_var, best_v = -1, -1.0
    target_col = data[:, target_var]
    for v in scope:
        if v == target_var:
            continue
        table = np.zeros((int(target_col.max()) + 1,
                          int(data[:, v].max()) + 1), dtype=int)
        for t_val, v_val in zip(target_col, data[:, v]):
            table[t_val, v_val] += 1
        try:
            chi2, _, _, _ = chi2_contingency(table)
            cramers = chi2 / (len(target_col) * (min(table.shape) - 1))
            if cramers > best_v:
                best_v, best_var = cramers, v
        except Exception:
            continue
    return best_var if best_var >= 0 else scope[0]


def eval_marginal_l1l2(spn: SPN, bn_model, target_var: int,
                        scope: list, test_data: np.ndarray,
                        var_names: list = None, device: str = "cpu"):
    """
    (A) L1/L2 of P(X_target) — SPN vs BN.
    var_names: list of column names (e.g. list(df.columns)) so pgmpy gets
               the correct string node identifiers.
    """
    cardinality = _get_cardinality(target_var, test_data)
    ve = VariableElimination(bn_model)

    probs_spn = np.zeros(cardinality)
    for v in range(cardinality):
        rows_v = test_data.copy()
        rows_v[:, target_var] = v
        lp = spn.eval(rows_v)
        probs_spn[v] = np.exp(np.mean(lp))
    s = probs_spn.sum()
    probs_spn = probs_spn / s if s > 0 else np.ones(cardinality) / cardinality

    probs_bn = _bn_marginal(ve, target_var, var_names)[:cardinality]
    return _l1_l2(probs_spn, probs_bn)


def eval_all_marginals_l1l2(spn: SPN, bn_model, scope: list,
                              test_data: np.ndarray,
                              var_names: list = None, device: str = "cpu"):
    """(B) Average L1/L2 of P(X_i) for every variable i in scope — SPN vs BN."""
    ve = VariableElimination(bn_model)
    l1s, l2s = [], []
    for var_id in scope:
        cardinality = _get_cardinality(var_id, test_data)
        probs_spn = np.zeros(cardinality)
        for v in range(cardinality):
            rows_v = test_data.copy()
            rows_v[:, var_id] = v
            lp = spn.eval(rows_v)
            probs_spn[v] = np.exp(np.mean(lp))
        s = probs_spn.sum()
        probs_spn = probs_spn / s if s > 0 else np.ones(cardinality) / cardinality
        try:
            probs_bn = _bn_marginal(ve, var_id, var_names)[:cardinality]
            l1, l2 = _l1_l2(probs_spn, probs_bn)
            l1s.append(l1); l2s.append(l2)
        except Exception:
            pass
    return float(np.mean(l1s)) if l1s else np.nan, float(np.mean(l2s)) if l2s else np.nan


def eval_conditional_l1l2(spn: SPN, bn_model, target_var: int,
                           scope: list, test_data: np.ndarray,
                           evidence_var: int = None,
                           var_names: list = None, device: str = "cpu"):
    """
    (C) L1/L2 of P(X_target | X_evidence = v) — SPN vs BN, averaged over
    all observed evidence values.
    """
    if evidence_var is None:
        evidence_var = _find_most_dependent_var(target_var, scope, test_data)

    cardinality_target = _get_cardinality(target_var, test_data)
    cardinality_ev = _get_cardinality(evidence_var, test_data)
    ve = VariableElimination(bn_model)

    l1s, l2s = [], []
    for ev_val in range(cardinality_ev):
        p_spn = _spn_conditional_probs(spn, target_var, evidence_var,
                                        ev_val, cardinality_target, test_data)
        p_bn  = _bn_conditional(ve, target_var, evidence_var, ev_val, var_names)
        if p_spn is None or p_bn is None:
            continue
        p_bn_trimmed = p_bn[:cardinality_target]
        l1, l2 = _l1_l2(p_spn, p_bn_trimmed)
        l1s.append(l1); l2s.append(l2)

    return (float(np.mean(l1s)) if l1s else np.nan,
            float(np.mean(l2s)) if l2s else np.nan,
            evidence_var)


# Public API — Einet

def eval_einet_marginal_l1l2(einet: Einet, bn_model, target_var: int,
                               scope: list, test_data: np.ndarray,
                               var_names: list = None, device: str = "cpu"):
    """(A) Marginal L1/L2 for Einet."""
    cardinality = _get_cardinality(target_var, test_data)
    ve = VariableElimination(bn_model)
    target_local = scope.index(target_var)

    probs_spn = np.zeros(cardinality)
    local_data = test_data[:, scope].astype(np.int64)
    einet = einet.to(device)
    einet.eval()
    with torch.no_grad():
        for v in range(cardinality):
            ld_v = local_data.copy()
            ld_v[:, target_local] = v
            t = torch.tensor(ld_v, dtype=torch.long, device=device)
            lp = einet(t).squeeze(-1).cpu().numpy()
            probs_spn[v] = np.exp(np.mean(lp))
    s = probs_spn.sum()
    probs_spn = probs_spn / s if s > 0 else np.ones(cardinality) / cardinality
    probs_bn = _bn_marginal(ve, target_var, var_names)[:cardinality]
    return _l1_l2(probs_spn, probs_bn)


def eval_einet_all_marginals_l1l2(einet: Einet, bn_model,
                                   scope: list, test_data: np.ndarray,
                                   var_names: list = None, device: str = "cpu"):
    """(B) Average marginal L1/L2 over all variables for Einet."""
    ve = VariableElimination(bn_model)
    local_data = test_data[:, scope].astype(np.int64)
    einet = einet.to(device)
    einet.eval()
    l1s, l2s = [], []
    for local_idx, var_id in enumerate(scope):
        cardinality = _get_cardinality(var_id, test_data)
        probs_spn = np.zeros(cardinality)
        with torch.no_grad():
            for v in range(cardinality):
                ld_v = local_data.copy()
                ld_v[:, local_idx] = v
                t = torch.tensor(ld_v, dtype=torch.long, device=device)
                lp = einet(t).squeeze(-1).cpu().numpy()
                probs_spn[v] = np.exp(np.mean(lp))
        s = probs_spn.sum()
        probs_spn = probs_spn / s if s > 0 else np.ones(cardinality) / cardinality
        try:
            probs_bn = _bn_marginal(ve, var_id, var_names)[:cardinality]
            l1, l2 = _l1_l2(probs_spn, probs_bn)
            l1s.append(l1); l2s.append(l2)
        except Exception:
            pass
    return (float(np.mean(l1s)) if l1s else np.nan,
            float(np.mean(l2s)) if l2s else np.nan)


def eval_einet_conditional_l1l2(einet: Einet, bn_model, target_var: int,
                                  scope: list, test_data: np.ndarray,
                                  evidence_var: int = None,
                                  var_names: list = None,
                                  device: str = "cpu"):
    """(C) Conditional L1/L2 for Einet."""
    if evidence_var is None:
        evidence_var = _find_most_dependent_var(target_var, scope, test_data)

    cardinality_target = _get_cardinality(target_var, test_data)
    cardinality_ev = _get_cardinality(evidence_var, test_data)
    ve = VariableElimination(bn_model)
    target_local   = scope.index(target_var)
    evidence_local = scope.index(evidence_var)

    l1s, l2s = [], []
    for ev_val in range(cardinality_ev):
        p_spn = _einet_conditional_probs(einet, scope, target_local,
                                          evidence_local, ev_val,
                                          cardinality_target, test_data, device)
        p_bn  = _bn_conditional(ve, target_var, evidence_var, ev_val, var_names)
        if p_spn is None or p_bn is None:
            continue
        p_bn_trimmed = p_bn[:cardinality_target]
        l1, l2 = _l1_l2(p_spn, p_bn_trimmed)
        l1s.append(l1); l2s.append(l2)

    return (float(np.mean(l1s)) if l1s else np.nan,
            float(np.mean(l2s)) if l2s else np.nan,
            evidence_var)
