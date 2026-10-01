
import copy
import os
import time
import json
import random
import math
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import bnlearn as bn
from pgmpy.metrics import BayesianModelProbability, log_likelihood_score

from lib.learnSPN import learn_spn
from lib.customspn import SPN, LeafNode, ProductNode
from lib.insertion import insert_variable
from lib.utils import count_nodes
from lib.einet_wrapper import einet_log_prob, build_einet, train_einet
from lib.einet_inserter import insert_variable_einet_native
from lib.einet_converter import train_and_convert_einet
from lib.synthetic import generate_synthetic_bn, sample_synthetic_data
from lib.query_evaluator import (
    eval_marginal_l1l2, eval_all_marginals_l1l2, eval_conditional_l1l2,
    eval_einet_marginal_l1l2, eval_einet_all_marginals_l1l2, eval_einet_conditional_l1l2,
)
from lib.auto_plot import generate_einet_plots, make_experiment_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Probability distance metrics (joint)
def prob_metrics(log_p_model, log_p_ref):
    p_model = np.exp(log_p_model)
    p_ref   = np.exp(log_p_ref)
    diff    = np.abs(p_model - p_ref)
    return float(np.mean(diff)), float(np.sqrt(np.mean(diff**2))), float(np.mean(diff / (p_ref + 1e-300)))




NOISE_LEVELS = [0.0, 0.01, 0.1]   # for unfrozen experiment


def compute_max_depth(num_features: int) -> int:
    return max(1, int(math.floor(math.log2(max(1, num_features)))))


def derive_einet_cfg(num_features: int, num_bins: int, num_params: str = "matched") -> dict:
    max_depth = compute_max_depth(num_features)

    base_num_sums = max(4, min(32, 2 * num_features))
    base_num_leaves = max(2, min(16, 2 * num_bins))
    base_num_repetitions = 4 if num_features <= 20 else 6

    if num_params == "matched":
        return {
            "depth": max(1, max_depth - 1),
            "num_sums": base_num_sums,
            "num_leaves": base_num_leaves,
            "num_repetitions": base_num_repetitions,
        }
    elif num_params == "overfit":
        return {
            "depth": max_depth,
            "num_sums": min(64, 2 * base_num_sums),
            "num_leaves": min(32, 2 * base_num_leaves),
            "num_repetitions": min(8, base_num_repetitions + 2),
        }
    else:
        raise ValueError(f"Unknown num_params: {num_params}")



# Data loading
def load_data(dataset_name, n_samples, seed, synthetic_model=None):
    if synthetic_model is not None:
        df = sample_synthetic_data(synthetic_model, num_samples=n_samples, seed=seed)
        for col in df.columns:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                df[col] = LabelEncoder().fit_transform(df[col])
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        bn_model = {"model": synthetic_model}
        return (
            train_df.to_numpy().astype(int),
            test_df.to_numpy().astype(int),
            train_df, test_df, bn_model,
        )

    bif_dir = os.path.join("data", "bns")
    os.makedirs(bif_dir, exist_ok=True)
    local_bif = os.path.join(bif_dir, f"{dataset_name}.bif")
    model_path = local_bif if os.path.exists(local_bif) else dataset_name

    bench_dir = os.path.join("data", "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    csv_path = os.path.join(bench_dir, f"{dataset_name}_{n_samples}_seed{seed}_benchmark.csv")

    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        model = bn.import_DAG(model_path, verbose=0)
    else:
        model = bn.import_DAG(model_path)
        np.random.seed(seed) 
        df = bn.sampling(model, n=n_samples)
        df.to_csv(csv_path, index=False)

    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = LabelEncoder().fit_transform(df[col])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    return (
        train_df.to_numpy().astype(int),
        test_df.to_numpy().astype(int),
        train_df, test_df, model,
    )


def eval_einet_ll(einet, data, scope, device="cpu"):
    return float(np.mean(einet_log_prob(einet, data, scope, device=device)))



# Query evaluation helper (runs all 3 query types for one method)
def run_query_evals_spn(spn, bn_model, target_var, scope, test_data,
                        var_names=None, device="cpu"):
    """Returns dict of query metrics for a custom SPN."""
    res = {}
    try:
        l1, l2 = eval_marginal_l1l2(spn, bn_model, target_var, scope, test_data,
                                     var_names=var_names)
        res["qA_marginal_target_l1"] = l1; res["qA_marginal_target_l2"] = l2
    except Exception as e:
        res["qA_marginal_target_l1"] = None; res["qA_marginal_target_l2"] = None
        print(f"    [Query A] Error: {e}")

    try:
        l1, l2 = eval_all_marginals_l1l2(spn, bn_model, scope, test_data,
                                          var_names=var_names)
        res["qB_marginal_all_l1"] = l1; res["qB_marginal_all_l2"] = l2
    except Exception as e:
        res["qB_marginal_all_l1"] = None; res["qB_marginal_all_l2"] = None
        print(f"    [Query B] Error: {e}")

    try:
        l1, l2, ev_var = eval_conditional_l1l2(spn, bn_model, target_var, scope, test_data,
                                               var_names=var_names)
        res["qC_conditional_l1"] = l1; res["qC_conditional_l2"] = l2
        res["qC_evidence_var"] = int(ev_var)
    except Exception as e:
        res["qC_conditional_l1"] = None; res["qC_conditional_l2"] = None
        print(f"    [Query C] Error: {e}")

    return res


def run_query_evals_einet(einet, bn_model, target_var, scope, test_data,
                          var_names=None, device="cpu"):
    """Returns dict of query metrics for an Einet."""
    res = {}
    try:
        l1, l2 = eval_einet_marginal_l1l2(einet, bn_model, target_var, scope, test_data,
                                           var_names=var_names, device=device)
        res["qA_marginal_target_l1"] = l1; res["qA_marginal_target_l2"] = l2
    except Exception as e:
        res["qA_marginal_target_l1"] = None; res["qA_marginal_target_l2"] = None
        print(f"    [Query A Einet] Error: {e}")

    try:
        l1, l2 = eval_einet_all_marginals_l1l2(einet, bn_model, scope, test_data,
                                                var_names=var_names, device=device)
        res["qB_marginal_all_l1"] = l1; res["qB_marginal_all_l2"] = l2
    except Exception as e:
        res["qB_marginal_all_l1"] = None; res["qB_marginal_all_l2"] = None
        print(f"    [Query B Einet] Error: {e}")

    try:
        l1, l2, ev_var = eval_einet_conditional_l1l2(einet, bn_model, target_var, scope,
                                                      test_data, var_names=var_names, device=device)
        res["qC_conditional_l1"] = l1; res["qC_conditional_l2"] = l2
        res["qC_evidence_var"] = int(ev_var)
    except Exception as e:
        res["qC_conditional_l1"] = None; res["qC_conditional_l2"] = None
        print(f"    [Query C Einet] Error: {e}")

    return res


def _prefix_dict(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items()}



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    n_samples    = cfg.dataset.n_samples
    seeds        = list(cfg.benchmark.seeds)
    threshold    = cfg.algorithm.g_test_threshold
    min_inst     = cfg.algorithm.min_instances
    save_dir     = cfg.experiment.save_dir
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    einet_cfg  = cfg.get("einet", {})
    epochs     = int(einet_cfg.get("epochs",     100))
    lr         = float(einet_cfg.get("lr",        0.01))
    batch_size = int(einet_cfg.get("batch_size", 256))

    # Synthetic network support
    synthetic_model = None
    syn_cfg = cfg.get("synthetic", {})
    if syn_cfg.get("use_synthetic", False):
        edge_prob  = float(syn_cfg.get("edge_prob", 0.3))
        num_nodes  = int(syn_cfg.get("num_nodes", 15))
        max_states = int(syn_cfg.get("max_states", 3))
        bn_seed    = int(syn_cfg.get("bn_seed", 123))
        synthetic_model = generate_synthetic_bn(num_nodes, edge_prob, max_states, bn_seed)
        dataset_name = f"syn_n{num_nodes}_ep{edge_prob}_ms{max_states}"
        print(f"Using synthetic BN: {dataset_name}")

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name} | Seeds: {list(seeds)}")

    all_results = []

    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f" EINET BENCHMARK | {dataset_name.upper()} | SEED {seed}")
        print(f"{'─'*60}")
        set_seed(seed)
        train_data, test_data, train_df, test_df, bn_model = load_data(
            dataset_name, n_samples, seed, synthetic_model=synthetic_model
        )
        total_vars = train_data.shape[1]
        n_test     = len(test_data)
        true_ll    = log_likelihood_score(bn_model["model"], test_df) / n_test
        var_names  = list(train_df.columns)   # maps int index → BN node name

        np.random.seed(100)
        num_to_test  = min(5, total_vars)
        test_variables = np.random.choice(total_vars, size=num_to_test, replace=False)

        # Full retrains 
        all_var_ids = list(range(total_vars))

        t0 = time.time()
        full_spn_root = learn_spn(
            train_data,
            scope=all_var_ids,
            p_threshold=threshold,
            min_instances=min_inst,
            lam=cfg.algorithm.get("cluster_penalty", 1.0),
            alpha=cfg.algorithm.get("dirichlet_alpha", 1.0),
            max_k=cfg.algorithm.get("max_clusters", 5),
            max_depth=cfg.algorithm.get("max_depth", 50),
            max_iter=cfg.algorithm.get("em_max_iter", 50),
            random_state=seed,
        )
        full_spn_time = time.time() - t0
        full_spn      = SPN(root=full_spn_root)
        full_spn_log_p = full_spn.eval(test_data)
        full_spn_ll    = float(np.mean(full_spn_log_p))
        print(f"LearnSPN full retrain  | LL: {full_spn_ll:.6f} | time: {full_spn_time:.2f}s")

        num_bins = int(train_data.max()) + 1
        full_matched_cfg = derive_einet_cfg(total_vars, num_bins, num_params="matched")
        full_overfit_cfg = derive_einet_cfg(total_vars, num_bins, num_params="overfit")
        
        t0 = time.time()
        einet_m = build_einet(total_vars, num_bins, **full_matched_cfg)
        einet_m = train_einet(einet_m, train_data, all_var_ids, epochs=epochs, lr=lr, batch_size=batch_size, device=device)
        einet_m_ll = eval_einet_ll(einet_m, test_data, all_var_ids, device)
        print(f"Einet matched retrain | cfg={full_matched_cfg} | LL: {einet_m_ll:.6f}")

        t0 = time.time()
        einet_o = build_einet(total_vars, num_bins, **full_overfit_cfg)
        einet_o = train_einet(einet_o, train_data, all_var_ids, epochs=epochs, lr=lr, batch_size=batch_size, device=device)
        einet_o_ll = eval_einet_ll(einet_o, test_data, all_var_ids, device)
        print(f"Einet overfit retrain | cfg={full_overfit_cfg} | LL: {einet_o_ll:.6f}")


        log_p_bn = BayesianModelProbability(bn_model['model']).log_probability(test_df)

        # Per-variable insertion loop
        for target_var_id in test_variables:
            print(f"\n  Inserting variable {target_var_id}")
            base_scope = [v for v in all_var_ids if v != target_var_id]

            # Naive baseline 
            old_root_learn = learn_spn(
                train_data,
                scope=all_var_ids,
                p_threshold=threshold,
                min_instances=min_inst,
                lam=cfg.algorithm.get("cluster_penalty", 1.0),
                alpha=cfg.algorithm.get("dirichlet_alpha", 1.0),
                max_k=cfg.algorithm.get("max_clusters", 5),
                max_depth=cfg.algorithm.get("max_depth", 50),
                max_iter=cfg.algorithm.get("em_max_iter", 50),
                random_state=seed,
            )
            y_counts = np.bincount(train_data[:, target_var_id])
            y_probs  = (y_counts + 0.1) / (y_counts.sum() + len(y_counts) * 0.1)
            naive_root = ProductNode(children=[
                copy.deepcopy(old_root_learn),
                LeafNode(scope=target_var_id, params=y_probs)
            ])
            naive_spn    = SPN(root=naive_root)
            naive_log_p  = naive_spn.eval(test_data)
            naive_ll     = float(np.mean(naive_log_p))

            # ContinualSPN
            t1 = time.time()
            continual_root = insert_variable(copy.deepcopy(old_root_learn), target_var_id, train_data)
            continual_time = time.time() - t1
            continual_spn  = SPN(root=continual_root)
            continual_log_p = continual_spn.eval(test_data)
            continual_ll    = float(np.mean(continual_log_p))
            
            full_scope = base_scope + [target_var_id]
            insert_num_features = len(full_scope)
            insert_num_bins = int(train_data[:, full_scope].max()) + 1
            nat_matched_cfg = derive_einet_cfg(insert_num_features, insert_num_bins, num_params="matched")
            nat_overfit_cfg = derive_einet_cfg(insert_num_features, insert_num_bins, num_params="overfit")

            # RAT-SPN Native Frozen - matched
            t1 = time.time(); set_seed(seed)
            einet_nat_m, nat_m_scope = insert_variable_einet_native(
                train_data, base_scope, target_var_id,
                **nat_matched_cfg, freeze_base=True,
                epochs_phase1=epochs, epochs_phase2=epochs // 2,
                lr=lr, batch_size=batch_size, device=device, matched=True,
            )
            nat_m_time  = time.time() - t1
            nat_m_log_p = einet_log_prob(einet_nat_m, test_data, nat_m_scope, device)
            nat_m_ll    = float(np.mean(nat_m_log_p))

            # RAT-SPN Native Frozen - overparameterized
            t1 = time.time(); set_seed(seed)
            einet_nat_o, nat_o_scope = insert_variable_einet_native(
                train_data, base_scope, target_var_id,
                **nat_overfit_cfg, freeze_base=True,
                epochs_phase1=epochs, epochs_phase2=epochs // 2,
                lr=lr, batch_size=batch_size, device=device, matched=False,
            )
            nat_o_time  = time.time() - t1
            nat_o_log_p = einet_log_prob(einet_nat_o, test_data, nat_o_scope, device)
            nat_o_ll    = float(np.mean(nat_o_log_p))

            # RAT-SPN Native Unfrozen + noise  
            unfrozen_results = {}
            for noise in NOISE_LEVELS:
                noise_key = str(noise).replace(".", "")   # "00", "001", "01"
                t1 = time.time(); set_seed(seed)
                einet_uf, uf_scope = insert_variable_einet_native(
                    train_data, base_scope, target_var_id,
                    **nat_matched_cfg, freeze_base=False, noise_std=noise,
                    epochs_phase1=epochs, epochs_phase2=epochs // 2,
                    lr=lr, batch_size=batch_size, device=device, matched=True,
                )
                uf_time  = time.time() - t1
                uf_log_p = einet_log_prob(einet_uf, test_data, uf_scope, device)
                uf_ll    = float(np.mean(uf_log_p))
                l1, l2, mare = prob_metrics(uf_log_p, log_p_bn)
                unfrozen_results[noise] = {
                    "ll": uf_ll, "time": uf_time,
                    "l1": l1, "l2": l2, "mare": mare,
                    "log_p": uf_log_p, "einet": einet_uf, "scope": uf_scope,
                }
                print(f"    RAT-SPN Unfrz noise={noise}  LL: {uf_ll:.6f} | time: {uf_time:.2f}s")

            # RAT-SPN Converted - matched 
            t1 = time.time(); set_seed(seed)
            conv_m_root = train_and_convert_einet(
                train_data, base_scope, **nat_matched_cfg,
                epochs=epochs, lr=lr, batch_size=batch_size, device=device, matched=True,
            )
            conv_m_root = insert_variable(conv_m_root, target_var_id, train_data)
            conv_m_time = time.time() - t1
            conv_m_log_p = SPN(root=conv_m_root).eval(test_data)
            conv_m_ll    = float(np.mean(conv_m_log_p))

            # RAT-SPN Converted - overparameterized
            t1 = time.time(); set_seed(seed)
            conv_o_root = train_and_convert_einet(
                train_data, base_scope, **nat_overfit_cfg,
                epochs=epochs, lr=lr, batch_size=batch_size, device=device, matched=False,
            )
            conv_o_root = insert_variable(conv_o_root, target_var_id, train_data)
            conv_o_time = time.time() - t1
            conv_o_log_p = SPN(root=conv_o_root).eval(test_data)
            conv_o_ll    = float(np.mean(conv_o_log_p))

            # Joint L1/L2 metrics
            naive_l1,   naive_l2,   naive_mare   = prob_metrics(naive_log_p,    log_p_bn)
            cont_l1,    cont_l2,    cont_mare    = prob_metrics(continual_log_p, log_p_bn)
            nat_m_l1,   nat_m_l2,   nat_m_mare   = prob_metrics(nat_m_log_p,    log_p_bn)
            nat_o_l1,   nat_o_l2,   nat_o_mare   = prob_metrics(nat_o_log_p,    log_p_bn)
            conv_m_l1,  conv_m_l2,  conv_m_mare  = prob_metrics(conv_m_log_p,   log_p_bn)
            conv_o_l1,  conv_o_l2,  conv_o_mare  = prob_metrics(conv_o_log_p,   log_p_bn)
            full_l1,    full_l2,    full_mare    = prob_metrics(full_spn_log_p, log_p_bn)

            print(f"    Naive baseline           LL: {naive_ll:.6f} | L1: {naive_l1:.4e}")
            print(f"    ContinualSPN             LL: {continual_ll:.6f} | L1: {cont_l1:.4e} | time: {continual_time:.2f}s")
            print(f"    RAT-SPN native frz (M)   LL: {nat_m_ll:.6f} | L1: {nat_m_l1:.4e} | time: {nat_m_time:.2f}s")
            print(f"    RAT-SPN native frz (O)   LL: {nat_o_ll:.6f} | L1: {nat_o_l1:.4e} | time: {nat_o_time:.2f}s")
            print(f"    RAT-SPN conv   (M)        LL: {conv_m_ll:.6f} | L1: {conv_m_l1:.4e} | time: {conv_m_time:.2f}s")
            print(f"    RAT-SPN conv   (O)        LL: {conv_o_ll:.6f} | L1: {conv_o_l1:.4e} | time: {conv_o_time:.2f}s")
            print(f"    LearnSPN full retrain     LL: {full_spn_ll:.6f}")
            print(f"    BN ground truth           LL: {true_ll:.6f}")

            # Probabilistic query evaluations 
            print("    Running query evaluations...")
            qr_naive  = run_query_evals_spn(naive_spn,    bn_model["model"], target_var_id, all_var_ids, test_data, var_names=var_names, device=device)
            qr_cont   = run_query_evals_spn(continual_spn,bn_model["model"], target_var_id, all_var_ids, test_data, var_names=var_names, device=device)
            qr_nat_m  = run_query_evals_einet(einet_nat_m, bn_model["model"], target_var_id, nat_m_scope, test_data, var_names=var_names, device=device)
            qr_nat_o  = run_query_evals_einet(einet_nat_o, bn_model["model"], target_var_id, nat_o_scope, test_data, var_names=var_names, device=device)
            
            # Queries for the full retrain baselines
            qr_full_spn = run_query_evals_spn(full_spn, bn_model["model"], target_var_id, all_var_ids, test_data, var_names=var_names, device=device)
            qr_einet_m  = run_query_evals_einet(einet_m, bn_model["model"], target_var_id, all_var_ids, test_data, var_names=var_names, device=device)
            qr_einet_o  = run_query_evals_einet(einet_o, bn_model["model"], target_var_id, all_var_ids, test_data, var_names=var_names, device=device)

            # Assemble result record 
            record = {
                "dataset":           dataset_name,
                "seed":              int(seed),
                "target_variable":   int(target_var_id),
                "target_variable_name": var_names[target_var_id],
                "ground_truth_ll":   float(true_ll),
                # Joint LL
                "naive_ll":          naive_ll,
                "continual_ll":      continual_ll,  "continual_time":    continual_time,
                "full_spn_ll":       full_spn_ll,   "full_spn_time":     full_spn_time,
                "einet_matched_full_ll": einet_m_ll,
                "einet_overfit_full_ll": einet_o_ll,
                "rat_native_matched_ll":  nat_m_ll, "rat_native_matched_time": nat_m_time,
                "rat_native_overfit_ll":  nat_o_ll, "rat_native_overfit_time": nat_o_time,
                "rat_conv_matched_ll":   conv_m_ll, "rat_conv_matched_time":  conv_m_time,
                "rat_conv_overfit_ll":   conv_o_ll, "rat_conv_overfit_time":  conv_o_time,
                
                "einet_full_matched_cfg": full_matched_cfg,
                "einet_full_overfit_cfg": full_overfit_cfg,
                "einet_insert_matched_cfg": nat_matched_cfg,
                "einet_insert_overfit_cfg": nat_overfit_cfg,
                # Joint L1/L2
                "naive_l1": naive_l1,  "naive_l2": naive_l2,  "naive_mare": naive_mare,
                "continual_l1": cont_l1,"continual_l2": cont_l2,"continual_mare": cont_mare,
                "rat_native_matched_l1": nat_m_l1, "rat_native_matched_l2": nat_m_l2, "rat_native_matched_mare": nat_m_mare,
                "rat_native_overfit_l1": nat_o_l1, "rat_native_overfit_l2": nat_o_l2, "rat_native_overfit_mare": nat_o_mare,
                "rat_conv_matched_l1":  conv_m_l1, "rat_conv_matched_l2":  conv_m_l2, "rat_conv_matched_mare":  conv_m_mare,
                "rat_conv_overfit_l1":  conv_o_l1, "rat_conv_overfit_l2":  conv_o_l2, "rat_conv_overfit_mare":  conv_o_mare,
                "full_l1": full_l1, "full_l2": full_l2, "full_mare": full_mare,
                # Probabilistic query metrics
                **_prefix_dict(qr_naive,  "naive"),
                **_prefix_dict(qr_cont,   "continual"),
                **_prefix_dict(qr_nat_m,  "rat_native_matched"),
                **_prefix_dict(qr_nat_o,  "rat_native_overfit"),
                **_prefix_dict(qr_full_spn, "full_spn"),
                **_prefix_dict(qr_einet_m,  "einet_matched_full"),
                **_prefix_dict(qr_einet_o,  "einet_overfit_full"),
            }

            # Unfrozen noise sweep results
            for noise, res in unfrozen_results.items():
                noise_key = f"rat_native_unfrozen_noise{noise}"
                record[f"{noise_key}_ll"]   = res["ll"]
                record[f"{noise_key}_time"] = res["time"]
                record[f"{noise_key}_l1"]   = res["l1"]
                record[f"{noise_key}_l2"]   = res["l2"]
                record[f"{noise_key}_mare"] = res["mare"]
                # Query evals for unfrozen
                qr_uf = run_query_evals_einet(res["einet"], bn_model["model"],
                                               target_var_id, res["scope"], test_data,
                                               var_names=var_names, device=device)
                for k, v in qr_uf.items():
                    record[f"{noise_key}_{k}"] = v

            all_results.append(record)

        # Save after each seed
        exp_dir = make_experiment_dir(
            save_dir, dataset_name, "einet",
            min_inst, threshold, seeds,
        )
        out_path = os.path.join(exp_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults saved → {out_path}")

    # Auto-generate all plots once every seed is done
    generate_einet_plots(json_path=out_path, dataset=dataset_name)


if __name__ == "__main__":
    main()