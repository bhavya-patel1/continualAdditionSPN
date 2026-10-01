import copy
import os
import numpy as np
import pandas as pd
import bnlearn as bn
import json
import hydra
import time
from omegaconf import DictConfig
from pgmpy.metrics import log_likelihood_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from lib.learnSPN import learn_spn_globally
from lib.customspn import SPN, LeafNode, ProductNode
from lib.insertion import insert_variable
from lib.utils import count_nodes, get_depth_differences
from lib.synthetic import generate_synthetic_bn, sample_synthetic_data


def load_data(dataset_name="asia", n_samples=10000, seed=42, cfg=None, synthetic_model=None):
    benchmark_dir = os.path.join("data", "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)

    if cfg is not None and 'synthetic' in cfg and cfg.synthetic.get('use_synthetic', False):
        num_nodes   = cfg.synthetic.num_nodes
        edge_prob   = cfg.synthetic.edge_prob
        max_states  = cfg.synthetic.get('max_states', 2)
        bn_seed     = cfg.synthetic.bn_seed
        print(f"Generating SYNTHETIC dataset | Nodes: {num_nodes} | Edge Prob: {edge_prob} | Max States: {max_states}")

        csv_filename = os.path.join(benchmark_dir,
            f"synthetic_bn{bn_seed}_n{num_nodes}_{n_samples}_seed{seed}.csv")

        model_obj = synthetic_model
        if model_obj is None:
            model_obj = generate_synthetic_bn(num_nodes, edge_prob, max_states, bn_seed)

        if os.path.exists(csv_filename):
            print(f"Loading existing synthetic benchmark from {csv_filename}...")
            df = pd.read_csv(csv_filename)
            df.columns = [int(col) if str(col).isdigit() else col for col in df.columns]
        else:
            df = sample_synthetic_data(model_obj, n_samples, seed)
            df.to_csv(csv_filename, index=False)

        model = {'model': model_obj}

    else:
        bif_dir = os.path.join("data", "bns")
        os.makedirs(bif_dir, exist_ok=True)
        local_bif_path = os.path.join(bif_dir, f"{dataset_name}.bif")

        model_path_or_name = local_bif_path if os.path.exists(local_bif_path) else dataset_name

        csv_filename = os.path.join(benchmark_dir, f"{dataset_name}_{n_samples}_seed{seed}_benchmark.csv")

        if os.path.exists(csv_filename):
            print(f"Loading existing benchmark from {csv_filename}...")
            df = pd.read_csv(csv_filename)
            model = bn.import_DAG(model_path_or_name, verbose=0)
        else:
            print(f"Generating new {dataset_name} dataset for seed {seed} into {benchmark_dir}/...")
            model = bn.import_DAG(model_path_or_name)
            df = bn.sampling(model, n=n_samples)
            df.to_csv(csv_filename, index=False)

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = LabelEncoder().fit_transform(df[col])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    return train_df.to_numpy().astype(int), test_df.to_numpy().astype(int), train_df, test_df, model



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Running sequential insertion experiment: {cfg.experiment.name}")

    dataset_name  = cfg.dataset.name
    n_samples     = cfg.dataset.n_samples
    seeds         = cfg.benchmark.seeds
    threshold     = cfg.algorithm.g_test_threshold
    min_inst      = cfg.algorithm.min_instances
    save_dir      = cfg.experiment.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Optional: override insertion order from config, else use random
    # cfg.sequential.insertion_order: list[int] | null
    manual_order = cfg.get("sequential", {}).get("insertion_order", None)

    synthetic_model = None
    if 'synthetic' in cfg and cfg.synthetic.get('use_synthetic', False):
        dataset_name = (
            f"synthetic_bn{cfg.synthetic.bn_seed}"
            f"_n{cfg.synthetic.num_nodes}"
            f"_p{cfg.synthetic.edge_prob}"
            f"_s{cfg.synthetic.get('max_states', 2)}"
        )
        synthetic_model = generate_synthetic_bn(
            cfg.synthetic.num_nodes,
            cfg.synthetic.edge_prob,
            cfg.synthetic.get('max_states', 2),
            cfg.synthetic.bn_seed,
        )

    all_results = []

    for seed in seeds:
        print(f"\n{'─'*55}")
        print(f" SEQUENTIAL INSERTION | DATASET: {dataset_name.upper()} | SEED: {seed}")
        print(f"{'─'*55}")

        train_data, test_data, train_df, test_df, bn_model = load_data(
            dataset_name, n_samples, seed, cfg, synthetic_model=synthetic_model
        )

        total_vars = train_data.shape[1]
        n_test     = len(test_data)
        log_n_test = np.log(n_test)

        true_ll = log_likelihood_score(bn_model['model'], test_df) / n_test

        # Determine insertion order 
        if manual_order is not None:
            insertion_order = list(manual_order)
            assert all(0 <= v < total_vars for v in insertion_order), \
                "insertion_order contains out-of-range variable ids"
        else:
            np.random.seed(seed)
            insertion_order = np.random.permutation(total_vars).tolist()

        print(f"Insertion order: {insertion_order}")

        # Full retrain on ALL variables
        all_var_ids = list(range(total_vars))
        t0 = time.time()
        full_spn_root = learn_spn_globally(
            train_data, scope=all_var_ids, min_instances=min_inst, p_threshold=threshold
        )
        full_retrain_time = time.time() - t0
        full_spn       = SPN(root=full_spn_root)
        full_nodes     = count_nodes(full_spn_root)
        full_spn_ll    = np.mean(full_spn.eval(test_data))
        full_total_ll  = full_spn_ll * n_test
        full_aic       = 2 * full_nodes - 2 * full_total_ll
        full_bic       = full_nodes * log_n_test - 2 * full_total_ll

        # Sequential insertion 
        # Start with the first variable as a single-leaf SPN, then insert the rest
        first_var          = insertion_order[0]
        vars_in_spn        = [first_var]
        current_root       = learn_spn_globally(
            train_data, scope=[first_var], min_instances=min_inst, p_threshold=threshold
        )
        cumulative_cont_time = 0.0

        for step, new_var in enumerate(insertion_order[1:], start=1):
            print(f"\n  Step {step}: inserting variable {new_var} | current scope: {vars_in_spn}")

            # Naive baseline: product of marginal leaf at root 
            y_counts      = np.bincount(train_data[:, new_var])
            y_probs       = (y_counts + 0.1) / (y_counts.sum() + len(y_counts) * 0.1)
            baseline_leaf = LeafNode(scope=new_var, params=y_probs)
            baseline_root = ProductNode(children=[copy.deepcopy(current_root), baseline_leaf])
            baseline_spn  = SPN(root=baseline_root)

            # Continual insertion
            t1 = time.time()
            updated_root = insert_variable(copy.deepcopy(current_root), new_var, train_data)
            step_time    = time.time() - t1
            cumulative_cont_time += step_time
            updated_spn  = SPN(root=updated_root)

            added_dict, total_added = get_depth_differences(current_root, updated_root)
            print(f"    Nodes added: {total_added} | depth breakdown: {added_dict}")

            # Full retrain on current scope + new var 
            current_scope_full = vars_in_spn + [new_var]
            t2 = time.time()
            step_full_root = learn_spn_globally(
                train_data, scope=current_scope_full, min_instances=min_inst, p_threshold=threshold
            )
            step_full_time = time.time() - t2
            step_full_spn  = SPN(root=step_full_root)

            # Evaluate on test set 
            baseline_ll  = np.mean(baseline_spn.eval(test_data))
            continual_ll = np.mean(updated_spn.eval(test_data))
            step_full_ll = np.mean(step_full_spn.eval(test_data))

            baseline_nodes  = count_nodes(baseline_root)
            continual_nodes = count_nodes(updated_root)
            step_full_nodes = count_nodes(step_full_root)

            def aic_bic(ll, n_nodes):
                total = ll * n_test
                return 2 * n_nodes - 2 * total, n_nodes * log_n_test - 2 * total

            base_aic_v,  base_bic_v  = aic_bic(baseline_ll,  baseline_nodes)
            cont_aic_v,  cont_bic_v  = aic_bic(continual_ll, continual_nodes)
            sfull_aic_v, sfull_bic_v = aic_bic(step_full_ll, step_full_nodes)

            print(f"    Baseline  LL: {baseline_ll:.6f}")
            print(f"    Continual LL: {continual_ll:.6f} | nodes: {continual_nodes} | step time: {step_time:.2f}s")
            print(f"    Step-Full LL: {step_full_ll:.6f} | nodes: {step_full_nodes} | step time: {step_full_time:.2f}s")
            print(f"    Ground Truth: {true_ll:.6f}")

            all_results.append({
                "dataset":               dataset_name,
                "seed":                  int(seed),
                "step":                  int(step),
                "new_variable":          int(new_var),
                "scope_size":            int(len(vars_in_spn) + 1),
                "baseline_ll":           float(baseline_ll),
                "continual_ll":          float(continual_ll),
                "step_full_ll":          float(step_full_ll),
                "full_spn_ll":           float(full_spn_ll),   # all-variable retrain
                "ground_truth_ll":       float(true_ll),
                "baseline_nodes":        int(baseline_nodes),
                "continual_nodes":       int(continual_nodes),
                "step_full_nodes":       int(step_full_nodes),
                "full_nodes":            int(full_nodes),
                "baseline_aic":          float(base_aic_v),
                "baseline_bic":          float(base_bic_v),
                "continual_aic":         float(cont_aic_v),
                "continual_bic":         float(cont_bic_v),
                "step_full_aic":         float(sfull_aic_v),
                "step_full_bic":         float(sfull_bic_v),
                "full_aic":              float(full_aic),
                "full_bic":              float(full_bic),
                "continual_step_time":   float(step_time),
                "step_full_time":        float(step_full_time),
                "continual_cumul_time":  float(cumulative_cont_time),
                "full_retrain_time":     float(full_retrain_time),
            })

            # Advance the current SPN and scope
            current_root = updated_root
            vars_in_spn.append(new_var)

        # Save after every seed 
        json_path = os.path.join(
            save_dir,
            f"{dataset_name}_min{min_inst}_pth{threshold}_sequential_results.json"
        )
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
