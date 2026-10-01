import copy
import os
import numpy as np
import pandas as pd
import bnlearn as bn
import json
from pgmpy.metrics import log_likelihood_score
from lib.learnSPN import learn_spn_globally
from lib.customspn import SPN, LeafNode, ProductNode
from lib.insertion import insert_variable

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lib.utils import count_nodes, get_depth_differences
import hydra
from omegaconf import DictConfig
import time
from lib.synthetic import generate_synthetic_bn, sample_synthetic_data
from rtpt import RTPT

def load_data(dataset_name="asia", n_samples=10000, seed=42, cfg=None, synthetic_model=None):
    benchmark_dir = os.path.join("data", "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)


    if cfg is not None and 'synthetic' in cfg and cfg.synthetic.get('use_synthetic', False):
        num_nodes = cfg.synthetic.num_nodes
        edge_prob = cfg.synthetic.edge_prob
        max_states = cfg.synthetic.get('max_states', 2) 
        bn_seed = cfg.synthetic.bn_seed
        print(f"Loading/generating synthetic data")
        
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
            
        # bnlearn dict structure
        model = {'model': model_obj} 

    
    else:
        bif_dir = os.path.join("data", "bns")
        os.makedirs(bif_dir, exist_ok=True)
        local_bif_path = os.path.join(bif_dir, f"{dataset_name}.bif")
        
        if os.path.exists(local_bif_path):
            model_path_or_name = local_bif_path
        else:
            print(f"Local file {local_bif_path} not found. Defaulting to bnlearn internal library...")
            model_path_or_name = dataset_name

        csv_filename = os.path.join(benchmark_dir, f"{dataset_name}_{n_samples}_seed{seed}_benchmark.csv")
        
        if os.path.exists(csv_filename):
            print(f"Loading existing benchmark from {csv_filename}...")
            df = pd.read_csv(csv_filename)
            model = bn.import_DAG(model_path_or_name, verbose=0) 
        else:
            print(f"Generating new {dataset_name} dataset for seed {seed} into {benchmark_dir}/...")
            model = bn.import_DAG(model_path_or_name)
            np.random.seed(seed)
            df = bn.sampling(model, n=n_samples)
            df.to_csv(csv_filename, index=False)
            
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = LabelEncoder().fit_transform(df[col])
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    return train_df.to_numpy().astype(int), test_df.to_numpy().astype(int), train_df, test_df, model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    rtpt = RTPT(name_initials='BP', experiment_name='TestingSPNs', max_iterations=50)
    rtpt.start()
    
    print(f"Running experiment: {cfg.experiment.name}")
    
    dataset_name = cfg.dataset.name
    n_samples = cfg.dataset.n_samples
    seeds = cfg.benchmark.seeds
    threshold = cfg.algorithm.g_test_threshold
    min_inst = cfg.algorithm.min_instances
    save_dir = cfg.experiment.save_dir
    os.makedirs(save_dir, exist_ok=True)
    experiment_results = []
    
    
    synthetic_model = None
    if 'synthetic' in cfg and cfg.synthetic.get('use_synthetic', False):
        dataset_name = f"synthetic_bn{cfg.synthetic.bn_seed}_n{cfg.synthetic.num_nodes}_p{cfg.synthetic.edge_prob}_s{cfg.synthetic.get('max_states', 2)}"
        synthetic_model = generate_synthetic_bn(
            cfg.synthetic.num_nodes,
            cfg.synthetic.edge_prob,
            cfg.synthetic.get('max_states', 2),
            cfg.synthetic.bn_seed
        )
    else:
        dataset_name = cfg.dataset.name

    for seed in seeds:
        print(f"\n{'-'*50}")
        print(f" RUNNING BENCHMARK | DATASET: {dataset_name.upper()} | SEED: {seed}")
        print(f"{'-'*50}")
        
        # Load data for this seed
        train_data, test_data, train_df, test_df, bn_model = load_data(
            dataset_name, n_samples, seed, cfg, synthetic_model=synthetic_model
        )
        total_vars = train_data.shape[1]
        n_test = len(test_data)
        log_n_test = np.log(n_test)
        # Ground truth ll
        true_ll = log_likelihood_score(bn_model['model'], test_df) / n_test
        
        
        np.random.seed(100) 
        num_to_test = min(5, total_vars)
        test_variables = np.random.choice(total_vars, size=num_to_test, replace=False)
        #Full retraining
        all_var_ids = list(range(total_vars))
        t0 = time.time()
        full_spn_root = learn_spn_globally(train_data, scope=all_var_ids, min_instances=min_inst, p_threshold=threshold)
        full_retrain_time = time.time() - t0
        full_spn = SPN(root=full_spn_root)
        full_nodes = count_nodes(full_spn_root)
        full_spn_ll = np.mean(full_spn.eval(test_data))
        full_total_ll = full_spn_ll * n_test
        full_aic = 2 * full_nodes - 2 * full_total_ll
        full_bic = full_nodes * log_n_test - 2 * full_total_ll
    
        for target_var_id in test_variables:
            print(f"\n Inserting variable {target_var_id} (seed {seed})")
            
            base_var_ids = [v for v in range(total_vars) if v != target_var_id]

            # Old SPN
            old_root = learn_spn_globally(train_data, scope=base_var_ids, min_instances=min_inst, p_threshold=threshold)

            #Baseline
            y_counts = np.bincount(train_data[:, target_var_id])
            y_probs = (y_counts + 0.1) / (y_counts.sum() + len(y_counts) * 0.1) 
            baseline_leaf = LeafNode(scope=target_var_id, params=y_probs)
            baseline_spn = SPN(root=ProductNode(children=[copy.deepcopy(old_root), baseline_leaf]))

            #ContinualSPN
            t1 = time.time()
            updated_root = insert_variable(copy.deepcopy(old_root), target_var_id, train_data)
            continual_update_time = time.time() - t1
            updated_spn = SPN(root=updated_root)
            added_dict, total_added = get_depth_differences(old_root, updated_root)
            print(f"Total Nodes Added: {total_added}")
            print(f"Additions by Depth: {added_dict}")

            baseline_ll = np.mean(baseline_spn.eval(test_data))
            updated_ll = np.mean(updated_spn.eval(test_data))
            

            initial_nodes = count_nodes(old_root)
            baseline_nodes = count_nodes(baseline_spn.root)
            continual_nodes = count_nodes(updated_root)
            
            
            
            base_total_ll = baseline_ll * n_test
            base_aic = 2 * baseline_nodes - 2 * base_total_ll
            base_bic = baseline_nodes * log_n_test - 2 * base_total_ll
            
            cont_total_ll = updated_ll * n_test
            cont_aic = 2 * continual_nodes - 2 * cont_total_ll
            cont_bic = continual_nodes * log_n_test - 2 * cont_total_ll
            
            print(f"Initial Nodes: {initial_nodes}")
            print(f"Baseline LL:      {baseline_ll:.6f} | Nodes: {baseline_nodes} | BIC: {base_bic:.2f}")
            print(f"Continual SPN LL: {updated_ll:.6f} | Nodes: {continual_nodes} | BIC: {cont_bic:.2f}")
            print(f"Full SPN LL:      {full_spn_ll:.6f} | Nodes: {full_nodes} | BIC: {full_bic:.2f}")
            print(f"BN Ground Truth:  {true_ll:.6f}")
            
            print(f"Continual Update Time:  {continual_update_time:.4f} seconds")
            print(f"Full Retraining Time:      {full_retrain_time:.4f} seconds")
            
            from pgmpy.metrics import BayesianModelProbability

            log_p_bn       = BayesianModelProbability(bn_model['model']).log_probability(test_df)
            # test_df already has all BN variables — same object used for true_ll above

            log_p_baseline = baseline_spn.eval(test_data)   # already returns per-point log-probs
            log_p_continual = updated_spn.eval(test_data)
            log_p_full     = full_spn.eval(test_data)

            def prob_metrics(log_p_model, log_p_ref):
                p_model = np.exp(log_p_model)
                p_ref   = np.exp(log_p_ref)
                diff    = np.abs(p_model - p_ref)
                return float(np.mean(diff)), float(np.sqrt(np.mean(diff**2))), float(np.mean(diff / (p_ref + 1e-300)))

            base_l1, base_l2, base_mare = prob_metrics(log_p_baseline, log_p_bn)
            cont_l1, cont_l2, cont_mare = prob_metrics(log_p_continual, log_p_bn)
            full_l1, full_l2, full_mare = prob_metrics(log_p_full,      log_p_bn)
            
            experiment_results.append({
                "dataset": dataset_name,
                "seed": int(seed),
                "target_variable": int(target_var_id),
                "baseline_ll": float(baseline_ll),
                "continual_ll": float(updated_ll),
                "full_spn_ll": float(full_spn_ll),
                "ground_truth_ll": float(true_ll),
                "continual_node_count": int(continual_nodes),
                "full_node_count": int(full_nodes),
                "baseline_aic": float(base_aic),
                "baseline_bic": float(base_bic),
                "continual_aic": float(cont_aic),
                "continual_bic": float(cont_bic),
                "full_aic": float(full_aic),
                "full_bic": float(full_bic),
                "continual_time": float(continual_update_time),
                "full_retraining_time": float(full_retrain_time),
                
                "baseline_l1":    base_l1,   "baseline_l2":    base_l2,   "baseline_mare":    base_mare,
                "continual_l1":   cont_l1,   "continual_l2":   cont_l2,   "continual_mare":   cont_mare,
                "full_l1":        full_l1,   "full_l2":        full_l2,   "full_mare":        full_mare
            })
            rtpt.step()
            
        # Save the JSON after every seed
        json_path = os.path.join(save_dir,
                    f"{dataset_name}_L1L2_min{min_inst}_pth{threshold}.json")
        with open(json_path, "w") as json_file:
            json.dump(experiment_results, json_file, indent=4)


if __name__ == "__main__":
    main()