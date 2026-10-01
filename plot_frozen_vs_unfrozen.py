import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

C_FRZ  = "#E5AE38"   
C_UF0  = "#B39DDB"   
C_UF01 = "#7E57C2"   
C_UF1  = "#512DA8"   

def load_data(path, metric="l1"):
    with open(path) as f:
        raw = json.load(f)
    
    buckets = defaultdict(lambda: {k: [] for k in ["frz", "uf0", "uf01", "uf1"]})
    
    for row in raw:
        if "rat_native_matched_ll" not in row:
            continue
        v = row["target_variable"]
        
        if metric == "l1":
            buckets[v]["frz"].append(row["rat_native_matched_l1"])
            buckets[v]["uf0"].append(row["rat_native_unfrozen_noise0.0_l1"])
            buckets[v]["uf01"].append(row["rat_native_unfrozen_noise0.01_l1"])
            buckets[v]["uf1"].append(row["rat_native_unfrozen_noise0.1_l1"])
        elif metric == "ll":
            buckets[v]["frz"].append(row["rat_native_matched_ll"])
            buckets[v]["uf0"].append(row["rat_native_unfrozen_noise0.0_ll"])
            buckets[v]["uf01"].append(row["rat_native_unfrozen_noise0.01_ll"])
            buckets[v]["uf1"].append(row["rat_native_unfrozen_noise0.1_ll"])
        
    dataset = raw[0]["dataset"]
    n_seeds = len(set(r["seed"] for r in raw))
    vars_   = sorted(buckets.keys())

    # Build var_id - name mapping 
    var_id_to_name = {}
    for row in raw:
        vid = row["target_variable"]
        vname = row.get("target_variable_name")
        if vname and vid not in var_id_to_name:
            var_id_to_name[vid] = str(vname)
    
    data = {
        "dataset": dataset, "n_seeds": n_seeds,
        "vars": [var_id_to_name.get(v, f"Var {v}") for v in vars_],
        "metric": metric,
        "frz_m": np.array([np.mean(buckets[v]["frz"]) for v in vars_]),
        "frz_s": np.array([np.std(buckets[v]["frz"], ddof=0) for v in vars_]),
        "uf0_m": np.array([np.mean(buckets[v]["uf0"]) for v in vars_]),
        "uf0_s": np.array([np.std(buckets[v]["uf0"], ddof=0) for v in vars_]),
        "uf01_m": np.array([np.mean(buckets[v]["uf01"]) for v in vars_]),
        "uf01_s": np.array([np.std(buckets[v]["uf01"], ddof=0) for v in vars_]),
        "uf1_m": np.array([np.mean(buckets[v]["uf1"]) for v in vars_]),
        "uf1_s": np.array([np.std(buckets[v]["uf1"], ddof=0) for v in vars_])
    }
    
    if metric == "l1" and "full_l1" in raw[0]:
        data["full_ref"] = np.mean([r["full_l1"] for r in raw])
    elif metric == "ll" and "full_spn_ll" in raw[0]:
        data["full_ref"] = np.mean([r["full_spn_ll"] for r in raw])
    else:
        data["full_ref"] = None
        
    return data

def plot_frz_uf(data, out_path):
    metric = data["metric"]
    vars_ = data["vars"]
    x = np.arange(len(vars_))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(max(8, len(vars_) * 1.5), 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    err_kw = dict(linewidth=1.2, capsize=4, capthick=1.2, color='black', alpha=0.7)
    
    ax.bar(x - width*1.5, data["frz_m"], width, yerr=data["frz_s"], color=C_FRZ, label="Frozen (Phase 2)", error_kw=err_kw)
    ax.bar(x - width*0.5, data["uf0_m"], width, yerr=data["uf0_s"], color=C_UF0, label="Unfrozen (σ=0)", error_kw=err_kw)
    ax.bar(x + width*0.5, data["uf01_m"], width, yerr=data["uf01_s"], color=C_UF01, label="Unfrozen (σ=0.01)", error_kw=err_kw)
    ax.bar(x + width*1.5, data["uf1_m"], width, yerr=data["uf1_s"], color=C_UF1, label="Unfrozen (σ=0.1)", error_kw=err_kw)
    
    if data["full_ref"] is not None:
        ax.axhline(data["full_ref"], color="#8C613C", linestyle="--", linewidth=2, label=f"Full Retrain SPN")
        
    if metric == "l1":
        ax.set_yscale("log")
        ax.set_ylabel("L1 Distance (log scale, lower is better)", fontsize=11)
        ax.set_title(f"{data['dataset'].capitalize()} - RAT-SPN Frozen vs Unfrozen L1 ({data['n_seeds']} seeds)", fontsize=12, fontweight="bold")
    else:
        # LL is negative, so normal scale but compute max bounds
        ax.set_ylabel("Log-Likelihood", fontsize=11)
        ax.set_title(f"{data['dataset'].capitalize()} - RAT-SPN Frozen vs Unfrozen LL ({data['n_seeds']} seeds)", fontsize=12, fontweight="bold")
        all_means = [data["frz_m"], data["uf0_m"], data["uf01_m"], data["uf1_m"]]
        all_stds = [data["frz_s"], data["uf0_s"], data["uf01_s"], data["uf1_s"]]
        min_y = min([np.min(m - s) for m, s in zip(all_means, all_stds)])
        max_y = max([np.max(m + s) for m, s in zip(all_means, all_stds)])
        if data["full_ref"] is not None:
            max_y = max(max_y, data["full_ref"])
        padding = (max_y - min_y) * 0.1
        ax.set_ylim(min_y - padding, max_y + padding)
        
    ax.set_xticks(x)
    ax.set_xticklabels(vars_)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, framealpha=0.95)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#CCCCCC")
    
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close()
    print(f"Saved {out_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_frozen_vs_unfrozen.py path/to/file1.json")
        sys.exit(1)
        
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)
        
    # Plot L1
    data_l1 = load_data(path, metric="l1")
    out_l1 = path.parent / f"{data_l1['dataset']}_frz_vs_ufrz_l1_bars.png"
    plot_frz_uf(data_l1, out_l1)
    
    # Plot LL
    data_ll = load_data(path, metric="ll")
    out_ll = path.parent / f"{data_ll['dataset']}_frz_vs_ufrz_ll_bars.png"
    plot_frz_uf(data_ll, out_ll)

if __name__ == "__main__":
    main()
