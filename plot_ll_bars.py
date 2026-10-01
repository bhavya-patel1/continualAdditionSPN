
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

C_NAIVE = "#D65F5F"  # Red
C_CONT  = "#4878CF"  # Blue
C_NAT   = "#E5AE38"  # Yellow/Gold
C_CONV  = "#6ACC64"  # Green
C_FULL  = "#8C613C"  # Brown
C_GT    = "#333333"  # Dark Gray


def load(path):
    with open(path) as f:
        raw = json.load(f)

    buckets = defaultdict(lambda: {k: [] for k in
        ["naive_ll", "cont_ll", "nat_m_ll", "conv_m_ll", "nat_o_ll", "conv_o_ll",
         "nat_uf0_ll", "nat_uf001_ll", "nat_uf01_ll"]})
        
    for row in raw:
        v = row["target_variable"]
        buckets[v]["naive_ll"].append(row.get("naive_ll", row.get("baseline_ll")))
        buckets[v]["cont_ll"].append(row["continual_ll"])
        
        if "rat_native_matched_ll" in row:
            buckets[v]["nat_m_ll"].append(row["rat_native_matched_ll"])
            buckets[v]["conv_m_ll"].append(row["rat_conv_matched_ll"])
            buckets[v]["nat_o_ll"].append(row["rat_native_overfit_ll"])
            buckets[v]["conv_o_ll"].append(row["rat_conv_overfit_ll"])
            
        if "rat_native_unfrozen_noise0.0_ll" in row:
            buckets[v]["nat_uf0_ll"].append(row["rat_native_unfrozen_noise0.0_ll"])
            buckets[v]["nat_uf001_ll"].append(row["rat_native_unfrozen_noise0.01_ll"])
            buckets[v]["nat_uf01_ll"].append(row["rat_native_unfrozen_noise0.1_ll"])

    dataset = raw[0]["dataset"]
    n_seeds = len(set(r["seed"] for r in raw))
    vars_   = sorted(buckets.keys())

    # Build var_id -> name mapping (falls back to "Var {id}" for old JSONs)
    var_id_to_name = {}
    for row in raw:
        vid = row["target_variable"]
        vname = row.get("target_variable_name")
        if vname and vid not in var_id_to_name:
            var_id_to_name[vid] = str(vname)

    data = {
        "dataset": dataset,
        "n_seeds": n_seeds,
        "vars":    [var_id_to_name.get(v, f"Var {v}") for v in vars_],
        "naive_ll_mean": np.array([np.mean(buckets[v]["naive_ll"]) for v in vars_]),
        "naive_ll_std":  np.array([np.std(buckets[v]["naive_ll"], ddof=0) for v in vars_]),
        "cont_ll_mean":  np.array([np.mean(buckets[v]["cont_ll"]) for v in vars_]),
        "cont_ll_std":   np.array([np.std(buckets[v]["cont_ll"],  ddof=0) for v in vars_]),
        "full_ll_mean":  np.mean([r["full_spn_ll"] for r in raw]) if "full_spn_ll" in raw[0] else None,
        "gt_ll_mean":    np.mean([r["ground_truth_ll"] for r in raw]) if "ground_truth_ll" in raw[0] else None,
    }
    
    if len(buckets[vars_[0]]["nat_m_ll"]) > 0:
        data.update({
            "has_einet": True,
            "nat_m_ll_mean": np.array([np.mean(buckets[v]["nat_m_ll"]) for v in vars_]),
            "nat_m_ll_std":  np.array([np.std(buckets[v]["nat_m_ll"], ddof=0) for v in vars_]),
            "conv_m_ll_mean": np.array([np.mean(buckets[v]["conv_m_ll"]) for v in vars_]),
            "conv_m_ll_std":  np.array([np.std(buckets[v]["conv_m_ll"], ddof=0) for v in vars_]),
            "nat_o_ll_mean": np.array([np.mean(buckets[v]["nat_o_ll"]) for v in vars_]),
            "nat_o_ll_std":  np.array([np.std(buckets[v]["nat_o_ll"], ddof=0) for v in vars_]),
            "conv_o_ll_mean": np.array([np.mean(buckets[v]["conv_o_ll"]) for v in vars_]),
            "conv_o_ll_std":  np.array([np.std(buckets[v]["conv_o_ll"], ddof=0) for v in vars_]),
        })
    else:
        data["has_einet"] = False

    if len(buckets[vars_[0]]["nat_uf0_ll"]) > 0:
        data.update({
            "has_unfrozen": True,
            "nat_uf0_ll_mean": np.array([np.mean(buckets[v]["nat_uf0_ll"]) for v in vars_]),
            "nat_uf0_ll_std":  np.array([np.std(buckets[v]["nat_uf0_ll"], ddof=0) for v in vars_]),
            "nat_uf001_ll_mean": np.array([np.mean(buckets[v]["nat_uf001_ll"]) for v in vars_]),
            "nat_uf001_ll_std":  np.array([np.std(buckets[v]["nat_uf001_ll"], ddof=0) for v in vars_]),
            "nat_uf01_ll_mean": np.array([np.mean(buckets[v]["nat_uf01_ll"]) for v in vars_]),
            "nat_uf01_ll_std":  np.array([np.std(buckets[v]["nat_uf01_ll"], ddof=0) for v in vars_]),
        })
    else:
        data["has_unfrozen"] = False

    return data


def plot_bars(data, out_path):
    dataset = data["dataset"]
    n_seeds = data["n_seeds"]
    vars_   = data["vars"]
    x       = np.arange(len(vars_))

    naive_m = data["naive_ll_mean"]
    naive_s = data["naive_ll_std"]
    cont_m  = data["cont_ll_mean"]
    cont_s  = data["cont_ll_std"]
    full_v  = data["full_ll_mean"]
    gt_v    = data["gt_ll_mean"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    err_kw = dict(linewidth=1.2, capsize=3, capthick=1.2, color='black', alpha=0.7)
    
    C_NAT_U0 = "#B39DDB"
    C_NAT_U1 = "#7E57C2"
    C_NAT_U2 = "#512DA8"
    
    # 9 bars per variable -> width = 0.09
    width = 0.09
    
    all_means = [naive_m, cont_m]
    all_stds = [naive_s, cont_s]
    
    if data.get("has_unfrozen"):
        nat_m = data["nat_m_ll_mean"]
        nat_s = data["nat_m_ll_std"]
        conv_m = data["conv_m_ll_mean"]
        conv_s = data["conv_m_ll_std"]
        nat_o_m = data["nat_o_ll_mean"]
        nat_o_s = data["nat_o_ll_std"]
        conv_o_m = data["conv_o_ll_mean"]
        conv_o_s = data["conv_o_ll_std"]
        
        uf0_m = data["nat_uf0_ll_mean"]
        uf0_s = data["nat_uf0_ll_std"]
        uf001_m = data["nat_uf001_ll_mean"]
        uf001_s = data["nat_uf001_ll_std"]
        uf01_m = data["nat_uf01_ll_mean"]
        uf01_s = data["nat_uf01_ll_std"]
        
        all_means.extend([nat_m, conv_m, nat_o_m, conv_o_m, uf0_m, uf001_m, uf01_m])
        all_stds.extend([nat_s, conv_s, nat_o_s, conv_o_s, uf0_s, uf001_s, uf01_s])
        
        ax.bar(x - width*4, naive_m, width, yerr=naive_s, color=C_NAIVE, label="Naive", error_kw=err_kw, zorder=3)
        ax.bar(x - width*3, cont_m,  width, yerr=cont_s,  color=C_CONT,  label="Continual SPN", error_kw=err_kw, zorder=3)
        ax.bar(x - width*2, nat_m, width, yerr=nat_s, color=C_NAT,   label="RAT-SPN Nat Frz (M)", error_kw=err_kw, zorder=3)
        ax.bar(x - width*1, conv_m,width, yerr=conv_s,color=C_CONV,  label="RAT-SPN Conv (M)", error_kw=err_kw, zorder=3)
        ax.bar(x, nat_o_m, width, yerr=nat_o_s, color=C_NAT, hatch='//', label="RAT-SPN Nat Frz (O)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*1, conv_o_m,width, yerr=conv_o_s,color=C_CONV, hatch='//', label="RAT-SPN Conv (O)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*2, uf0_m, width, yerr=uf0_s, color=C_NAT_U0, label="RAT-SPN Unfrz (σ=0)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*3, uf001_m, width, yerr=uf001_s, color=C_NAT_U1, label="RAT-SPN Unfrz (σ=0.01)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*4, uf01_m, width, yerr=uf01_s, color=C_NAT_U2, label="RAT-SPN Unfrz (σ=0.1)", error_kw=err_kw, zorder=3)
        
    elif data.get("has_einet"):
        width = 0.13
        nat_m = data["nat_m_ll_mean"]
        nat_s = data["nat_m_ll_std"]
        conv_m = data["conv_m_ll_mean"]
        conv_s = data["conv_m_ll_std"]
        nat_o_m = data["nat_o_ll_mean"]
        nat_o_s = data["nat_o_ll_std"]
        conv_o_m = data["conv_o_ll_mean"]
        conv_o_s = data["conv_o_ll_std"]
        
        all_means.extend([nat_m, conv_m, nat_o_m, conv_o_m])
        all_stds.extend([nat_s, conv_s, nat_o_s, conv_o_s])
        
        ax.bar(x - width*2.5, naive_m, width, yerr=naive_s, color=C_NAIVE, label="Naive Baseline", error_kw=err_kw, zorder=3)
        ax.bar(x - width*1.5, cont_m,  width, yerr=cont_s,  color=C_CONT,  label="Continual SPN",  error_kw=err_kw, zorder=3)
        ax.bar(x - width*0.5, nat_m, width, yerr=nat_s, color=C_NAT,   label="RAT-SPN Native (M)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*0.5, conv_m,width, yerr=conv_s,color=C_CONV,  label="RAT-SPN Conv. (M)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*1.5, nat_o_m, width, yerr=nat_o_s, color=C_NAT, hatch='//', label="RAT-SPN Native (O)", error_kw=err_kw, zorder=3)
        ax.bar(x + width*2.5, conv_o_m,width, yerr=conv_o_s,color=C_CONV, hatch='//', label="RAT-SPN Conv. (O)", error_kw=err_kw, zorder=3)
        
    else:
        width = 0.35
        ax.bar(x - width/2, naive_m, width, yerr=naive_s, color=C_NAIVE, label="Naive Baseline", error_kw=err_kw, zorder=3)
        ax.bar(x + width/2, cont_m,  width, yerr=cont_s,  color=C_CONT,  label="Continual SPN",  error_kw=err_kw, zorder=3)

    if full_v is not None:
        ax.axhline(full_v, color=C_FULL, linestyle="--", linewidth=2,
                   label=f"Full Retrain ({full_v:.2f})", zorder=4)
                   
    if gt_v is not None:
        ax.axhline(gt_v, color=C_GT, linestyle=":", linewidth=2,
                   label=f"Ground Truth ({gt_v:.2f})", zorder=4)

    # Make Y-axis visually meaningful since LLs are negative. 
    # Find min and max values to set appropriate y-limits
    min_ll = min([np.min(m - s) for m, s in zip(all_means, all_stds)])
    max_ll = max([np.max(m + s) for m, s in zip(all_means, all_stds)])
    
    if gt_v is not None: max_ll = max(max_ll, gt_v)
    if full_v is not None: max_ll = max(max_ll, full_v)
    
    padding = (max_ll - min_ll) * 0.1
    # Add extra horizontal padding so edges don't get clipped
    ax.set_xlim(-0.9, x[-1] + 0.9)

    # Explicitly set y-limits with padding on log scale to prevent clipping top error bars
    if max_ll > -float("inf"):
        # Make sure full retrain or other lines are included
        if full_v and not np.isnan(full_v): max_ll = max(max_ll, full_v)
        if full_v and not np.isnan(full_v): min_ll = min(min_ll, full_v)
        ax.set_ylim(min_ll - padding, max_ll + padding)

    ax.set_xticks(x)
    ax.set_xticklabels(vars_, fontsize=11)
    ax.set_xlabel("Target Variable", fontsize=11)
    ax.set_ylabel("Log-Likelihood (Higher is Better)", fontsize=11)
    ax.set_title(
        f"{dataset.capitalize()} — Log-Likelihood Comparison ({n_seeds} seeds, mean ± std)",
        fontsize=12, fontweight="bold", pad=10
    )
    
    ax.legend(fontsize=9, framealpha=0.95,
               loc="upper center", bbox_to_anchor=(0.5, -0.15),
               ncol=4, borderaxespad=0)
               
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_ll_bars.py path/to/file1.json ...")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

        data = load(path)
        dataset = data["dataset"]

        out = path.parent / f"{dataset}_ll_bars.png"
        plot_bars(data, out)


if __name__ == "__main__":
    main()
