
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


METHODS = [
    # (result_key_prefix, label, color, marker)
    ("naive",                              "Naive Baseline",         "#e8503a", "o"),
    ("continual",                          "Continual SPN",          "#4878cf", "o"),
    ("rat_native_matched",                 "RAT-SPN Nat Frz (M)",    "#e8a628", "s"),
    ("rat_native_overfit",                 "RAT-SPN Nat Frz (O)",    "#e8a628", "X"),
    ("rat_conv_matched",                   "RAT-SPN Conv (M)",       "#6acc65", "s"),
    ("rat_conv_overfit",                   "RAT-SPN Conv (O)",       "#6acc65", "X"),
    ("rat_native_unfrozen_noise0.0",       "RAT-SPN Unfrz (σ=0)",   "#c69fde", "^"),
    ("rat_native_unfrozen_noise0.01",      "RAT-SPN Unfrz (σ=0.01)","#9467bd", "s"),
    ("rat_native_unfrozen_noise0.1",       "RAT-SPN Unfrz (σ=0.1)", "#4a3870", "v"),
]

FULL_RETRAIN_KEY = "full_spn"

# Query definitions: (subplot_title, key_suffix)
QUERIES = [
    ("Query A - Marginal of Target Variable  P(Y)",        "qA_marginal_target"),
    ("Query B - All Marginals  P(Xᵢ) for all i",          "qB_marginal_all"),
    ("Query C - Conditional  P(Y | X_e = x_e)",           "qC_conditional"),
]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_per_var(results, dataset, method_prefix, query_suffix, metric):
    """
    Returns dict: {var_id -> list of values across seeds}
    metric: "l1" or "l2"
    """
    key = f"{method_prefix}_{query_suffix}_{metric}"
    per_var = defaultdict(list)
    for rec in results:
        if rec["dataset"] != dataset:
            continue
        val = rec.get(key)
        if val is not None:
            per_var[rec["target_variable"]].append(val)
    return per_var


def aggregate(per_var):
    """Returns sorted (var_ids, means, stds)"""
    var_ids = sorted(per_var.keys())
    means = [np.mean(per_var[v]) for v in var_ids]
    stds  = [np.std(per_var[v])  for v in var_ids]
    return var_ids, means, stds


def compute_avg(per_var):
    """Mean and std across all variables and seeds (pooled)"""
    all_vals = [v for vals in per_var.values() for v in vals]
    if not all_vals:
        return None, None
    return float(np.mean(all_vals)), float(np.std(all_vals))


def get_full_retrain_val(results, dataset, query_suffix, metric):
    key = f"{FULL_RETRAIN_KEY}_{query_suffix}_{metric}"
    vals = [r.get(key) for r in results if r["dataset"] == dataset and r.get(key) is not None]
    if not vals:
        return None
    # same value repeated per (seed, var) - just average
    return float(np.mean(vals))


def plot_query_results(results_path, dataset, metric="l1", out_dir="."):
    results = load_results(results_path)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle(
        f"{dataset.capitalize()} — Query Evaluation L{metric[-1].upper()} Distance  (5 seeds, mean ± std)",
        fontsize=14, fontweight="bold", y=1.02
    )

    # Collect all target variable ids for x-axis layout
    var_ids_all = sorted({r["target_variable"] for r in results if r["dataset"] == dataset})
    n_vars = len(var_ids_all)

    # Build var_id - name mapping
    var_id_to_name = {}
    for r in results:
        if r["dataset"] == dataset:
            vid = r["target_variable"]
            vname = r.get("target_variable_name")
            if vname and vid not in var_id_to_name:
                var_id_to_name[vid] = str(vname)

    # x positions: each var gets a cluster, then a gap, then Avg
    x_positions = {v: i for i, v in enumerate(var_ids_all)}
    avg_x = n_vars + 0.8   # Avg column position

    offset_step = 0.06
    n_methods = len(METHODS)
    offsets = np.linspace(-(n_methods // 2) * offset_step,
                          (n_methods // 2) * offset_step, n_methods)

    legend_handles = []

    for ax_idx, (query_label, query_suffix) in enumerate(QUERIES):
        ax = axes[ax_idx]

        # Full retrain reference line
        full_val = get_full_retrain_val(results, dataset, query_suffix, metric)
        if full_val is not None:
            ax.axhline(full_val, color="#8B4513", linewidth=1.5,
                       linestyle="--", label=f"Full Retrain ({full_val:.2e})", zorder=1)

        # Vertical separator before Avg
        ax.axvline(avg_x - 0.45, color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.6, zorder=1)

        for m_idx, (mkey, mlabel, mcolor, mmarker) in enumerate(METHODS):
            per_var = extract_per_var(results, dataset, mkey, query_suffix, metric)
            if not per_var:
                continue

            var_ids, means, stds = aggregate(per_var)
            avg_mean, avg_std = compute_avg(per_var)

            xs       = [x_positions[v] + offsets[m_idx] for v in var_ids]
            xs_avg   = [avg_x + offsets[m_idx]]
            means_avg = [avg_mean] if avg_mean is not None else []
            stds_avg  = [avg_std]  if avg_std  is not None else []

            all_xs    = xs + xs_avg
            all_means = means + means_avg
            all_stds  = stds + stds_avg

            h = ax.errorbar(
                all_xs, all_means, yerr=all_stds,
                fmt=mmarker, color=mcolor, markersize=6,
                capsize=3, elinewidth=1, capthick=1,
                linewidth=0, label=mlabel, zorder=3
            )
            if ax_idx == 0:
                legend_handles.append(h)

        # X-axis ticks
        xtick_pos    = [x_positions[v] for v in var_ids_all] + [avg_x]
        xtick_labels = [var_id_to_name.get(v, f"Var {v}") for v in var_ids_all] + ["Avg"]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=9)

        ax.set_title(query_label, fontsize=10, pad=8)
        ax.set_xlabel("Target Variable", fontsize=9)
        ax.set_ylabel(f"L{metric[-1].upper()} Distance to BN Ground Truth", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_xlim(-0.6, avg_x + 0.6)

        # Add full retrain to legend on first subplot
        if ax_idx == 0 and full_val is not None:
            fr_patch = mpatches.Patch(
                color="#8B4513", label=f"Full Retrain ({full_val:.2e})"
            )
            legend_handles = [fr_patch] + legend_handles

    # Single legend below all subplots
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5, fontsize=8,
        frameon=True, framealpha=0.9
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_query_{metric}_dot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# CLI
def main():
    parser = argparse.ArgumentParser(description="Plot query evaluation results")
    parser.add_argument("--results", required=True,
                        help="Path to benchmark results JSON file")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. asia, child, insurance)")
    parser.add_argument("--metric", default="l1", choices=["l1", "l2"],
                        help="Distance metric to plot (default: l1)")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory for plots (default: current dir)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_query_results(args.results, args.dataset, args.metric, args.out_dir)


if __name__ == "__main__":
    main()