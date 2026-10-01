

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict


C_NAIVE   = "#D65F5F"   # Red        — Naive Baseline
C_CONT    = "#4878CF"   # Blue       — Continual SPN
C_NAT_F   = "#E5AE38"   # Gold       — RAT-SPN Native Frozen
C_NAT_U0  = "#B39DDB"   # Light Purple — RAT-SPN Native Unfrozen (noise=0.0)
C_NAT_U1  = "#7E57C2"   # Medium Purple — RAT-SPN Native Unfrozen (noise=0.01)
C_NAT_U2  = "#512DA8"   # Dark Purple  — RAT-SPN Native Unfrozen (noise=0.1)
C_CONV    = "#6ACC64"   # Green      — RAT-SPN Converted
C_FULL    = "#8C613C"   # Brown      — Full Retrain reference


def _safe_mean(lst):
    return float(np.mean(lst)) if lst else np.nan

def _safe_std(lst):
    return float(np.std(lst, ddof=0)) if lst else np.nan


def load(path):
    with open(path) as f:
        raw = json.load(f)

    keys = [
        "naive_l1", "naive_l2",
        "cont_l1",  "cont_l2",
        "nat_m_l1", "nat_m_l2",
        "nat_o_l1", "nat_o_l2",
        "conv_m_l1","conv_m_l2",
        "conv_o_l1","conv_o_l2",
        "nat_uf0_l1","nat_uf0_l2",       # unfrozen noise=0.0
        "nat_uf001_l1","nat_uf001_l2",   # unfrozen noise=0.01
        "nat_uf01_l1","nat_uf01_l2",     # unfrozen noise=0.1
        "full_l1",  "full_l2",
    ]
    buckets = defaultdict(lambda: {k: [] for k in keys})

    for row in raw:
        v = row["target_variable"]
        buckets[v]["naive_l1"].append(row.get("naive_l1", row.get("baseline_l1", np.nan)))
        buckets[v]["naive_l2"].append(row.get("naive_l2", row.get("baseline_l2", np.nan)))
        buckets[v]["cont_l1"].append(row.get("continual_l1", np.nan))
        buckets[v]["cont_l2"].append(row.get("continual_l2", np.nan))
        buckets[v]["nat_m_l1"].append(row.get("rat_native_matched_l1", np.nan))
        buckets[v]["nat_m_l2"].append(row.get("rat_native_matched_l2", np.nan))
        buckets[v]["nat_o_l1"].append(row.get("rat_native_overfit_l1", np.nan))
        buckets[v]["nat_o_l2"].append(row.get("rat_native_overfit_l2", np.nan))
        buckets[v]["conv_m_l1"].append(row.get("rat_conv_matched_l1", np.nan))
        buckets[v]["conv_m_l2"].append(row.get("rat_conv_matched_l2", np.nan))
        buckets[v]["conv_o_l1"].append(row.get("rat_conv_overfit_l1", np.nan))
        buckets[v]["conv_o_l2"].append(row.get("rat_conv_overfit_l2", np.nan))
        # Unfrozen noise variants (may not be present yet)
        buckets[v]["nat_uf0_l1"].append(row.get("rat_native_unfrozen_noise0.0_l1", np.nan))
        buckets[v]["nat_uf0_l2"].append(row.get("rat_native_unfrozen_noise0.0_l2", np.nan))
        buckets[v]["nat_uf001_l1"].append(row.get("rat_native_unfrozen_noise0.01_l1", np.nan))
        buckets[v]["nat_uf001_l2"].append(row.get("rat_native_unfrozen_noise0.01_l2", np.nan))
        buckets[v]["nat_uf01_l1"].append(row.get("rat_native_unfrozen_noise0.1_l1", np.nan))
        buckets[v]["nat_uf01_l2"].append(row.get("rat_native_unfrozen_noise0.1_l2", np.nan))
        buckets[v]["full_l1"].append(row.get("full_l1", np.nan))
        buckets[v]["full_l2"].append(row.get("full_l2", np.nan))

    dataset = raw[0]["dataset"]
    n_seeds = len(set(r["seed"] for r in raw))
    vars_   = sorted(buckets.keys())

    # Build var_id -> name mapping
    var_id_to_name = {}
    for row in raw:
        vid = row["target_variable"]
        vname = row.get("target_variable_name")
        if vname and vid not in var_id_to_name:
            var_id_to_name[vid] = str(vname)

    def arr_mean(key):
        return np.array([_safe_mean(buckets[v][key]) for v in vars_])
    def arr_std(key):
        return np.array([_safe_std(buckets[v][key]) for v in vars_])
    def flat_mean(key):
        vals = [x for v in vars_ for x in buckets[v][key] if not np.isnan(x)]
        return float(np.mean(vals)) if vals else np.nan
    def flat_std(key):
        vals = [x for v in vars_ for x in buckets[v][key] if not np.isnan(x)]
        return float(np.std(vals, ddof=0)) if vals else np.nan

    data = {
        "dataset": dataset,
        "n_seeds": n_seeds,
        "vars":    [var_id_to_name.get(v, f"Var {v}") for v in vars_],
        # per-variable arrays
        "naive_l1_m":   arr_mean("naive_l1"),   "naive_l1_s":   arr_std("naive_l1"),
        "naive_l2_m":   arr_mean("naive_l2"),   "naive_l2_s":   arr_std("naive_l2"),
        "cont_l1_m":    arr_mean("cont_l1"),    "cont_l1_s":    arr_std("cont_l1"),
        "cont_l2_m":    arr_mean("cont_l2"),    "cont_l2_s":    arr_std("cont_l2"),
        "nat_m_l1_m":   arr_mean("nat_m_l1"),   "nat_m_l1_s":   arr_std("nat_m_l1"),
        "nat_m_l2_m":   arr_mean("nat_m_l2"),   "nat_m_l2_s":   arr_std("nat_m_l2"),
        "nat_o_l1_m":   arr_mean("nat_o_l1"),   "nat_o_l1_s":   arr_std("nat_o_l1"),
        "nat_o_l2_m":   arr_mean("nat_o_l2"),   "nat_o_l2_s":   arr_std("nat_o_l2"),
        "conv_m_l1_m":  arr_mean("conv_m_l1"),  "conv_m_l1_s":  arr_std("conv_m_l1"),
        "conv_m_l2_m":  arr_mean("conv_m_l2"),  "conv_m_l2_s":  arr_std("conv_m_l2"),
        "conv_o_l1_m":  arr_mean("conv_o_l1"),  "conv_o_l1_s":  arr_std("conv_o_l1"),
        "conv_o_l2_m":  arr_mean("conv_o_l2"),  "conv_o_l2_s":  arr_std("conv_o_l2"),
        "nat_uf0_l1_m":  arr_mean("nat_uf0_l1"),  "nat_uf0_l1_s":  arr_std("nat_uf0_l1"),
        "nat_uf0_l2_m":  arr_mean("nat_uf0_l2"),  "nat_uf0_l2_s":  arr_std("nat_uf0_l2"),
        "nat_uf001_l1_m":arr_mean("nat_uf001_l1"),"nat_uf001_l1_s":arr_std("nat_uf001_l1"),
        "nat_uf001_l2_m":arr_mean("nat_uf001_l2"),"nat_uf001_l2_s":arr_std("nat_uf001_l2"),
        "nat_uf01_l1_m": arr_mean("nat_uf01_l1"), "nat_uf01_l1_s": arr_std("nat_uf01_l1"),
        "nat_uf01_l2_m": arr_mean("nat_uf01_l2"), "nat_uf01_l2_s": arr_std("nat_uf01_l2"),
        
        # global flat stats for the Avg column
        "naive_l1_gf": flat_mean("naive_l1"),   "naive_l1_gs": flat_std("naive_l1"),
        "naive_l2_gf": flat_mean("naive_l2"),   "naive_l2_gs": flat_std("naive_l2"),
        "cont_l1_gf":  flat_mean("cont_l1"),    "cont_l1_gs":  flat_std("cont_l1"),
        "cont_l2_gf":  flat_mean("cont_l2"),    "cont_l2_gs":  flat_std("cont_l2"),
        "nat_m_l1_gf": flat_mean("nat_m_l1"),   "nat_m_l1_gs": flat_std("nat_m_l1"),
        "nat_m_l2_gf": flat_mean("nat_m_l2"),   "nat_m_l2_gs": flat_std("nat_m_l2"),
        "nat_o_l1_gf": flat_mean("nat_o_l1"),   "nat_o_l1_gs": flat_std("nat_o_l1"),
        "nat_o_l2_gf": flat_mean("nat_o_l2"),   "nat_o_l2_gs": flat_std("nat_o_l2"),
        "conv_m_l1_gf": flat_mean("conv_m_l1"), "conv_m_l1_gs": flat_std("conv_m_l1"),
        "conv_m_l2_gf": flat_mean("conv_m_l2"), "conv_m_l2_gs": flat_std("conv_m_l2"),
        "conv_o_l1_gf": flat_mean("conv_o_l1"), "conv_o_l1_gs": flat_std("conv_o_l1"),
        "conv_o_l2_gf": flat_mean("conv_o_l2"), "conv_o_l2_gs": flat_std("conv_o_l2"),
        "nat_uf0_l1_gf": flat_mean("nat_uf0_l1"), "nat_uf0_l1_gs": flat_std("nat_uf0_l1"),
        "nat_uf0_l2_gf": flat_mean("nat_uf0_l2"), "nat_uf0_l2_gs": flat_std("nat_uf0_l2"),
        "nat_uf001_l1_gf": flat_mean("nat_uf001_l1"), "nat_uf001_l1_gs": flat_std("nat_uf001_l1"),
        "nat_uf001_l2_gf": flat_mean("nat_uf001_l2"), "nat_uf001_l2_gs": flat_std("nat_uf001_l2"),
        "nat_uf01_l1_gf": flat_mean("nat_uf01_l1"), "nat_uf01_l1_gs": flat_std("nat_uf01_l1"),
        "nat_uf01_l2_gf": flat_mean("nat_uf01_l2"), "nat_uf01_l2_gs": flat_std("nat_uf01_l2"),

        # single scalars for full retrain
        "full_l1_v": flat_mean("full_l1"),
        "full_l2_v": flat_mean("full_l2"),
        # flags
        "has_einet": not np.all(np.isnan(arr_mean("nat_m_l1"))),
        "has_unfrozen": not np.all(np.isnan(arr_mean("nat_uf0_l1"))),
    }
    return data


def _has_data(arr):
    return not np.all(np.isnan(arr))


def plot_dots(data, metric, out_path):
    m = metric  # "l1" or "l2"
    dataset = data["dataset"]
    n_seeds = data["n_seeds"]
    vars_   = data["vars"]
    nv      = len(vars_)
    x_var   = np.arange(nv)           # per-variable positions
    x_avg   = nv + 0.8                 # "Avg" column position 

    # Gather series: (mean_array, std_array, label, color, marker, linestyle)
    # Only include series that actually have data
    series = []

    naive_m = data[f"naive_{m}_m"];  naive_s = data[f"naive_{m}_s"]
    cont_m  = data[f"cont_{m}_m"];   cont_s  = data[f"cont_{m}_s"]
    naive_gf = data[f"naive_{m}_gf"]; naive_gs = data[f"naive_{m}_gs"]
    cont_gf  = data[f"cont_{m}_gf"];  cont_gs  = data[f"cont_{m}_gs"]

    series.append((naive_m, naive_s, "Naive Baseline",         C_NAIVE, "o", "solid", naive_gf, naive_gs))
    series.append((cont_m,  cont_s,  "Continual SPN",          C_CONT,  "o", "solid", cont_gf, cont_gs))

    if data["has_einet"]:
        nat_m_m  = data[f"nat_m_{m}_m"];  nat_m_s  = data[f"nat_m_{m}_s"]
        nat_o_m  = data[f"nat_o_{m}_m"];  nat_o_s  = data[f"nat_o_{m}_s"]
        conv_m_m = data[f"conv_m_{m}_m"]; conv_m_s = data[f"conv_m_{m}_s"]
        conv_o_m = data[f"conv_o_{m}_m"]; conv_o_s = data[f"conv_o_{m}_s"]
        
        nat_m_gf  = data[f"nat_m_{m}_gf"];  nat_m_gs  = data[f"nat_m_{m}_gs"]
        nat_o_gf  = data[f"nat_o_{m}_gf"];  nat_o_gs  = data[f"nat_o_{m}_gs"]
        conv_m_gf = data[f"conv_m_{m}_gf"]; conv_m_gs = data[f"conv_m_{m}_gs"]
        conv_o_gf = data[f"conv_o_{m}_gf"]; conv_o_gs = data[f"conv_o_{m}_gs"]

        series.append((nat_m_m,  nat_m_s,  "RAT-SPN Native Frz (M)", C_NAT_F, "o", "solid", nat_m_gf, nat_m_gs))
        series.append((nat_o_m,  nat_o_s,  "RAT-SPN Native Frz (O)", C_NAT_F, "X", "solid", nat_o_gf, nat_o_gs))
        series.append((conv_m_m, conv_m_s, "RAT-SPN Conv (M)",       C_CONV,  "o", "solid", conv_m_gf, conv_m_gs))
        series.append((conv_o_m, conv_o_s, "RAT-SPN Conv (O)",       C_CONV,  "X", "solid", conv_o_gf, conv_o_gs))

    if data["has_unfrozen"]:
        unfrozen_configs = [
            ("nat_uf0",   "σ=0",    C_NAT_U0, "D"),
            ("nat_uf001", "σ=0.01", C_NAT_U1, "s"),
            ("nat_uf01",  "σ=0.1",  C_NAT_U2, "^"),
        ]
        for tag, noise_label, color, marker in unfrozen_configs:
            um = data.get(f"{tag}_{m}_m"); us = data.get(f"{tag}_{m}_s")
            ugf = data.get(f"{tag}_{m}_gf"); ugs = data.get(f"{tag}_{m}_gs")
            if um is not None and _has_data(um):
                series.append((um, us, f"RAT-SPN Unfrz ({noise_label})", color, marker, "solid", ugf, ugs))

    n_series = len(series)
    # Spread offsets symmetrically
    if n_series > 1:
        span    = min(0.85, n_series * 0.14)
        offsets = np.linspace(-span / 2, span / 2, n_series)
    else:
        offsets = [0.0]

  
    fig, ax = plt.subplots(figsize=(max(12, nv * 2.2 + 3), 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    err_kw = dict(linewidth=0, elinewidth=1.6, capsize=5, capthick=1.6)

    avg_means = []  # track avg values for the rightmost column

    for (arr_m, arr_s, label, color, marker, _, gf, gs), offset in zip(series, offsets):
        # Per-variable dots
        ax.errorbar(
            x_var + offset, arr_m, yerr=arr_s,
            fmt=marker, color=color, markersize=8,
            label=label, zorder=4, **err_kw
        )
        # Avg column dot
        if gf is not None and not np.isnan(gf):
            ax.errorbar(
                x_avg + offset, gf, yerr=gs,
                fmt=marker, color=color, markersize=9,
                zorder=4, **err_kw
            )
            avg_means.append(gf)

    # Full-retrain horizontal line
    full_v = data[f"full_{m}_v"]
    if full_v and not np.isnan(full_v):
        ax.axhline(full_v, color=C_FULL, linestyle="--", linewidth=2,
                   label=f"Full Retrain ({full_v:.2e})", zorder=2)

    # Vertical separator before Avg column
    sep_x = nv + 0.35
    ax.axvline(sep_x, color="#AAAAAA", linestyle=":", linewidth=1.2, zorder=1)

    # X-axis ticks: variable labels + "Avg"
    all_ticks = list(x_var) + [x_avg]
    all_labels = list(vars_) + ["Avg"]
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=10)
    
    # Add extra horizontal padding so edges don't get clipped
    ax.set_xlim(-0.9, x_avg + 0.9)

    # Use log scale (powers of 10) for better visibility of small differences
    ax.set_yscale("log")
    ax.set_xlabel("Target Variable", fontsize=11)
    ax.set_ylabel(f"{m.upper()} Distance to BN Ground Truth (log scale)", fontsize=11)
    ax.set_title(
        f"{dataset.capitalize()} — {m.upper()} Distance  ({n_seeds} seeds, mean ± std)",
        fontsize=12, fontweight="bold", pad=15,
    )

    # Move legend further down and explicitly add bottom margin
    ax.legend(
        fontsize=9, framealpha=0.95,
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=3, borderaxespad=0,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")

    # allocate space for the legend at the bottom
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close()
    print(f"Saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_l1l2_dots.py path/to/file1.json ...")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

        data = load(path)
        dataset = data["dataset"]

        for metric in ["l1", "l2"]:
            out = path.parent / f"{dataset}_{metric}_dot.png"
            plot_dots(data, metric, out)


if __name__ == "__main__":
    main()