
import os
import importlib
from pathlib import Path


def _import_plot_l1l2():
    """Import load/plot_dots from plot_l1l2_dots (project root)."""
    spec = importlib.util.spec_from_file_location(
        "plot_l1l2_dots",
        os.path.join(os.path.dirname(__file__), "..", "plot_l1l2_dots.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_plot_ll_bars():
    spec = importlib.util.spec_from_file_location(
        "plot_ll_bars",
        os.path.join(os.path.dirname(__file__), "..", "plot_ll_bars.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_plot_frz_uf():
    spec = importlib.util.spec_from_file_location(
        "plot_frozen_vs_unfrozen",
        os.path.join(os.path.dirname(__file__), "..", "plot_frozen_vs_unfrozen.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_plot_query_results():
    spec = importlib.util.spec_from_file_location(
        "plot_query_results",
        os.path.join(os.path.dirname(__file__), "..", "plot_query_results.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_experiment_dir(save_dir: str, dataset: str, method_tag: str,
                        min_inst: int, threshold: float, seeds: list) -> str:
    """
    Create and return the experiment directory:
        <save_dir>/<dataset>_<method_tag>_min<mi>_pth<th>_seeds<s1>-<s2>-…/
    Also creates the plots/ subfolder.
    """
    seed_str = "-".join(str(s) for s in sorted(seeds))
    folder_name = (
        f"{dataset}_{method_tag}"
        f"_min{min_inst}_pth{threshold}"
        f"_seeds{seed_str}"
    )
    exp_dir = os.path.join(save_dir, folder_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir


def generate_einet_plots(json_path: str, dataset: str):
    """
    Called after run_einet_benchmarks finishes writing results.
    Generates plots into a plots/ subfolder next to the JSON.
    """
    json_path = Path(json_path)
    plot_dir = json_path.parent / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n Generating plots into: {plot_dir}")

    try:
        m = _import_plot_l1l2()
        data = m.load(json_path)
        for metric in ["l1", "l2"]:
            out = os.path.join(plot_dir, f"{dataset}_{metric}_dot.png")
            m.plot_dots(data, metric, out)
    except Exception as e:
        print(f"  [auto_plot] L1/L2 dot plots failed: {e}")

    try:
        m = _import_plot_ll_bars()
        data = m.load(json_path)
        out = os.path.join(plot_dir, f"{dataset}_ll_bars.png")
        m.plot_bars(data, out)
    except Exception as e:
        print(f"  [auto_plot] LL bar plot failed: {e}")

    try:
        m = _import_plot_frz_uf()
        for metric in ["l1", "ll"]:
            data = m.load_data(json_path, metric=metric)
            out = os.path.join(plot_dir, f"{dataset}_frz_vs_ufrz_{metric}_bars.png")
            m.plot_frz_uf(data, out)
    except Exception as e:
        print(f"  [auto_plot] Frozen vs Unfrozen plots failed: {e}")

    try:
        m = _import_plot_query_results()
        for metric in ["l1", "l2"]:
            m.plot_query_results(str(json_path), dataset, metric=metric, out_dir=str(plot_dir))
    except Exception as e:
        print(f"  [auto_plot] Query plots failed: {e}")

    print(f"── Plots saved to {plot_dir}\n")
    return str(plot_dir)


def generate_spn_plots(json_path: str, dataset: str):
    """
    Called after run_benchmarks finishes writing results.
    Generates plots into a plots/ subfolder next to the JSON.
    """
    json_path = Path(json_path)
    plot_dir = json_path.parent / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n── Auto-generating SPN plots into: {plot_dir}")

    try:
        m = _import_plot_l1l2()
        data = m.load(json_path)
        for metric in ["l1", "l2"]:
            out = os.path.join(plot_dir, f"{dataset}_{metric}_dot.png")
            m.plot_dots(data, metric, out)
    except Exception as e:
        print(f"  [auto_plot] L1/L2 dot plots failed: {e}")

    try:
        m = _import_plot_ll_bars()
        data = m.load(json_path)
        out = os.path.join(plot_dir, f"{dataset}_ll_bars.png")
        m.plot_bars(data, out)
    except Exception as e:
        print(f"  [auto_plot] LL bar plot failed: {e}")

    print(f"── SPN plots saved to {plot_dir}\n")
    return str(plot_dir)
