import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import numbers

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# disable MathJax loading in interactive contexts
pio.mathjax = ""


def extract_accuracy(entry: Dict[str, float]) -> float:
    return entry.get(
        "acc,none",
        entry.get(
            "exact_match,strict-match",
            entry.get("bleu,none", np.nan),
        ),
    )


def extract_perplexity(entry: Dict[str, float], metric_key: str = "perplexity,none") -> float:
    return entry.get(metric_key, np.nan)


def find_result_files(results_dir: Path, experiment_name: str) -> List[Path]:
    pattern = f"{experiment_name}_seed*_results.json"

    def seed_sort_key(path: Path) -> int:
        match = re.search(r"_seed(\\d+)_results$", path.stem)
        if match:
            return int(match.group(1))
        return int(1e9)

    candidates = sorted(results_dir.glob(pattern), key=seed_sort_key)
    if candidates:
        return candidates

    fallback = results_dir / f"{experiment_name}_results.json"
    if fallback.exists():
        return [fallback]
    return []


def load_seed_result(path: Path) -> Dict[str, object]:
    with open(path, "r") as f:
        data = json.load(f)

    comp_keys = sorted(data.keys(), key=lambda k: float(k))
    cleaned = {}
    compute = []
    compute_map: Dict[str, float] = {}
    for key in comp_keys:
        entry = data[key]
        entry = entry.copy()
        pct = entry.pop("percent_tokens_skipped", None)
        if pct is None:
            pct = float(key)
        compute_val = pct * 100
        compute.append(compute_val)
        compute_map[key] = compute_val
        cleaned[key] = entry

    return {
        "path": path,
        "keys": comp_keys,
        "compute": np.array(compute, dtype=float),
        "compute_map": compute_map,
        "data": cleaned,
    }


def merge_compute_keys(seed_results: List[Dict[str, object]]) -> List[str]:
    unique = set()
    for seed in seed_results:
        unique.update(seed["keys"])
    return sorted(unique, key=lambda k: float(k))


def build_metric_tensor(
    seed_results: List[Dict[str, object]],
    comp_keys: Iterable[str],
    benchmark: str,
    extractor: Callable[[Dict[str, float]], float],
) -> np.ndarray:
    rows = []
    for seed in seed_results:
        row = []
        seed_data = seed["data"]
        for key in comp_keys:
            entry = seed_data.get(key)
            if entry is None:
                row.append(np.nan)
                continue
            bench_entry = entry.get(benchmark)
            if not isinstance(bench_entry, dict):
                row.append(np.nan)
                continue
            row.append(extractor(bench_entry))
        rows.append(row)
    return np.array(rows, dtype=float)


def build_compute_matrix(
    seed_results: List[Dict[str, object]], comp_keys: Iterable[str]
) -> np.ndarray:
    comp_keys = list(comp_keys)
    idx_map = {key: idx for idx, key in enumerate(comp_keys)}
    matrix = np.full((len(seed_results), len(comp_keys)), np.nan, dtype=float)
    for seed_idx, seed in enumerate(seed_results):
        for key, value in seed["compute_map"].items():
            matrix[seed_idx, idx_map[key]] = value
    return matrix


def interpolate_tensor(
    tensor: np.ndarray, compute_arrays: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    per_seed = []
    for seed_idx in range(tensor.shape[0]):
        values = tensor[seed_idx]
        compute = compute_arrays[seed_idx]
        mask = ~np.isnan(values) & ~np.isnan(compute)
        if mask.sum() < 2:
            per_seed.append(np.full_like(targets, np.nan, dtype=float))
            continue
        valid_values = values[mask]
        valid_compute = compute[mask]
        edge_left = valid_values[0]
        edge_right = valid_values[-1]
        per_seed.append(
            np.interp(
                targets,
                valid_compute,
                valid_values,
                left=edge_left,
                right=edge_right,
            )
        )
    return np.array(per_seed, dtype=float)


def format_mean_std(mean_arr: Iterable[float], std_arr: Iterable[float], precision: int = 2) -> List[str]:
    formatted = []
    for mean, std in zip(mean_arr, std_arr):
        if np.isnan(mean) and np.isnan(std):
            formatted.append("nan")
        else:
            formatted.append(f"{mean:.{precision}f}±{std:.{precision}f}")
    return formatted


def nanmean(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)


def nanstd(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanstd(arr, axis=axis)


def compute_saved_at_retained(
    acc: np.ndarray, compute: np.ndarray, target_ratio: float = 0.90
) -> float:
    """
    Find the rightmost compute-saved percentage where accuracy >= target_ratio * base accuracy.
    Falls back to linear interpolation if none meet the threshold.
    """
    mask = ~np.isnan(acc) & ~np.isnan(compute)
    if mask.sum() == 0:
        return np.nan

    acc = acc[mask]
    compute = compute[mask]

    if len(acc) == 0:
        return np.nan

    base_acc = acc[0]
    if np.isnan(base_acc):
        return np.nan

    target_acc = base_acc * target_ratio
    mask = acc >= target_acc
    if mask.any():
        return float(np.max(compute[mask]))
    if len(acc) < 2:
        return np.nan
    # fallback interpolation
    try:
        return float(np.interp(target_acc, acc[::-1], compute[::-1]))
    except ValueError:
        return np.nan


def compute_saved_stats(
    metric_tensor: np.ndarray,
    compute_arrays: np.ndarray,
    target_ratio: float = 0.90,
) -> tuple[float, float]:
    per_seed = []
    for idx in range(metric_tensor.shape[0]):
        value = compute_saved_at_retained(
            metric_tensor[idx], compute_arrays[idx], target_ratio=target_ratio
        )
        if not np.isnan(value):
            per_seed.append(value)
    if not per_seed:
        return np.nan, np.nan
    per_seed = np.array(per_seed, dtype=float)
    return float(np.mean(per_seed)), float(np.std(per_seed))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment JSON and summarize results",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment name used when launching jobs (e.g. GateSkip-separate-vector_cot)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing *_results.json files (defaults to $BASE_CACHE_DIR/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for output artifacts (defaults to <results-dir>/<experiment-name>)",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        base_cache = Path(os.environ.get("BASE_CACHE_DIR", "."))
        results_dir = base_cache / "results"
    else:
        results_dir = args.results_dir
    results_dir = results_dir.expanduser().resolve()

    files = find_result_files(results_dir, args.experiment_name)
    if not files:
        raise FileNotFoundError(
            f"No results found for {args.experiment_name} in {results_dir}"
        )

    seed_results = [load_seed_result(path) for path in files]
    comp_keys = merge_compute_keys(seed_results)
    compute_matrix = build_compute_matrix(seed_results, comp_keys)
    compute_mean = nanmean(compute_matrix, axis=0)
    compute_std = nanstd(compute_matrix, axis=0)

    print(
        f"Loaded {len(files)} seed result files for {args.experiment_name}:"
        f" {', '.join(path.name for path in files)}"
    )

    sample_entry = None
    sample_key = None
    sample_seed_idx = None
    for seed_idx, seed in enumerate(seed_results):
        for key in comp_keys:
            candidate = seed["data"].get(key)
            if isinstance(candidate, dict) and candidate:
                sample_entry = candidate
                sample_key = key
                sample_seed_idx = seed_idx
                break
        if sample_entry:
            break
    if sample_entry is None:
        raise ValueError("Could not locate any benchmark entries in the results")
    benchmarks = [
        bm
        for bm, val in sample_entry.items()
        if isinstance(val, dict)
        if not (bm.startswith("mmlu_") and bm not in ["mmlu_stem", "mmlu_gen"])
    ]

    profile_metrics: List[str] = []
    profile_metric_set = set()
    for seed in seed_results:
        for entry in seed["data"].values():
            profile = entry.get("vllm_profile")
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if isinstance(value, numbers.Number):
                        profile_metric_set.add(key)
    if profile_metric_set:
        profile_metrics = sorted(profile_metric_set)
        benchmarks = [bm for bm in benchmarks if bm != "vllm_profile"]

    out_dir = args.output_dir or (results_dir / args.experiment_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    accuracy_tensors: Dict[str, np.ndarray] = {}
    perplexity_tensors: Dict[str, np.ndarray] = {}
    for bm in benchmarks:
        accuracy_tensors[bm] = build_metric_tensor(
            seed_results, comp_keys, bm, extract_accuracy
        )
        perplexity_tensors[bm] = build_metric_tensor(
            seed_results, comp_keys, bm, extract_perplexity
        )

    compute_labels = [
        f"{mean:.2f}%±{std:.2f}%" for mean, std in zip(compute_mean, compute_std)
    ]

    # 1) Compute Saved @90%
    saved_rows = {}
    for bm in benchmarks:
        mean_saved, std_saved = compute_saved_stats(
            accuracy_tensors[bm], compute_matrix, target_ratio=0.90
        )
        saved_rows[bm] = {
            "Compute Saved Mean (%)": mean_saved,
            "Compute Saved Std (%)": std_saved,
        }
    df_saved = pd.DataFrame.from_dict(saved_rows, orient="index")
    df_saved.index.name = "Benchmark"
    df_saved.to_csv(Path(out_dir) / "saved_at_90.csv")
    df_saved_print = pd.DataFrame(index=df_saved.index)
    df_saved_print["Compute Saved @90%"] = format_mean_std(
        df_saved["Compute Saved Mean (%)"], df_saved["Compute Saved Std (%)"]
    )
    print("=== Compute Saved @90% ===")
    print(df_saved_print)

    # 2) Accuracy @0% compute omitted
    acc0_rows = {}
    for bm in benchmarks:
        acc_mean = nanmean(accuracy_tensors[bm], axis=0)[0] * 100
        acc_std = nanstd(accuracy_tensors[bm], axis=0)[0] * 100
        acc0_rows[bm] = {"Accuracy Mean (%)": acc_mean, "Accuracy Std (%)": acc_std}
    df_acc0 = pd.DataFrame.from_dict(acc0_rows, orient="index")
    df_acc0.index.name = "Benchmark"
    df_acc0.to_csv(Path(out_dir) / "accuracy_at_0.csv")
    df_acc0_print = pd.DataFrame(index=df_acc0.index)
    df_acc0_print["Accuracy @0%"] = format_mean_std(
        df_acc0["Accuracy Mean (%)"], df_acc0["Accuracy Std (%)"]
    )
    print("\n=== Accuracy @0% Compute Omitted ===")
    print(df_acc0_print)

    # 3) Raw results table with acc±std (percent with two decimals) sorted by compute saved
    idx_labels = compute_labels
    raw_numeric = pd.DataFrame(index=idx_labels)
    raw_numeric.index.name = "Compute Saved (%)"
    raw_display = pd.DataFrame(index=idx_labels)
    raw_display.index.name = "Compute Saved (%)"

    for bm in benchmarks:
        acc_mean = nanmean(accuracy_tensors[bm], axis=0) * 100
        acc_std = nanstd(accuracy_tensors[bm], axis=0) * 100
        raw_numeric[f"{bm} Mean"] = acc_mean
        raw_numeric[f"{bm} Std"] = acc_std
        raw_display[bm] = format_mean_std(acc_mean, acc_std)

        ppl_tensor = perplexity_tensors[bm]
        if not np.isnan(ppl_tensor).all():
            ppl_mean = nanmean(ppl_tensor, axis=0)
            ppl_std = nanstd(ppl_tensor, axis=0)
            raw_numeric[f"{bm} Perplexity Mean"] = ppl_mean
            raw_numeric[f"{bm} Perplexity Std"] = ppl_std
            raw_display[f"{bm} Perplexity"] = format_mean_std(
                ppl_mean, ppl_std
            )

    raw_numeric.to_csv(Path(out_dir) / "raw_results.csv")
    print("\n=== Raw Results (sorted) ===")
    print(raw_display)

    profile_display = None
    if profile_metrics:
        profile_numeric = pd.DataFrame(index=idx_labels)
        profile_numeric.index.name = "Compute Saved (%)"
        profile_display = pd.DataFrame(index=idx_labels)
        profile_display.index.name = "Compute Saved (%)"

        def make_profile_extractor(metric_name: str) -> Callable[[Dict[str, float]], float]:
            def _extract(entry: Dict[str, float]) -> float:
                value = entry.get(metric_name, np.nan)
                return float(value) if isinstance(value, numbers.Number) else np.nan

            return _extract

        for metric in profile_metrics:
            tensor = build_metric_tensor(
                seed_results,
                comp_keys,
                "vllm_profile",
                make_profile_extractor(metric),
            )
            metric_mean = nanmean(tensor, axis=0)
            metric_std = nanstd(tensor, axis=0)
            profile_numeric[f"{metric} Mean"] = metric_mean
            profile_numeric[f"{metric} Std"] = metric_std
            profile_display[metric] = format_mean_std(metric_mean, metric_std, precision=3)

        profile_numeric.to_csv(Path(out_dir) / "vllm_profile_metrics.csv")
        print("\n=== vLLM Profile Metrics ===")
        print(profile_display)

    # 4) Interpolated accuracies + average across benchmarks
    targets = np.array([0, 5, 10, 15, 20, 25, 30, 45, 60], dtype=float)
    target_labels = [f"{t:.0f}%" for t in targets]
    df_interp = pd.DataFrame(index=target_labels)
    df_interp.index.name = "Compute Saved (%)"
    df_interp_print = pd.DataFrame(index=target_labels)
    df_interp_print.index.name = "Compute Saved (%)"

    interp_cache = {}
    ppl_interp_cache = {}

    for bm in benchmarks:
        seed_interp = interpolate_tensor(
            accuracy_tensors[bm] * 100, compute_matrix, targets
        )
        interp_cache[bm] = seed_interp
        mean_interp = nanmean(seed_interp, axis=0)
        std_interp = nanstd(seed_interp, axis=0)
        df_interp[f"{bm} Mean"] = mean_interp
        df_interp[f"{bm} Std"] = std_interp
        df_interp_print[bm] = format_mean_std(mean_interp, std_interp)

        ppl_tensor = perplexity_tensors[bm]
        if not np.isnan(ppl_tensor).all():
            ppl_interp = interpolate_tensor(ppl_tensor, compute_matrix, targets)
            ppl_interp_cache[bm] = ppl_interp
            ppl_mean = nanmean(ppl_interp, axis=0)
            ppl_std = nanstd(ppl_interp, axis=0)
            df_interp[f"{bm} PPL Mean"] = ppl_mean
            df_interp[f"{bm} PPL Std"] = ppl_std
            df_interp_print[f"{bm} PPL"] = format_mean_std(ppl_mean, ppl_std)

    if interp_cache:
        stack = np.stack(list(interp_cache.values()), axis=0)
        avg_seed = nanmean(stack, axis=0)
        avg_mean = nanmean(avg_seed, axis=0)
        avg_std = nanstd(avg_seed, axis=0)
        df_interp["Average Mean"] = avg_mean
        df_interp["Average Std"] = avg_std
        df_interp_print["Average"] = format_mean_std(avg_mean, avg_std)

    if ppl_interp_cache:
        ppl_stack = np.stack(list(ppl_interp_cache.values()), axis=0)
        ppl_avg_seed = nanmean(ppl_stack, axis=0)
        ppl_avg_mean = nanmean(ppl_avg_seed, axis=0)
        ppl_avg_std = nanstd(ppl_avg_seed, axis=0)
        df_interp["PPL Average Mean"] = ppl_avg_mean
        df_interp["PPL Average Std"] = ppl_avg_std
        df_interp_print["PPL Average"] = format_mean_std(
            ppl_avg_mean, ppl_avg_std
        )

    df_interp.to_csv(Path(out_dir) / "interpolated_accuracies.csv")
    print("\n=== Interpolated Accuracies + Average ===")
    print(df_interp_print)

    summary = df_saved.join(df_acc0)
    summary_print = pd.DataFrame(index=summary.index)
    summary_print["Compute Saved @90%"] = df_saved_print["Compute Saved @90%"]
    summary_print["Accuracy @0%"] = df_acc0_print["Accuracy @0%"]

    print("\n% LaTeX: Summary table")
    print(summary_print.to_latex(escape=False))
    print("\n% LaTeX: Raw results table (sorted)")
    print(raw_display.to_latex(escape=False))
    if profile_display is not None:
        print("\n% LaTeX: vLLM profile metrics")
        print(profile_display.to_latex(escape=False))
    print("\n% LaTeX: Interpolated Accuracies + Average")
    print(df_interp_print.to_latex(escape=False))

    order = np.argsort(compute_mean)
    x = compute_mean[order]
    for bm in benchmarks:
        acc_mean = nanmean(accuracy_tensors[bm], axis=0) * 100
        acc_std = nanstd(accuracy_tensors[bm], axis=0) * 100
        y = acc_mean[order]
        y_std = acc_std[order]
        upper = y + y_std
        lower = np.maximum(y - y_std, 0)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line_shape="spline",
                name=bm,
            )
        )

        yaxis_label = "Accuracy"
        sample_metrics = {}
        if sample_seed_idx is not None and sample_key is not None:
            seed_data = seed_results[sample_seed_idx]["data"].get(sample_key, {})
            if isinstance(seed_data, dict):
                sample_metrics = seed_data.get(bm, {})
        if isinstance(sample_metrics, dict) and "bleu,none" in sample_metrics:
            yaxis_label = "BLEU"

        fig.update_layout(
            xaxis_title="Compute Saved (%)",
            yaxis_title=f"{yaxis_label} (%)",
            template="simple_white",
        )
        fig.update_yaxes(rangemode="tozero")

        out_path = Path(out_dir) / f"{bm}_accuracy_vs_saved.pdf"
        fig.write_image(str(out_path), engine="kaleido")
        time.sleep(2)
        fig.write_image(str(out_path), engine="kaleido")
    print(f"Plots written to '{out_dir}/' directory.")


if __name__ == "__main__":
    main()
