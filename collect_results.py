import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import time

# disable MathJax loading in interactive contexts
pio.mathjax = ""


def load_data(json_path: Path):
    """Load JSON experiment data, extract actual compute-saved percentages, and clean raw data."""
    with open(json_path, "r") as f:
        data = json.load(f)

    comp_keys = sorted(data.keys(), key=lambda k: float(k))
    compute = []
    for k in comp_keys:
        pct = data[k].pop("percent_tokens_skipped", None)
        if pct is None:
            pct = float(k)
        compute.append(pct * 100)
    return data, np.array(compute)


def extract_metric_series(
    data: dict, compute: np.ndarray, benchmark: str, metric_key: str = "acc,none"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract accuracy and stderr arrays for a given benchmark across thresholds.
    """
    acc_list, stderr_list = [], []
    for comp_frac in sorted(data.keys(), key=lambda k: float(k)):
        entry = data[comp_frac].get(benchmark, {})
        metric = entry.get(
            "acc,none",
            entry.get("exact_match,strict-match", entry.get("bleu,none", np.nan)),
        )
        acc_list.append(metric)
        std = entry.get(
            "acc_stderr,none",
            entry.get(
                "exact_match_stderr,strict-match", entry.get("bleu_stderr,none", np.nan)
            ),
        )
        stderr_list.append(std)
    return np.array(acc_list), np.array(stderr_list)


def extract_perplexity_series(
    data: dict, compute: np.ndarray, benchmark: str, metric_key: str = "perplexity,none"
) -> np.ndarray:
    """Extract perplexity values for a given benchmark across thresholds."""
    ppl_list = []
    for comp_frac in sorted(data.keys(), key=lambda k: float(k)):
        entry = data[comp_frac].get(benchmark, {})
        ppl = entry.get(metric_key, np.nan)
        ppl_list.append(ppl)
    return np.array(ppl_list)


def compute_saved_at_retained(
    acc: np.ndarray, compute: np.ndarray, target_ratio: float = 0.90
) -> float:
    """
    Find the rightmost compute-saved percentage where accuracy >= target_ratio * base accuracy.
    Falls back to linear interpolation if none meet the threshold.
    """
    base_acc = acc[0]
    target_acc = base_acc * target_ratio
    mask = acc >= target_acc
    if mask.any():
        return float(np.max(compute[mask]))
    # fallback interpolation
    return float(np.interp(target_acc, acc[::-1], compute[::-1]))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment JSON and summarize results"
    )
    parser.add_argument(
        "json_file", type=Path, help="Path to JSON file with experiment data"
    )
    args = parser.parse_args()

    data, compute = load_data(args.json_file)
    first_key = sorted(data.keys(), key=lambda k: float(k))[0]
    benchmarks = [
        bm
        for bm, val in data[first_key].items()
        if isinstance(val, dict) 
    ]

    out_dir = args.json_file.with_suffix("").name
    Path(out_dir).mkdir(exist_ok=True)

    # 1) Compute Saved @90%
    saved = {}
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        saved[bm] = compute_saved_at_retained(acc, compute, target_ratio=0.90)
    df_saved = pd.DataFrame.from_dict(
        saved, orient="index", columns=["Compute Saved @90%"]
    )
    df_saved.index.name = "Benchmark"
    df_saved.to_csv(Path(out_dir) / "saved_at_90.csv")
    print("=== Compute Saved @90% ===")
    print(df_saved)

    # 2) Accuracy @0% compute omitted (percent with two decimals)
    acc0 = {}
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        acc0[bm] = round(acc[0] * 100, 2)
    df_acc0 = pd.DataFrame.from_dict(acc0, orient="index", columns=["Accuracy @0%"])
    df_acc0.index.name = "Benchmark"
    df_acc0.to_csv(Path(out_dir) / "accuracy_at_0.csv")
    print("\n=== Accuracy @0% Compute Omitted ===")
    print(df_acc0)

    # 3) Raw results table with acc±stderr (percent with two decimals) sorted by compute saved
    # build raw DataFrame
    idx_labels = [f"{c:.2f}%" for c in compute]
    raw = pd.DataFrame(index=idx_labels)
    raw.index.name = "Compute Saved (%)"
    for bm in benchmarks:
        acc, stderr = extract_metric_series(data, compute, bm)
        raw[bm] = [f"{a*100:.2f}\u00B1{s*100:.2f}" for a, s in zip(acc, stderr)]
        ppl = extract_perplexity_series(data, compute, bm)
        if not np.isnan(ppl).all():
            ppl_values = [f"{p:.2f}" for p in ppl]
            insert_idx = raw.columns.get_loc(bm) + 1
            raw.insert(insert_idx, f"{bm} Perplexity", ppl_values)
    # sort rows by numeric compute-saved
    order = np.argsort(compute)
    raw = raw.iloc[order]
    raw.to_csv(Path(out_dir) / "raw_results.csv")
    print("\n=== Raw Results (sorted) ===")
    print(raw)

    # ——————————————————————————————————————————————————————————
    # 4) Interpolated accuracies + average across benchmarks
    targets = np.array([0, 5, 10, 15, 20, 25, 30, 45, 60])  # desired Compute Saved (%)
    interp_dict = {}

    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        acc_pct = acc * 100
        # linear interpolation over the sorted compute axis
        interp_acc = np.interp(targets, compute, acc_pct)
        interp_dict[bm] = interp_acc

    # build DataFrame (rows are the target compute‐saved %, cols are benchmarks)
    df_interp = pd.DataFrame(
        interp_dict,
        index=[f"{t:.0f}%" for t in targets],
    )
    df_interp.index.name = "Compute Saved (%)"

    # add the “Average” column
    df_interp["Average"] = df_interp.mean(axis=1)

    # write out and display
    df_interp.to_csv(Path(out_dir) / "interpolated_accuracies.csv")
    print("\n=== Interpolated Accuracies + Average ===")
    print(df_interp)

    # Print LaTeX
    summary = df_saved.join(df_acc0)
    print("\n% LaTeX: Summary table")
    print(summary.to_latex(float_format="%.2f"))
    print("\n% LaTeX: Raw results table (sorted)")
    print(raw.to_latex(escape=False))

    # 4) Plots: Accuracy vs Compute Saved
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        x = compute[order]
        y = acc[order] * 100
        fig = go.Figure(
            go.Scatter(x=x, y=y, mode="lines+markers", line_shape="spline", name=bm)
        )

        yaxis_label = "Accuracy"
        if "bleu,none" in list(list(data.values())[0].values())[0].keys():
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
