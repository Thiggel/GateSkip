#!/usr/bin/env python
import json
import argparse
from pathlib import Path
import numpy as np
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
    data: dict, compute: np.ndarray, benchmark: str
) -> np.ndarray:
    """
    Extract accuracy array for a given benchmark across thresholds.
    Defaults to 'acc,none' but falls back to exact_match or bleu if needed.
    Returns accuracies as a float array (not in %).
    """
    acc_list = []
    for comp_frac in sorted(data.keys(), key=lambda k: float(k)):
        entry = data[comp_frac].get(benchmark, {})
        metric = entry.get(
            "acc,none",
            entry.get("exact_match,strict-match", entry.get("bleu,none", np.nan)),
        )
        acc_list.append(metric)
    return np.array(acc_list)


def average_accuracy(data, compute, benchmarks):
    """
    Build a (num_benchmarks x num_thresholds) matrix of accuracies,
    then average across benchmarks (ignoring NaNs).
    Returns (compute, avg_acc_pct)
    """
    acc_matrix = []
    for bm in benchmarks:
        acc = extract_metric_series(data, compute, bm)
        acc_matrix.append(acc * 100)  # convert to percent
    acc_matrix = np.vstack(acc_matrix)  # shape = (B, T)
    avg = np.nanmean(acc_matrix, axis=0)  # length T
    return avg


def main():
    parser = argparse.ArgumentParser(
        description="Compare averaged benchmark curves for two experiment JSONs"
    )
    parser.add_argument("json1", type=Path, help="First JSON file with experiment data")
    parser.add_argument(
        "json2", type=Path, help="Second JSON file with experiment data"
    )
    parser.add_argument(
        "label1", type=str, help="Legend label for the first experiment"
    )
    parser.add_argument(
        "label2", type=str, help="Legend label for the second experiment"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("average_comparison.pdf"),
        help="Where to save the figure (PDF)",
    )
    args = parser.parse_args()

    # Load both experiments
    data1, compute1 = load_data(args.json1)
    data2, compute2 = load_data(args.json2)

    # Determine common benchmarks (ignore mmlu-* except 'mmlu_stem')
    first1 = sorted(data1.keys(), key=lambda k: float(k))[0]
    bms1 = {
        bm
        for bm, val in data1[first1].items()
        if isinstance(val, dict) and not (bm.startswith("mmlu") and bm != "mmlu_stem")
    }
    first2 = sorted(data2.keys(), key=lambda k: float(k))[0]
    bms2 = {
        bm
        for bm, val in data2[first2].items()
        if isinstance(val, dict) and not (bm.startswith("mmlu") and bm != "mmlu_stem")
    }
    benchmarks = sorted(bms1 & bms2)
    if not benchmarks:
        raise ValueError("No overlapping benchmarks found between the two files.")

    # Compute averages
    avg1 = average_accuracy(data1, compute1, benchmarks)
    avg2 = average_accuracy(data2, compute2, benchmarks)

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=compute1,
            y=avg1,
            mode="lines+markers",
            name=args.label1,
            line_shape="spline",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=compute2,
            y=avg2,
            mode="lines+markers",
            name=args.label2,
            line_shape="spline",
        )
    )

    fig.update_layout(
        xaxis_title="Compute Saved (%)",
        yaxis_title="Average Accuracy (%)",
        template="simple_white",
        legend=dict(
            x=0.95,
            y=0.95,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",  # transparent background
        ),
    )
    fig.update_yaxes(rangemode="tozero")

    # Save to PDF
    fig.write_image(str(args.output), engine="kaleido")
    time.sleep(1)
    fig.write_image(str(args.output), engine="kaleido")
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
