#!/usr/bin/env python3
import sys
import json
import os


def load_samples(file_path, bench_key):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    if bench_key not in data or not isinstance(data[bench_key], list):
        print(
            f"Benchmark '{bench_key}' not found or not a list in {file_path}. Available keys: {list(data.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    return data[bench_key]


def main():
    if len(sys.argv) < 4:
        print(
            f"Usage: {os.path.basename(sys.argv[0])} <benchmark> <file1.json> <file2.json> [file3.json ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    bench_key = sys.argv[1].lower()
    if bench_key not in ("gsm8k", "csqa_gen"):
        print("Benchmark must be 'gsm8k' or 'csqa_gen'", file=sys.stderr)
        sys.exit(1)

    file_paths = sys.argv[2:]
    samples_lists = [load_samples(fp, bench_key) for fp in file_paths]

    # Check all files have the same number of samples
    lengths = [len(lst) for lst in samples_lists]
    if len(set(lengths)) != 1:
        print(
            "Error: JSON files contain differing numbers of samples:",
            lengths,
            file=sys.stderr,
        )
        sys.exit(1)
    total = lengths[0]

    for idx in range(total):
        print(f"\n=== Sample {idx+1}/{total} ===\n")
        for fp, samples in zip(file_paths, samples_lists):
            sample = samples[idx]
            # Extract question prompt and model response
            try:
                question = sample["arguments"][0][0]
            except (KeyError, IndexError, TypeError):
                question = "<could not extract question>"
            try:
                answer = sample["resps"][0][0]
            except (KeyError, IndexError, TypeError):
                answer = "<could not extract response>"

            print(f"--- {os.path.basename(fp)} ---")
            print(question)
            print()
            print(answer)
            print("\n" + "-" * 40 + "\n")

        input("Press Enter to continue to the next sample...")


if __name__ == "__main__":
    main()
