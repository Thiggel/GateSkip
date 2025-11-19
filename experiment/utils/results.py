import os
from pathlib import Path


def _base_cache_dir() -> Path:
    """Return the base cache directory (defaults to current working directory)."""
    base_cache = os.environ.get("BASE_CACHE_DIR", "")
    if base_cache:
        return Path(base_cache)
    return Path(".")


def get_results_suffix() -> str:
    """Return the optional suffix appended to evaluation result filenames."""
    suffix = os.environ.get("EVAL_RESULTS_SUFFIX", "").strip()
    if not suffix:
        return ""
    if not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return suffix


def get_results_file_path(experiment_name: str, seed: int) -> Path:
    """Construct the output path for a (possibly suffixed) evaluation result file."""
    results_dir = _base_cache_dir() / "results"
    suffix = get_results_suffix()
    filename = f"{experiment_name}{suffix}_seed{seed}_results.json"
    return results_dir / filename
