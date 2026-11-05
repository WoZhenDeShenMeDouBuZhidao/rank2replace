import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("R2R_DATA", ROOT / "data")).resolve()
OUTPUT = Path(os.environ.get("R2R_OUTPUT", ROOT / "outputs")).resolve()


def experiment_path(experiment, original=False):
    root = ROOT / "outputs" if original else OUTPUT
    return (
        root
        / f"exp{experiment}"
        / ("original_results" if original else "reproduced_results")
    )


def split_path(dataset):
    if dataset.startswith("nicopp_"):
        return DATA / "nicopp" / "splits" / dataset.removeprefix("nicopp_")
    return DATA / dataset / "splits" / "default"


def representation(job):
    return "slot_index" if job["arm"] == "A_raw" else "rank"


def input_path(job):
    return split_path(job["dataset"]) / "prepared" / f"{representation(job)}.pt"


def group_path(job):
    return (
        split_path(job["dataset"])
        / "environment"
        / job.get("quality", "qwen")
        / f"s{job['seed']}.npz"
    )


def key(job):
    return "__".join(
        str(job.get(k, "none"))
        for k in (
            "experiment",
            "arm",
            "dataset",
            "method",
            "quality",
            "variant",
            "seed",
        )
    )


def run_path(job, smoke=False):
    return (
        experiment_path(job["experiment"]) / ("smoke" if smoke else "runs") / key(job)
    )


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temp.replace(path)
