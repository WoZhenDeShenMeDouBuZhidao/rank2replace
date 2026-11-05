import argparse
import json
from . import paths


def main(experiment, smoke):
    config = json.loads(
        (paths.ROOT / "configs" / f"experiment{experiment}.json").read_text()
    )
    for dataset in config["datasets"]:
        jobs = [j for j in config["jobs"] if j["dataset"] == dataset]
        done = sum((paths.run_path(j, smoke) / "result.json").exists() for j in jobs)
        started = sum(
            (paths.run_path(j, smoke) / "manifest.json").exists() for j in jobs
        )
        print(
            f"{dataset:24s} completed {done:5d}/{len(jobs):5d}; started but not complete {started - done}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, type=int, choices=(1, 2, 3))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    main(a.experiment, a.smoke)
