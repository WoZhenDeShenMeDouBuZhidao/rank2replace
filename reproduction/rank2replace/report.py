"""Per-split validation selection and seed-wise macro aggregation."""

import argparse
from collections import defaultdict
import csv
import importlib
import json

import numpy as np
from . import paths


def condition_name(condition):
    return {
        "A_raw": "Original slot order",
        "B_rank": "Slot ranking",
        "C_rank_corrupted": "Slot ranking + slot corruption",
    }.get(condition, condition)


def metric(r, name):
    if name == "validation":
        return r["validation"]["selection"]
    g = (
        list(r["test"]["environment_accuracy"].values())
        if "environment_accuracy" in r["test"]
        else [v["accuracy"] for v in r["test"]["groups"].values()]
    )
    return 100 * float(np.mean(g) if name == "average" else min(g))


def macro(values):
    a = np.array(values, dtype=float)
    assert a.shape == (6, 3)
    return np.vstack([a[:3], a[3:].mean(0)]).mean(0)


def main(args):
    config = json.loads(
        (paths.ROOT / "configs" / f"experiment{args.experiment}.json").read_text()
    )
    datasets, seeds, methods = config["datasets"], config["seeds"], config["methods"]
    if args.reference:
        rs = json.loads(
            (
                paths.experiment_path(args.experiment, original=True) / "results.json"
            ).read_text()
        )
        records = {paths.key(r["job"]): r for r in rs}
    else:
        records = {}
        for j in config["jobs"]:
            p = paths.run_path(j) / "result.json"
            if p.exists():
                r = json.loads(p.read_text())
                assert r["verified"] and r["test_evaluated"]
                records[paths.key(j)] = r
    groups = defaultdict(lambda: defaultdict(list))
    for j in config["jobs"]:
        condition = (
            j.get("quality")
            if args.experiment == 3
            else j["arm"]
            if args.experiment == 2
            else "overall"
        )
        bucket = (j["dataset"], j["method"], condition)
        groups[bucket][j["variant"]].append(j)
    selected = {}
    rows = []
    pending = []
    for bucket, options in groups.items():
        candidates = []
        for variant, jobs in options.items():
            results = [records[paths.key(j)] for j in jobs if paths.key(j) in records]
            assert [j["seed"] for j in jobs] == seeds
            row = dict(
                dataset=bucket[0],
                method=bucket[1],
                condition=condition_name(bucket[2]),
                variant=variant,
                completed=len(results),
                validation=float(np.mean([metric(r, "validation") for r in results]))
                if len(results) == 3
                else None,
            )
            rows.append(row)
            candidates.append((row, results))
        if any(len(v) != 3 for _, v in candidates):
            pending.append(bucket)
            continue
        row, results = max(candidates, key=lambda pair: pair[0]["validation"])
        selected[bucket] = dict(setting=dict(variant=row["variant"]), entries=results)
    dest = paths.experiment_path(args.experiment, original=args.reference)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    macro_rows = []
    conditions = list(dict.fromkeys(b[2] for b in groups))
    for m in methods:
        for condition in conditions:
            if not all((d, m, condition) in selected for d in datasets):
                continue
            for name in ("average", "worst"):
                values = macro(
                    [
                        [metric(r, name) for r in selected[d, m, condition]["entries"]]
                        for d in datasets
                    ]
                )
                macro_rows.append(
                    dict(
                        method=m,
                        condition=condition_name(condition),
                        metric=name,
                        mean=float(values.mean()),
                        std=float(values.std(ddof=1)),
                        seed_values=values.tolist(),
                    )
                )
    paths.save(
        dest / "summary.json",
        dict(
            reference=args.reference,
            completed=len(records),
            total=len(config["jobs"]),
            pending=[(d, m, condition_name(c)) for d, m, c in pending],
            macro=macro_rows,
            selected=[
                dict(
                    dataset=d,
                    method=m,
                    condition=condition_name(c),
                    variant=v["setting"]["variant"],
                    validation=[metric(r, "validation") for r in v["entries"]],
                    metrics={
                        n: [metric(r, n) for r in v["entries"]]
                        for n in ("average", "worst")
                    },
                )
                for (d, m, c), v in selected.items()
            ],
        ),
    )
    if pending:
        print(f"{len(pending)} incomplete sweeps; tables saved, final plot withheld.")
        return
    labels = json.loads((paths.ROOT / "configs/labels.json").read_text())
    plot = importlib.import_module(f".plot{args.experiment}", __package__)
    if args.experiment == 1:
        plot.draw(
            {d: {m: selected[d, m, "overall"] for m in methods} for d in datasets},
            config,
            labels,
            dest,
        )
    elif args.experiment == 2:
        arms = ("A_raw", "B_rank", "C_rank_corrupted")
        plot.draw(
            {
                d: {
                    m: {a: selected[d, m, a] for a in arms if (d, m, a) in selected}
                    for m in methods
                }
                for d in datasets
            },
            config,
            labels,
            dest,
        )
    else:
        ref_config = json.loads((paths.ROOT / "configs/experiment1.json").read_text())
        if args.reference:
            reference = json.loads(
                (paths.experiment_path(1, original=True) / "results.json").read_text()
            )
            reference = [r for r in reference if r["job"]["method"] == "rank2replace"]
        else:
            jobs = [j for j in ref_config["jobs"] if j["method"] == "rank2replace"]
            missing = [
                j for j in jobs if not (paths.run_path(j) / "result.json").exists()
            ]
            if missing:
                print(
                    "Tables complete; experiment 3 plot also needs experiment 1 Rank2Replace reference runs."
                )
                return
            reference = [
                json.loads((paths.run_path(j) / "result.json").read_text())
                for j in jobs
            ]
        refs = {(r["job"]["dataset"], r["job"]["seed"]): r for r in reference}
        reference = []
        for n in ("average", "worst"):
            values = macro([[metric(refs[d, s], n) for s in seeds] for d in datasets])
            reference.append(
                dict(metric=n, mean=float(values.mean()), std=float(values.std(ddof=1)))
            )
        values = dict(
            downstream=[dict(r, quality=r["condition"]) for r in macro_rows],
            reference=reference,
        )
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 12,
                "pdf.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        plot.draw_downstream(values, methods, conditions, dest)
    print("REPORT", dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--reference", action="store_true")
    main(p.parse_args())
