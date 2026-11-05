"""Local multi-GPU launcher. Each run owns a file lock and an independent process."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from . import paths


def main(args):
    config = json.loads(
        (paths.ROOT / "configs" / f"experiment{args.experiment}.json").read_text()
    )
    jobs = [
        j
        for j in config["jobs"]
        if (not args.methods or j["method"] in args.methods)
        and (not args.datasets or j["dataset"] in args.datasets)
        and (not args.arms or j["arm"] in args.arms)
        and (not args.qualities or j.get("quality") in args.qualities)
    ]
    if args.limit:
        jobs = jobs[: args.limit]
    complete = sum(
        (paths.run_path(j, args.smoke) / "result.json").exists() for j in jobs
    )
    print(f"{len(jobs)} selected runs; {complete} already complete", flush=True)
    if args.dry_run:
        return
    active = []
    failed = []
    devices = args.gpus.split(",") if args.device == "cuda" else ["cpu"]
    pending = list(jobs)
    try:
        while pending or active:
            for item in active[:]:
                child, gpu, job, lock, log = item
                code = child.poll()
                if code is None:
                    continue
                log.close()
                lock.close()
                active.remove(item)
                if code:
                    failed.append(paths.key(job))
            for gpu in devices:
                if sum(item[1] == gpu for item in active) >= args.workers_per_gpu:
                    continue
                while pending:
                    job = pending.pop(0)
                    folder = paths.run_path(job, args.smoke)
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder / "result.json").exists():
                        continue
                    lock = (folder / "run.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    log = (folder / "train.log").open("a")
                    command = [
                        sys.executable,
                        "-m",
                        "rank2replace.train",
                        "--job",
                        json.dumps(job),
                        "--device",
                        "cpu" if gpu == "cpu" else "cuda:0",
                    ]
                    if args.smoke:
                        command.append("--smoke")
                    env = dict(os.environ)
                    if gpu != "cpu":
                        env["CUDA_VISIBLE_DEVICES"] = gpu
                    child = subprocess.Popen(
                        command,
                        cwd=paths.ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock.fileno(),),
                    )
                    active.append((child, gpu, job, lock, log))
                    print("START", gpu, paths.key(job), flush=True)
                    break
            time.sleep(2)
    except KeyboardInterrupt:
        print(
            "Stop requested; waiting for active runs to finish. No new runs will start.",
            flush=True,
        )
        for child, gpu, job, lock, log in active:
            code = child.wait()
            log.close()
            lock.close()
            if code:
                failed.append(paths.key(job))
        raise SystemExit(130)
    if failed:
        print("FAILED", failed, flush=True)
        raise SystemExit(1)
    print(
        "Finished selected runs. Use rank2replace.report to select hyperparameters and aggregate.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=int, required=True, choices=(1, 2, 3))
    p.add_argument("--gpus", default="0")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--qualities", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
