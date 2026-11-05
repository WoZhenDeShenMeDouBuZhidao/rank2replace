import argparse
import json
import os
from pathlib import Path
import urllib.request
from prompts import DATASET_CONFIG

ROOT = Path(__file__).resolve().parent


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Folder names in original_results; default: all images",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--output", type=Path)
    return parser


def samples(args, source):
    records = []
    for image in sorted(source.glob("*/image.jpg")):
        sample_id = image.parent.name
        dataset, row = sample_id.rsplit("_", 1)
        records.append(
            dict(
                id=sample_id,
                dataset=dataset,
                row=int(row),
                image=str(image.relative_to(source)),
                **DATASET_CONFIG[dataset],
            )
        )
    if args.samples:
        available = {r["id"] for r in records}
        if not set(args.samples) <= available:
            raise ValueError(f"Available samples: {sorted(available)}")
        records = [r for r in records if r["id"] in args.samples]
    return records


def request(body, args):
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ.get("VLLM_API_KEY")
    )
    if not key:
        raise ValueError("Set VLLM_API_KEY or --api-key-file")
    body = dict(body, model=args.model)
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("VLM response was truncated")
    return json.loads(choice["message"]["content"]), result.get("usage")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
