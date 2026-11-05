"""Download pinned model weights into this demo; reuse complete local files."""

import argparse
import json
from pathlib import Path

from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(__file__).resolve().parent / "models"
FT_REPO = "mseitzer/ftdinosaur"
FT_REVISION = "396e49cdd2f578e731bc3e62370634f801f4b3a3"
FT_FILENAME = "dinosaur_small_patch14_224_topk3.coco_dv2_ft_s7_300k-fbc7a7c7.pth"
QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def ensure_ft():
    folder = MODELS / "ft-dinosaur"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FT_FILENAME
    with FileLock(str(folder / ".download.lock")):
        if not path.is_file() or path.stat().st_size == 0:
            hf_hub_download(
                FT_REPO, FT_FILENAME, revision=FT_REVISION, local_dir=folder
            )
    return path


def qwen_complete(folder):
    marker = folder / "download.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get("revision") != QWEN_REVISION:
        return False
    return all(
        (folder / name).is_file() and (folder / name).stat().st_size == size
        for name, size in record["files"].items()
    )


def ensure_qwen():
    folder = MODELS / "Qwen3-VL-8B-Instruct"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".download.lock")):
        if not qwen_complete(folder):
            marker = folder / "download.json"
            if marker.exists():
                # HF's local metadata can otherwise reuse a truncated file.
                for name, size in json.loads(marker.read_text())["files"].items():
                    path = folder / name
                    if path.is_file() and path.stat().st_size != size:
                        hf_hub_download(
                            QWEN_REPO,
                            name,
                            revision=QWEN_REVISION,
                            local_dir=folder,
                            force_download=True,
                        )
            snapshot_download(
                QWEN_REPO,
                revision=QWEN_REVISION,
                local_dir=folder,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.jinja",
                    "README.md",
                    "LICENSE*",
                ],
                max_workers=4,
            )
            index = json.loads((folder / "model.safetensors.index.json").read_text())
            required = set(index["weight_map"].values()) | {
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "chat_template.json",
                "model.safetensors.index.json",
            }
            if not all(
                (folder / name).is_file() and (folder / name).stat().st_size > 0
                for name in required
            ):
                raise RuntimeError(
                    "Qwen download is incomplete; rerun download_models.py"
                )
            files = {
                p.name: p.stat().st_size
                for p in folder.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.name != "download.json"
            }
            record = {"repo_id": QWEN_REPO, "revision": QWEN_REVISION, "files": files}
            temporary = folder / ".download.json.tmp"
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(folder / "download.json")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("ft", "qwen", "all"), default="all")
    args = parser.parse_args()
    if args.only in ("ft", "all"):
        print(ensure_ft())
    if args.only in ("qwen", "all"):
        print(ensure_qwen())
