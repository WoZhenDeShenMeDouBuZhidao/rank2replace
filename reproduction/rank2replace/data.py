"""Read copied slots/images and optionally rebuild train-standardized inputs."""

import argparse
import json
import tarfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch
from . import paths


def dataset_folder(dataset):
    return paths.DATA / ("nicopp" if dataset.startswith("nicopp_") else dataset)


def slot_image(dataset, row, slot):
    folder = dataset_folder(dataset)
    manifest = json.loads((folder / "manifest.json").read_text())
    chunk = next(c for c in manifest["chunks"] if row in c["rows"])
    with tarfile.open(folder / chunk["slot_images"]) as archive:
        with archive.extractfile(f"{row:06d}/slot_{slot}.png") as stream:
            return Image.open(stream).convert("RGB")


def rebuild(dataset, verify_only=False):
    folder = dataset_folder(dataset)
    manifest = json.loads((folder / "manifest.json").read_text())
    frame = pd.read_csv(
        paths.split_path(dataset) / "metadata.csv", dtype={"image_id": str}
    )
    slots = torch.empty(len(frame), 7, 256)
    seen = np.zeros(len(frame), bool)
    for chunk in manifest["chunks"]:
        pack = torch.load(folder / chunk["file"], weights_only=True, mmap=True)
        rows = pack["rows"]
        assert not seen[rows].any()
        slots[rows] = pack["slots"]
        seen[rows] = True
    assert seen.all()
    ranks = json.loads((folder / "ranking/qwen.json").read_text())
    assert [r["row"] for r in ranks] == list(range(len(frame)))
    orders = torch.tensor([r["rank_order"] for r in ranks])
    for representation in ("slot_index", "rank"):
        values = (
            (
                slots
                if representation == "slot_index"
                else slots.gather(1, orders[:, :, None].expand_as(slots))
            )
            .flatten(1)
            .numpy()
        )
        scaler = StandardScaler().fit(values[frame.split.to_numpy() == 0])
        pack = dict(
            X=torch.from_numpy(scaler.transform(values).astype("float32")),
            scaler_mean=torch.from_numpy(scaler.mean_),
            scaler_scale=torch.from_numpy(scaler.scale_),
            rows=torch.arange(len(frame)),
            image_ids=frame.image_id.tolist(),
            **{
                k: torch.tensor(frame[k].to_numpy(), dtype=torch.long)
                for k in ("y", "environment", "split")
            },
        )
        if representation == "rank":
            pack["orders"] = orders
        target = paths.split_path(dataset) / "prepared" / f"{representation}.pt"
        if verify_only:
            original = torch.load(target, weights_only=True, mmap=True)
            for key, value in pack.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, original[key], atol=0, rtol=0)
                else:
                    assert value == original[key]
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pack, target.with_suffix(".tmp"))
            target.with_suffix(".tmp").replace(target)
        print(
            dataset,
            representation,
            "verified" if verify_only else "rebuilt",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    for dataset in a.datasets:
        rebuild(dataset, a.verify_only)
