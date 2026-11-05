"""Frozen object-centric extraction, then constrained VLM slot ranking."""

import os

for key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(key, "1")
import sys
from PIL import Image, ImageDraw
import torch
from common import ROOT, arguments, samples, request, save
from prompts import rank_request, slot_order


def regions(image, masks):
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    hard = masks.argmax(0).reshape(16, 16).numpy()
    empty = [not (hard == i).any() for i in range(7)]
    images = []
    for i in range(7):
        mask = Image.fromarray((hard == i).astype("uint8") * 255).resize(
            image.size, Image.Resampling.NEAREST
        )
        images.append(
            Image.composite(image, Image.new("RGB", image.size, (128, 128, 128)), mask)
        )
    return images, empty


def main(args):
    torch.set_num_threads(1)
    workflow = ROOT / "outputs" / "extract_rank"
    source = workflow / "original_results"
    output = args.output or workflow / "reproduced_results"
    from download_models import ensure_ft

    sys.path.insert(0, str(ROOT / "vendor"))
    from ftdinosaur_inference import build_dinosaur

    name = "dinosaur_small_patch14_224_topk3.coco_dv2_ft_s7_300k"
    model = build_dinosaur.build(name, pretrained=False).to(args.device).eval()
    model.load_state_dict(
        torch.load(ensure_ft(), weights_only=True, map_location=args.device)
    )
    transform = build_dinosaur.build_preprocessing(name)
    initial = torch.load(ROOT / "assets/fixed_slot_init.pt", weights_only=True).to(
        args.device
    )
    for item in samples(args, source):
        folder = output / item["id"]
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(source / item["image"]) as original:
            image = original.convert("RGB")
        with torch.inference_mode():
            result = model(transform(image)[None].to(args.device), slot_init=initial)
        pack = {k: result[k][0].cpu() for k in ("slots", "masks")}
        torch.save(pack, folder / "extracted.pt")
        images, empty = regions(image, pack["masks"])
        for i, img in enumerate(images):
            img.save(folder / f"slot_{i}.png")
        if args.extract_only:
            order = list(range(7))
            record = dict(slot_order=order, ranked=False)
        else:
            if all(empty):
                ranking = []
            else:
                parsed, _ = request(
                    rank_request(images, empty, item["ranking_task"]), args
                )
                ranking = [int(s.removeprefix("S")) for s in parsed["ranking"]]
            order = slot_order(
                empty, f"{item['dataset']}:{item['row']}", ranking=ranking
            )
            record = dict(rank_order=order)
        save(folder / "ranking.json", record)
        torch.save(
            dict(
                order=torch.tensor(order),
                slots=pack["slots"][order],
                standardized=False,
            ),
            folder / "ordered_embeddings.pt",
        )
        sheet = Image.new("RGB", (7 * 224, 2 * 248), "white")
        draw = ImageDraw.Draw(sheet)
        for line, indices in enumerate((range(7), order)):
            for col, slot in enumerate(indices):
                sheet.paste(images[slot], (col * 224, line * 248 + 24))
                draw.text(
                    (col * 224 + 8, line * 248 + 4),
                    f"Slot {slot}" if line == 0 else f"Rank {col + 1}: slot {slot}",
                    fill="black",
                )
        sheet.save(folder / "comparison.png")
        print(item["id"], order, folder, flush=True)


if __name__ == "__main__":
    p = arguments(__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="Run the real extractor without calling a VLM",
    )
    main(p.parse_args())
