"""Independent original-image environment annotation; no slot extraction or ranking."""

from PIL import Image
from common import ROOT, arguments, samples, request, save
from prompts import environment_request


def main(args):
    workflow = ROOT / "outputs" / "environment"
    source = workflow / "original_results"
    output = args.output or workflow / "reproduced_results"
    for item in samples(args, source):
        with Image.open(source / item["image"]) as original:
            image = original.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        options = item["environment_options"]
        body = environment_request(image, item["environment_task"], list(options))
        body["seed"] = 9803
        parsed, usage = request(body, args)
        label = parsed["environment"]
        record = dict(
            environment=label,
            environment_id=options[label],
            row=item["row"],
            usage=usage,
        )
        save(output / item["id"] / "environment.json", record)
        print(item["id"], record["environment"], flush=True)


if __name__ == "__main__":
    main(arguments(__doc__).parse_args())
