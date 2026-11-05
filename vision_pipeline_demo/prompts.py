import base64
import hashlib
import io
import json
import numpy as np

DATASET_CONFIG = {
    "waterbirds": {
        "ranking_task": "classify bird species as waterbird versus landbird, robust to background changes",
        "environment_task": "classify bird species as waterbird versus landbird, robust to background changes; identify whether the background is land or water, not the bird type",
        "environment_options": {"land": 0, "water": 1},
    },
    "celeba": {
        "ranking_task": "classify whether the person has blond hair or not, robust to changes in other attributes",
        "environment_task": "classify blond hair versus not blond hair; the environment attribute here is the person's apparent visual gender presentation (female or male). Describe only visual presentation, not gender identity; do not use hair color as the criterion",
        "environment_options": {"female": 0, "male": 1},
    },
    "metashift": {
        "ranking_task": "classify the main animal as cat or dog, robust to indoor or outdoor environment changes",
        "environment_task": "classify cat versus dog; identify indoor or outdoor surroundings, not the animal type",
        "environment_options": {"indoor": 0, "outdoor": 1},
    },
    "nicopp": {
        "ranking_task": "classify the main object into one of these categories, robust to environment changes: car, flower, chair, truck, tiger, wheat, seal, wolf, lion, dolphin, lifeboat, corn, fishing rod, owl, sunflower, cow, bird, clock, shrimp, goose, airplane, rabbit, hot air balloon, lizard, hat, spider, motorcycle, tortoise, dog, crocodile, elephant, gun, fox, bus, cat, sailboat, giraffe, cactus, pumpkin, train, ship, helicopter, bicycle, racket, squirrel, bear, scooter, mailbox, horse, pineapple, frog, football, ostrich, tent, kangaroo, monkey, crab, sheep, butterfly, umbrella",
        "environment_task": "classify the main object; identify the visual context rather than the object class. Context definitions: autumn = autumn scenery or foliage; dim = dim lighting; grass = grassy surroundings; outdoor = general outdoor surroundings; rock = rocky surroundings; water = water surroundings. Choose the best matching allowed context",
        "environment_options": {"dim": 1, "grass": 2, "outdoor": 3, "water": 5},
    },
}


def permutation_grammar(ids):
    # Each state consumes one ID and transitions to the remaining subset.
    rules = ['root ::= "{\\"ranking\\":[" s%d "]}"' % ((1 << len(ids)) - 1)]
    for subset in range(1, 1 << len(ids)):
        alternatives = []
        for index, slot in enumerate(ids):
            if subset & (1 << index):
                rest = subset ^ (1 << index)
                token = json.dumps(slot) + ("," if rest else "")
                alternatives.append(json.dumps(token) + (f" s{rest}" if rest else ""))
        rules.append(f"s{subset} ::= " + " | ".join(alternatives))
    return "\n".join(rules)


def empty_order(empty, image_id, seed=42):
    key = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    return rng.permutation(np.flatnonzero(empty)).tolist()


def slot_order(empty, image_id, *, ranking, seed=42):
    empty = np.asarray(empty, dtype=bool)
    nonempty = np.flatnonzero(~empty).tolist()
    if len(ranking) != len(nonempty) or set(ranking) != set(nonempty):
        raise ValueError("Ranking must include every nonempty slot exactly once")
    return list(ranking) + empty_order(empty, image_id, seed)


def image_content(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        },
    }


def rank_request(images, empty, task):
    ids = tuple(f"S{i}" for i in range(7) if not empty[i])
    if not ids:
        raise ValueError("All-empty image: no ranking request needed")
    prompt = (
        f"Task: {task}. These regions come from one image. Gray pixels are excluded. "
        "Rank by intrinsic evidence for the target class, robust to environment changes. "
        "Use only visible evidence. Break indistinguishable ties by ascending slot ID. "
        f"Return JSON with key ranking, containing exactly these IDs once, best first: {','.join(ids)}."
    )
    content = [{"type": "text", "text": prompt}]
    for i in range(7):
        if not empty[i]:
            content += [
                {"type": "text", "text": f"Region S{i}:"},
                image_content(images[i]),
            ]
    return {
        "model": "qwen3-vl-8b",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 96,
        "seed": 9803,
        "structured_outputs": {"grammar": permutation_grammar(ids)},
    }


def environment_request(image, task, environments):
    prompt = (
        f"Task: {task}. Identify the environment visible in this image. "
        f"Choose one of: {', '.join(environments)}. Return JSON with key environment."
    )
    return {
        "model": "qwen3-vl-8b",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, image_content(image)],
            }
        ],
        "temperature": 0,
        "max_tokens": 48,
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "environment": {"type": "string", "enum": list(environments)}
                },
                "required": ["environment"],
                "additionalProperties": False,
            }
        },
    }
