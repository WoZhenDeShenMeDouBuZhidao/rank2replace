import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def metrics(y, prediction, environment, dataset, training_counts):
    correct = y == prediction
    ba = float(balanced_accuracy_score(y, prediction))
    f1 = float(f1_score(y, prediction, average="macro", zero_division=0))
    groups = {}
    for label, env in sorted(set(zip(y.tolist(), environment.tolist()))):
        mask = (y == label) & (environment == env)
        groups[f"y{label}_e{env}"] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
        }
    result = {
        "balanced_accuracy": ba,
        "macro_f1": f1,
        "selection": ba + f1,
        "raw_accuracy": float(correct.mean()),
        "groups": groups,
    }
    if dataset.startswith("nicopp_"):
        result["environment_accuracy"] = {
            str(e): float(correct[environment == e].mean())
            for e in np.unique(environment)
        }
        result["mean_environment_accuracy"] = float(
            np.mean(list(result["environment_accuracy"].values()))
        )
    else:
        result["worst_group_accuracy"] = min(v["accuracy"] for v in groups.values())
        if dataset == "waterbirds":
            result["adjusted_accuracy"] = sum(
                groups[k]["accuracy"] * n for k, n in training_counts.items()
            ) / sum(training_counts.values())
    return result
