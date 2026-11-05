"""Shared frozen-slot training protocol for experiments 1, 2, and 3."""

import argparse
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import paths
from .corruption import SharedDonorCorruption
from .losses import (
    CrossEntropyCELoss,
    balanced_batches,
    balanced_ce,
    irm_terms,
    penalty_weight,
    GroupDRO,
    jtt_weights,
    rsc_loss,
    MulticlassLAAT,
    coral_loss,
    mixup_loss,
    scer_terms,
)
from .metrics import metrics
from .model import MLP


def make_corruption(job, features, device, offset=100000):
    k = features // 256
    return SharedDonorCorruption(
        [list(range(i * 256, (i + 1) * 256)) for i in range(k)],
        [r / 6 for r in range(k)],
        job["seed"] + offset,
        device,
    )


def training_groups(job, pack, rows):
    with np.load(paths.group_path(job)) as cache:
        all_rows = torch.where(pack["split"] == 0)[0]
        np.testing.assert_array_equal(cache["rows"], pack["rows"][all_rows].numpy())
        groups = cache["groups"].copy()
        if job["method"] == "coral":
            _, groups = np.unique(cache["environment"], return_inverse=True)
        elif job["method"] == "scer":
            groups = groups if job["dataset"].startswith("nicopp") else groups % 2
            _, groups = np.unique(groups, return_inverse=True)
    mapping = np.full(len(pack["y"]), -1, dtype=np.int64)
    mapping[all_rows.numpy()] = groups
    selected = mapping[rows.numpy()]
    assert (selected >= 0).all()
    return selected


def stratified_rows(pack, split, limit):
    rng = np.random.RandomState(9803 + split)
    rows = torch.where(pack["split"] == split)[0].numpy()
    if len(rows) <= limit:
        return torch.tensor(rows)
    labels = pack["y"][rows].numpy()
    mandatory = [int(rng.choice(rows[labels == c])) for c in np.unique(labels)]
    remaining = np.setdiff1d(rows, mandatory)
    return torch.tensor(
        np.sort(
            np.r_[
                mandatory, rng.choice(remaining, limit - len(mandatory), replace=False)
            ]
        )
    )


def train(job, smoke=False, device="cuda:0"):
    folder = paths.run_path(job, smoke)
    if (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        assert result["job"] == job and result["verified"]
        return result
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed, method = job["seed"], job["method"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pack = torch.load(paths.input_path(job), weights_only=True, mmap=True)
    X = pack["X"]
    if method == "hard_removal":
        X = X[:, : 256 * job["keep"]]
    y, env, split = (pack[k].numpy() for k in ("y", "environment", "split"))
    tr, va = (torch.where(pack["split"] == s)[0] for s in (0, 1))
    if smoke:
        tr, va = (stratified_rows(pack, s, n) for s, n in ((0, 1024), (1, 512)))
    tx, ty = X[tr].to(device), pack["y"][tr].to(device)
    vx, vy = X[va].to(device), pack["y"][va].to(device)
    classes, features = len(np.unique(y[split == 0])), X.shape[1]
    counts = {
        f"y{c}_e{e}": int(((y == c) & (env == e) & (split == 0)).sum())
        for c, e in set(zip(y[split == 0], env[split == 0]))
    }
    model = MLP(features, num_classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        torch.arange(len(ty)), batch_size=256, shuffle=True, generator=generator
    )
    rng = np.random.RandomState(seed)
    ce = CrossEntropyCELoss(reweighting=True)
    extra_method = method in ("elrep", "sagm", "swad", "scer")
    corrupt = method == "rank2replace" or job["arm"] == "C_rank_corrupted"
    corruption = make_corruption(job, features, device) if corrupt else None
    bn_corruption = (
        make_corruption(job, features, device, 200000)
        if corrupt and method == "swad"
        else None
    )
    groups = (
        training_groups(job, pack, tr)
        if method in ("irm", "group_dro", "coral", "scer")
        else None
    )
    group_tensor = torch.tensor(groups, device=device) if groups is not None else None
    dro = (
        GroupDRO(len(np.unique(groups)), job["dro_step_size"], device)
        if method == "group_dro"
        else None
    )
    laat = (
        MulticlassLAAT(list(range(7, 0, -1))).to(device) if method == "laat" else None
    )
    scale = torch.tensor(1.0, device=device, requires_grad=True)
    control, segment = None, None
    if method == "sagm":
        from .vendor.sagm.sagm import SAGM

        optimizer = SAGM(
            model.parameters(),
            optimizer,
            model,
            job["alpha"],
            SimpleNamespace(step=lambda: job["rho"]),
            adaptive=False,
        )
    if method == "swad":
        from .vendor.swad import LossValley
        from .vendor import swa_utils

        evaluator = SimpleNamespace(
            logger=SimpleNamespace(error=lambda message: print(message, flush=True))
        )
        control = LossValley(
            evaluator,
            n_converge=job["n_converge"],
            n_tolerance=job["n_tolerance"],
            tolerance_ratio=job["tolerance_ratio"],
        )
        segment = swa_utils.AveragedModel(model)
    epochs, patience = (2 if smoke else 5000), 200
    paths.save(
        folder / "manifest.json",
        dict(
            job=job,
            representation=paths.representation(job),
            features=features,
            lr=1e-4,
            batch_size=256,
            max_epochs=epochs,
            patience=patience,
            weight_decay=0,
            selection="clean validation balanced accuracy + macro F1",
            smoke=smoke,
            corruption="batch-shared permutation donor, independent rank masks, recipient label"
            if corrupt
            else None,
            jtt_stage1="clean inputs and clean error identification",
            group_assignment="recipient",
            device=device,
        ),
    )

    def evaluate(net, inputs, labels, environments):
        net.eval()
        with torch.no_grad():
            logits = torch.cat(
                [net(b) for b in inputs.split(1024 if extra_method else 2048)]
            )
            loss = float(balanced_ce(logits, labels))
        prediction = logits.argmax(1).cpu().numpy()
        return (
            metrics(
                labels.cpu().numpy(), prediction, environments, job["dataset"], counts
            ),
            loss,
            prediction,
        )

    weights, extra = None, {}
    if method == "jtt":
        for epoch in range(1, (1 if smoke else job["jtt_T"]) + 1):
            model.train()
            for ix in loader:
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(tx[ix]), ty[ix]).backward()
                optimizer.step()
            paths.save(
                folder / "progress.json", dict(stage="jtt_identification", epoch=epoch)
            )
        model.eval()
        with torch.no_grad():
            predicted = torch.cat([model(b).argmax(1).cpu() for b in tx.split(2048)])
        weights = jtt_weights(predicted, ty.cpu(), job["jtt_u"])
        errors = predicted != ty.cpu()
        torch.save(dict(rows=pack["rows"][tr], errors=errors), folder / "jtt_errors.pt")
        extra["jtt_error_count"] = int(errors.sum())
        torch.manual_seed(seed + 1)
        model = MLP(features, num_classes=classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history, resets = [], []
    best, best_epoch, state, step = -float("inf"), 0, None, 0
    started = time.monotonic()
    stop = "max_epochs"
    for epoch in range(1, epochs + 1):
        model.train()
        totals = torch.zeros(3, device=device)
        n = 0
        if method in ("irm", "group_dro", "coral"):
            batches = balanced_batches(groups, 256, rng)
        elif weights is not None:
            batches = torch.multinomial(
                weights, len(ty), replacement=True, generator=generator
            ).split(256)
        else:
            batches = loader
        for ix in batches:
            ix = torch.as_tensor(ix)
            if extra_method and len(ix) == 1:
                continue
            bx, by = tx[ix], ty[ix]
            if corruption is not None:
                bx = corruption(bx)
            if method == "irm" and step == 500:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                resets.append(step)
            auxiliary = bx.new_zeros(())
            if method == "sagm":
                optimizer.set_closure(balanced_ce, bx, by)
                _, loss = optimizer.step()
                base = loss
            else:
                optimizer.zero_grad(set_to_none=True)
                if method == "irm":
                    scale.grad = None
                    base, auxiliary = irm_terms(model(bx), by, scale)
                    loss = base + penalty_weight(job["strength"], step, 500) * auxiliary
                elif method == "group_dro":
                    loss, _ = dro(model(bx), by, group_tensor[ix])
                    base = loss
                elif method == "jtt":
                    loss = base = F.cross_entropy(model(bx), by)
                elif method == "rsc":
                    loss, _ = rsc_loss(model, bx, by, job["rsc_p_f"], job["rsc_p_b"])
                    base = loss
                elif method == "laat":
                    loss, base, auxiliary = laat(model, bx, by, job["strength"])
                elif method == "coral":
                    loss, base, auxiliary = coral_loss(
                        model, bx, by, group_tensor[ix], job["strength"]
                    )
                elif method == "mixup":
                    loss, base, auxiliary = mixup_loss(model, bx, by, job["alpha"], rng)
                elif method in ("elrep", "scer"):
                    hidden = model.mlp[:-1](bx)
                    if method == "elrep":
                        auxiliary = (
                            job["theta1"]
                            * torch.linalg.matrix_norm(hidden, ord="nuc")
                            / len(hidden)
                        )
                        auxiliary = auxiliary + job[
                            "theta2"
                        ] * hidden.square().sum() / len(hidden)
                    else:
                        spur, core, _ = scer_terms(
                            hidden,
                            by,
                            group_tensor[ix],
                            model.mlp[-1].weight,
                            classes,
                            len(np.unique(groups)),
                        )
                        auxiliary = (
                            job["lambda_spur"] * spur + job["lambda_core"] * core
                        )
                    base = balanced_ce(model.mlp[-1](hidden), by)
                    loss = base + auxiliary
                else:
                    loss = base = ce(model, bx, by)[1]
                loss.backward()
                optimizer.step()
            totals += torch.stack(
                [loss.detach(), base.detach(), auxiliary.detach()]
            ) * len(by)
            n += len(by)
            if control is not None:
                segment.update_parameters(model, step=step)
            step += 1
        terms = (totals / n).cpu().tolist()
        if not np.isfinite(terms).all():
            raise FloatingPointError(paths.key(job))
        val, val_loss, _ = evaluate(model, vx, vy, env[va])
        history.append(
            dict(
                epoch=epoch,
                train=dict(zip(("total", "base", "auxiliary"), terms)),
                validation=val,
                validation_loss=val_loss,
            )
        )
        if val["selection"] > best:
            best, best_epoch = val["selection"], epoch
            if control is None:
                state = copy.deepcopy(model.state_dict())
        if control is not None:
            control.update_and_evaluate(
                segment, val["selection"], val_loss, lambda *a: None
            )
            segment = swa_utils.AveragedModel(model)
        if epoch == 1 or epoch % 10 == 0:
            paths.save(folder / "history.json", history)
            paths.save(
                folder / "progress.json",
                dict(
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_selection=best,
                    elapsed_seconds=time.monotonic() - started,
                    seconds_per_epoch=(time.monotonic() - started) / epoch,
                ),
            )
            print(paths.key(job), epoch, val["selection"], flush=True)
        if control is not None and control.dead_valley:
            stop = "swad_dead_valley"
            break
        if epoch - best_epoch >= patience:
            stop = "patience"
            break
    if control is not None:
        averaged = control.get_final_model()
        model = averaged.module.to(device)

        def bn_batches():
            for batch in tx.split(256):
                if len(batch) > 1:
                    yield [{"x": bn_corruption(batch) if bn_corruption else batch}]

        swa_utils.update_bn(
            iter(bn_batches()),
            model,
            sum(len(b) > 1 for b in tx.split(256)),
            device=device,
        )
        extra.update(
            selected_model="official SWAD loss-valley average",
            trajectory_best_epoch=best_epoch,
            averaged_start_step=averaged.start_step,
            averaged_end_step=averaged.end_step,
        )
    else:
        model.load_state_dict(state)
    val, val_loss, pred = evaluate(model, vx, vy, env[va])
    if control is None:
        assert val["selection"] == best
    checkpoint = dict(
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        job=job,
        best_epoch=best_epoch,
    )
    torch.save(checkpoint, folder / "checkpoint.pt")
    reloaded = MLP(features, num_classes=classes).to(device).eval()
    reloaded.load_state_dict(
        torch.load(folder / "checkpoint.pt", weights_only=True)["state_dict"]
    )
    other, _, other_pred = evaluate(reloaded, vx, vy, env[va])
    assert other == val
    np.testing.assert_array_equal(pred, other_pred)
    result = dict(
        job=job,
        verified=True,
        status="complete",
        representation=paths.representation(job),
        features=features,
        epochs=epoch,
        best_epoch=best_epoch,
        stop_reason=stop,
        validation=val,
        test_evaluated=not smoke,
        elapsed_seconds=time.monotonic() - started,
        **extra,
    )
    predictions = dict(validation=pred, validation_rows=pack["rows"][va].numpy())
    if not smoke:
        rows = torch.where(pack["split"] == 2)[0]
        with torch.no_grad():
            predicted = torch.cat(
                [
                    reloaded(X[ix].to(device)).argmax(1).cpu()
                    for ix in rows.split(1024 if extra_method else 2048)
                ]
            ).numpy()
        result["test"] = metrics(y[rows], predicted, env[rows], job["dataset"], counts)
        predictions.update(test=predicted, test_rows=pack["rows"][rows].numpy())
    if corruption is not None:
        result["corruption"] = corruption.report(
            [f"R{i + 1}" for i in range(features // 256)]
        )
    if method == "group_dro":
        result["final_group_weights"] = dro.weights.cpu().tolist()
    if method == "irm":
        result["optimizer_reset_steps"] = resets
    np.savez_compressed(folder / "predictions.npz", **predictions)
    paths.save(folder / "history.json", history)
    paths.save(folder / "result.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=json.loads, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    train(a.job, a.smoke, a.device)
