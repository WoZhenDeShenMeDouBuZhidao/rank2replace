from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from itertools import combinations
from typing import Dict, List, Tuple

LossOutput = Tuple[
    torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]


class _ClassReweightedLossMixin:
    def forward(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        if self.reweighting:
            return self._forward_reweighted(model, X, Y)
        return self._forward_unweighted(model, X, Y)


def _present_class_indices(Y: torch.Tensor) -> List[torch.Tensor]:
    return [
        (Y == class_value).nonzero(as_tuple=True)[0]
        for class_value in torch.unique(Y.detach(), sorted=True)
    ]


def _mean_over_present_classes(
    values: torch.Tensor,
    class_indices: List[torch.Tensor],
) -> torch.Tensor:
    return torch.stack(
        [values.index_select(0, indices).mean() for indices in class_indices]
    ).mean()


class CrossEntropyCELoss(_ClassReweightedLossMixin, nn.Module):
    def __init__(self, reweighting: bool = False):
        super().__init__()
        self.reweighting = reweighting

    def _forward_unweighted(
        self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor
    ) -> LossOutput:
        logits = model(X)
        ce = F.cross_entropy(logits, Y)
        return logits, ce, {"CE_loss": ce, "total_loss": ce}, {}

    def _forward_reweighted(
        self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor
    ) -> LossOutput:
        logits = model(X)
        class_indices = _present_class_indices(Y)
        ce = _mean_over_present_classes(
            F.cross_entropy(logits, Y, reduction="none"),
            class_indices,
        )
        return logits, ce, {"CE_loss": ce, "total_loss": ce}, {}


def irm_terms(logits, labels, scale):
    # WILDS uses the product of gradients from two disjoint halves per group.
    losses = F.cross_entropy(scale * logits, labels, reduction="none")
    penalties = []
    for group_losses in losses.chunk(4):
        grad1 = torch.autograd.grad(
            group_losses[0::2].mean(), scale, create_graph=True
        )[0]
        grad2 = torch.autograd.grad(
            group_losses[1::2].mean(), scale, create_graph=True
        )[0]
        penalties.append((grad1 * grad2).sum())
    return losses.mean(), torch.stack(penalties).mean()


def penalty_weight(strength, step, anneal_steps):
    if strength == 0:
        return 0.0
    return strength if step >= anneal_steps else 1.0


def group_norms(attributions):
    return attributions.reshape(-1, 7, 256).square().sum(-1).clamp_min(1e-24).sqrt()


def balanced_mean(values, y):
    return torch.stack([values[y == label].mean(0) for label in torch.unique(y)]).mean(
        0
    )


class GroupLAAT(torch.nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.register_buffer(
            "target", F.normalize(torch.tensor(prior, dtype=torch.float32), dim=0)
        )
        self.ce = CrossEntropyCELoss(reweighting=True)

    def attribution(self, model, X, create_graph):
        previous = model.training
        model.eval()
        try:
            inputs = X.detach().requires_grad_(True)
            logits = model(inputs)
            gradients = torch.autograd.grad(
                (logits[:, 1] - logits[:, 0]).sum(), inputs, create_graph=create_graph
            )[0]
            amplitudes = group_norms(inputs * gradients)
            normalized = F.normalize(amplitudes, p=2, dim=1, eps=1e-12)
            return amplitudes, normalized
        finally:
            model.train(previous)

    def alignment(self, model, X, y, create_graph=True):
        amplitudes, normalized = self.attribution(model, X, create_graph)
        per_sample = (normalized - self.target).square().mean(1)
        return balanced_mean(per_sample, y), amplitudes, normalized

    def forward(self, model, X, y, reg_scale):
        _, ce, _, _ = self.ce(model, X, y)
        alignment = self.alignment(model, X, y)[0] if reg_scale else ce.new_zeros(())
        return ce + reg_scale * alignment, ce, alignment


def balanced_batches(groups, batch_size, rng):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if batch_size % len(unique):
        raise ValueError("Batch must divide evenly across groups")
    n = batch_size // len(unique)
    members = [np.flatnonzero(groups == g) for g in unique]
    for _ in range(max(1, len(groups) // batch_size)):
        yield np.concatenate(
            [
                rng.choice(members[j], n, replace=len(members[j]) <= n)
                for j in rng.permutation(len(unique))
            ]
        )


class GroupDRO:
    def __init__(self, n_groups, step_size, device):
        self.weights = torch.full((n_groups,), 1 / n_groups, device=device)
        self.step_size = step_size

    def __call__(self, logits, y, group):
        values = F.cross_entropy(logits, y, reduction="none")
        losses = torch.stack(
            [values[group == g].mean() for g in range(len(self.weights))]
        )
        self.weights = F.softmax(
            self.weights.log() + self.step_size * losses.detach(), dim=0
        )
        return (self.weights * losses).sum(), losses


def jtt_weights(prediction, labels, upweight):
    return torch.where(prediction != labels, float(upweight), 1.0).double()


def rsc_loss(model, X, y, p_f, p_b):
    hidden = model.mlp[:-1](X)
    logits = model.mlp[-1](hidden)
    gradient = torch.autograd.grad(
        logits.gather(1, y[:, None]).sum(), hidden, retain_graph=True
    )[0]
    threshold = torch.quantile(gradient.detach(), 1 - p_f, dim=1, keepdim=True)
    feature_mask = gradient.detach() < threshold
    muted = model.mlp[-1](hidden * feature_mask)
    delta = (
        (logits.softmax(1) - muted.softmax(1)).gather(1, y[:, None]).squeeze(1).detach()
    )
    exempt = delta < torch.quantile(delta, 1 - p_b)
    mask = feature_mask | exempt[:, None]
    result = model.mlp[-1](hidden * mask)
    loss = balanced_mean(F.cross_entropy(result, y, reduction="none"), y)
    return loss, mask


def class_margin(logits, labels):
    own = logits.gather(1, labels[:, None]).squeeze(1)
    return own - (logits.sum(1) - own) / (logits.shape[1] - 1)


class MulticlassLAAT(GroupLAAT):
    def alignment(self, model, X, y, create_graph=True):
        previous = model.training
        model.eval()
        try:
            inputs = X.detach().requires_grad_(True)
            margins = class_margin(model(inputs), y)
            grad = torch.autograd.grad(
                margins.sum(), inputs, create_graph=create_graph
            )[0]
            norms = group_norms(inputs * grad)
            normalized = F.normalize(norms, dim=1, eps=1e-12)
            return (
                balanced_mean((normalized - self.target).square().mean(1), y),
                norms,
                normalized,
            )
        finally:
            model.train(previous)


def coral_penalty(features):
    moments = []
    for h in features:
        mean = h.mean(0, keepdim=True)
        centered = h - mean
        moments.append((mean, centered.T @ centered / (len(h) - 1)))
    return torch.stack(
        [
            (m1 - m2).square().mean() + (c1 - c2).square().mean()
            for (m1, c1), (m2, c2) in combinations(moments, 2)
        ]
    ).mean()


def coral_loss(model, x, y, environments, strength):
    hidden = model.mlp[:-1](x)
    logits = model.mlp[-1](hidden)
    masks = [environments == e for e in environments.unique(sorted=True)]
    base = torch.stack(
        [F.cross_entropy(logits[mask], y[mask]) for mask in masks]
    ).mean()
    auxiliary = coral_penalty([hidden[mask] for mask in masks])
    return base + strength * auxiliary, base, auxiliary


def mixed_ce(logits, y, donor_y, coefficient):
    return coefficient * balanced_mean(
        F.cross_entropy(logits, y, reduction="none"), y
    ) + (1 - coefficient) * balanced_mean(
        F.cross_entropy(logits, donor_y, reduction="none"), donor_y
    )


def mixup_loss(model, x, y, alpha, rng):
    coefficient = float(rng.beta(alpha, alpha))
    donor = torch.as_tensor(rng.permutation(len(y)), device=x.device)
    logits = model(coefficient * x + (1 - coefficient) * x[donor])
    loss = mixed_ce(logits, y, y[donor], coefficient)
    return loss, loss, loss.new_zeros(())


def scer_terms(h, y, environment, classifier_weight, classes, environments):
    g = y * environments + environment
    indicator = F.one_hot(g, classes * environments).to(h.dtype)
    counts = indicator.sum(0)
    means = (indicator.T @ h) / counts.clamp_min(1)[:, None]
    means = means.reshape(classes, environments, -1)
    present = counts.reshape(classes, environments) > 0

    def direction(m, valid):
        # Average available pairs within each class/domain, then across those strata.
        pairs = torch.triu(
            torch.ones((m.shape[1], m.shape[1]), device=h.device, dtype=torch.bool),
            diagonal=1,
        )
        mask = valid[:, :, None] & valid[:, None, :] & pairs
        differences = (m[:, :, None, :] - m[:, None, :, :]).abs()
        totals = (differences * mask[..., None]).sum((1, 2))
        n = mask.sum((1, 2))
        exists = n > 0
        d = ((totals / n.clamp_min(1)[:, None]) * exists[:, None]).sum(
            0
        ) / exists.sum().clamp_min(1)
        return d / (1e-6 + d.norm()), exists.sum()

    spur, nspur = direction(means, present)
    core, ncore = direction(means.transpose(0, 1), present.T)
    centered = h - h.mean(0)
    covariance = centered.T @ centered / (len(h) - 1)
    sigma = torch.eye(h.shape[1], device=h.device, dtype=h.dtype) + 0.1 * covariance / (
        1e-6 + covariance.norm()
    )
    weight = classifier_weight.detach()
    beta_norm = ((weight @ sigma) * weight).sum(1).clamp_min(1e-12).sqrt()

    def term(d):
        norm = (d @ sigma @ d).clamp_min(1e-12).sqrt()
        return ((weight @ d) / (1e-6 + beta_norm * norm)).mean() * norm

    return term(spur), -term(core), dict(spurious_strata=nspur, core_strata=ncore)


def balanced_ce(logits, y):
    losses = F.cross_entropy(logits, y, reduction="none")
    return torch.stack([losses[y == c].mean() for c in y.unique()]).mean()
