import torch


class Corruption:
    def __init__(self, groups, q, seed, device, track=True):
        import torch

        self.groups, self.q, self.track = groups, q, track
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.q_tensor = torch.tensor(q, device=device)[:, None]
        columns = [0] * sum(map(len, groups))
        for j, indices in enumerate(groups):
            for index in indices:
                columns[index] = j
        self.column_groups = torch.tensor(columns, device=device)
        self.counts = torch.zeros((len(groups), 4), dtype=torch.float64, device=device)

    def __call__(self, X):
        import torch

        n = len(X)
        shape = (len(self.groups), n)
        selected = (
            torch.rand(shape, device=X.device, generator=self.generator) < self.q_tensor
        )
        # Pack selected destinations and randomly ordered selected donors first.
        # Scatter a bijection within each selected subset, leaving other rows fixed.
        destinations = torch.argsort((~selected).int(), dim=1, stable=True)
        random_keys = torch.rand(shape, device=X.device, generator=self.generator)
        donors = torch.argsort(random_keys.masked_fill(~selected, 2.0), dim=1)
        sources = torch.where(selected.gather(1, destinations), donors, destinations)
        permutation = torch.empty_like(destinations).scatter_(1, destinations, sources)
        out = X.detach()[
            permutation[self.column_groups].T, torch.arange(X.shape[1], device=X.device)
        ]
        if self.track:
            changed = torch.zeros((n, len(self.groups)), device=X.device)
            changed.scatter_add_(
                1, self.column_groups.expand(n, -1), (out != X).float()
            )
            self.counts[:, 0] += n
            self.counts[:, 1] += selected.sum(1)
            self.counts[:, 2] += (permutation != torch.arange(n, device=X.device)).sum(
                1
            )
            self.counts[:, 3] += (changed > 0).sum(0)
        return out

    def report(self, names):
        result = []
        for name, q, counts in zip(names, self.q, self.counts.cpu().tolist()):
            n, selected, moved, changed = counts
            result.append(
                dict(
                    feature=name,
                    q=q,
                    observations=n,
                    selected_fraction=selected / max(n, 1),
                    donor_moved_fraction=moved / max(n, 1),
                    value_changed_fraction=changed / max(n, 1),
                )
            )
        return result


class SharedDonorCorruption(Corruption):
    def draw_assignment(self, n, device):
        selected = (
            torch.rand((len(self.groups), n), device=device, generator=self.generator)
            < self.q_tensor
        )
        donor = torch.randperm(n, device=device, generator=self.generator)
        return selected, donor

    def __call__(self, X):
        n = len(X)
        selected, donor = self.draw_assignment(n, X.device)
        rows = torch.arange(n, device=X.device)
        sources = torch.where(selected, donor[None, :], rows[None, :])
        out = X.detach()[
            sources[self.column_groups].T, torch.arange(X.shape[1], device=X.device)
        ]
        if self.track:
            changed = torch.zeros((n, len(self.groups)), device=X.device)
            changed.scatter_add_(
                1, self.column_groups.expand(n, -1), (out != X).float()
            )
            self.counts[:, 0] += n
            self.counts[:, 1] += selected.sum(1)
            self.counts[:, 2] += (sources != rows).sum(1)
            self.counts[:, 3] += (changed > 0).sum(0)
        return out
