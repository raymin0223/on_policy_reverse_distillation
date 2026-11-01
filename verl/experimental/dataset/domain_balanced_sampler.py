"""Deterministic equal-per-domain sampling for multi-teacher training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sized

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class DomainBalancedSampler(AbstractSampler):
    """Shuffle domain pools independently and emit equal-domain batches.

    By default, each batch is emitted in configured domain blocks for backward
    compatibility.  ``shuffle_within_batch`` deterministically mixes those
    blocks; the multi-teacher trainer subsequently partitions rows by
    ``data_source`` before dispatching them to teacher-owning DP ranks.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        sampler_config = data_config.get("domain_balanced_sampler", {})
        self.domains = tuple(str(value) for value in sampler_config.get("domains", []))
        if not self.domains:
            raise ValueError("data.domain_balanced_sampler.domains must be non-empty")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError(f"domain list contains duplicates: {self.domains}")

        train_batch_size = int(data_config.get("train_batch_size", 0))
        if train_batch_size <= 0 or train_batch_size % len(self.domains) != 0:
            raise ValueError(
                "data.train_batch_size must be positive and divisible by the number of domains; "
                f"got batch={train_batch_size}, domains={len(self.domains)}"
            )
        inferred_per_domain = train_batch_size // len(self.domains)
        self.samples_per_domain = int(sampler_config.get("samples_per_domain", inferred_per_domain))
        if self.samples_per_domain != inferred_per_domain:
            raise ValueError(
                "samples_per_domain must equal train_batch_size / number_of_domains; "
                f"got {self.samples_per_domain} vs {inferred_per_domain}"
            )
        self.shuffle_within_batch = bool(sampler_config.get("shuffle_within_batch", False))

        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None or "data_source" not in dataframe.column_names:
            raise ValueError("DomainBalancedSampler requires dataset.dataframe['data_source']")
        pools: dict[str, list[int]] = defaultdict(list)
        for index, domain in enumerate(dataframe["data_source"]):
            pools[str(domain)].append(index)
        unexpected = sorted(set(pools) - set(self.domains))
        missing = [domain for domain in self.domains if not pools.get(domain)]
        if unexpected or missing:
            raise ValueError(
                f"dataset domains do not match configured domains; missing={missing}, unexpected={unexpected}"
            )
        self._pools = {domain: np.asarray(pools[domain], dtype=np.int64) for domain in self.domains}
        self._batches_per_epoch = min(len(pool) // self.samples_per_domain for pool in self._pools.values())
        if self._batches_per_epoch <= 0:
            raise ValueError("each domain needs at least one full per-domain batch")

        self.seed = int(data_config.get("seed", 0) or 0)
        self.epoch = 0
        self.position = 0
        self._epoch_order: list[int] | None = None

    def _build_epoch_order(self) -> list[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled = {domain: rng.permutation(pool) for domain, pool in self._pools.items()}
        order: list[int] = []
        for batch_index in range(self._batches_per_epoch):
            start = batch_index * self.samples_per_domain
            stop = start + self.samples_per_domain
            batch: list[int] = []
            for domain in self.domains:
                batch.extend(shuffled[domain][start:stop].tolist())
            if self.shuffle_within_batch:
                batch = rng.permutation(batch).tolist()
            order.extend(batch)
        return order

    def __iter__(self) -> Iterator[int]:
        if self._epoch_order is None:
            self._epoch_order = self._build_epoch_order()
        while self.position < len(self._epoch_order):
            index = self._epoch_order[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self.position = 0
        self._epoch_order = None

    def __len__(self) -> int:
        return self._batches_per_epoch * self.samples_per_domain * len(self.domains)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        epoch = int(state_dict.get("epoch", 0))
        position = int(state_dict.get("position", 0))
        if epoch < 0 or position < 0 or position > len(self):
            raise ValueError(f"invalid sampler state: epoch={epoch}, position={position}")
        self.epoch = epoch
        self.position = position
        self._epoch_order = self._build_epoch_order() if position else None
