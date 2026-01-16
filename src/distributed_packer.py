"""
Distributed sequence packing for FSDP training.

This module provides lockstep microbatch synchronization across distributed ranks,
ensuring all ranks perform exactly K accumulation steps per optimizer update.

Algorithm:
1. Each rank packs sequences locally using FFD (First-Fit Decreasing)
2. Ranks agree on global K = max(K_r) using all_reduce
3. Ranks with fewer microbatches split their heaviest bins until reaching K
4. Bins are sorted by descending cost for better load balancing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence
import torch
import torch.distributed as dist
from src.data_utils import JsonlDataset


@dataclass
class Bin:
    """A bin containing sequence indices with bounded total cost."""

    indices: list[int] = field(default_factory=list)
    total_cost: int = 0
    total_tokens: int = 0

    def add(self, idx: int, cost: int, tokens: int):
        self.indices.append(idx)
        self.total_cost += cost
        self.total_tokens += tokens

    def can_fit(self, tokens: int, max_tokens: int) -> bool:
        return self.total_tokens + tokens <= max_tokens

    def __len__(self):
        return len(self.indices)


def cost_fn(length: int, use_quadratic: bool = True) -> int:
    """
    Compute cost proxy for a sequence.

    For transformers, attention dominates, so L^2 is a good proxy.
    For KV-cache / linear attention, L is sufficient.
    """
    if use_quadratic:
        return length * length
    return length


def ffd_pack(
    lengths: Sequence[int],
    max_tokens: int,
    use_quadratic_cost: bool = True,
) -> list[Bin]:
    """
    Pack sequences into bins using First-Fit Decreasing (FFD).

    Args:
        lengths: Sequence lengths (in tokens)
        max_tokens: Maximum tokens per bin
        use_quadratic_cost: Use L^2 as cost proxy (better for attention)

    Returns:
        List of bins, each containing indices of assigned sequences
    """
    n = len(lengths)
    if n == 0:
        return []

    # Create (index, length, cost) tuples and sort by cost descending
    items = [(i, lengths[i], cost_fn(lengths[i], use_quadratic_cost)) for i in range(n)]
    items.sort(key=lambda x: x[2], reverse=True)

    bins: list[Bin] = []

    for idx, length, cost in items:
        # Handle case where a single sequence exceeds max_tokens
        # (we still need to process it, just put it alone in a bin)
        if length > max_tokens:
            new_bin = Bin()
            new_bin.add(idx, cost, length)
            bins.append(new_bin)
            continue

        # First-fit: find the first bin that can accommodate this sequence
        placed = False
        for bin_ in bins:
            if bin_.can_fit(length, max_tokens):
                bin_.add(idx, cost, length)
                placed = True
                break

        if not placed:
            # Create a new bin
            new_bin = Bin()
            new_bin.add(idx, cost, length)
            bins.append(new_bin)

    return bins


def split_bin(bin_: Bin, lengths: Sequence[int], max_tokens: int, use_quadratic_cost: bool = True) -> tuple[Bin, Bin]:
    """
    Split a bin into two bins while respecting max_tokens constraint.

    Sorts sequences in the bin by cost descending, then greedily moves
    sequences to a new bin until it's about half the cost.

    Args:
        bin_: The bin to split
        lengths: Original sequence lengths array
        max_tokens: Maximum tokens per bin
        use_quadratic_cost: Use L^2 as cost proxy

    Returns:
        Tuple of (bin1, bin2) where both respect max_tokens
    """
    if len(bin_) <= 1:
        # Cannot split a single-item bin, return original and empty
        return bin_, Bin()

    # Get items with their costs
    items = [(idx, lengths[idx], cost_fn(lengths[idx], use_quadratic_cost)) for idx in bin_.indices]
    items.sort(key=lambda x: x[2], reverse=True)

    target_cost = bin_.total_cost // 2

    bin1 = Bin()
    bin2 = Bin()

    for idx, length, cost in items:
        # Try to balance costs while respecting token limits
        if bin1.total_cost < target_cost and bin1.can_fit(length, max_tokens):
            bin1.add(idx, cost, length)
        elif bin2.can_fit(length, max_tokens):
            bin2.add(idx, cost, length)
        else:
            # Fallback: put in bin1 even if over target
            bin1.add(idx, cost, length)

    return bin1, bin2


def split_bins_to_reach_k(
    bins: list[Bin], k: int, lengths: Sequence[int], max_tokens: int, use_quadratic_cost: bool = True
) -> list[Bin]:
    """
    Split the heaviest bins until we have exactly k bins.

    Args:
        bins: Current list of bins
        k: Target number of bins
        lengths: Original sequence lengths
        max_tokens: Maximum tokens per bin
        use_quadratic_cost: Use L^2 as cost proxy

    Returns:
        List of exactly k bins
    """
    while len(bins) < k:
        if not bins:
            # Edge case: create empty/padding bins
            bins.append(Bin())
            continue

        # Find the bin with the highest cost that can be split
        max_cost_idx = -1
        max_cost = -1
        for i, bin_ in enumerate(bins):
            if len(bin_) > 1 and bin_.total_cost > max_cost:
                max_cost = bin_.total_cost
                max_cost_idx = i

        if max_cost_idx == -1:
            # No bins can be split, add empty padding bins
            bins.append(Bin())
        else:
            # Split the heaviest bin
            bin_to_split = bins.pop(max_cost_idx)
            bin1, bin2 = split_bin(bin_to_split, lengths, max_tokens, use_quadratic_cost)
            bins.append(bin1)
            if len(bin2) > 0:
                bins.append(bin2)
            else:
                # The split produced an empty bin2, add a padding bin
                bins.append(Bin())

    return bins


def synchronize_microbatch_count(local_k: int, device: torch.device) -> int:
    """
    Synchronize microbatch count across all ranks using all_reduce(MAX).

    Args:
        local_k: Number of microbatches this rank has
        device: Torch device for tensor operations

    Returns:
        Global K = max across all ranks
    """
    k_tensor = torch.tensor([local_k], dtype=torch.int64, device=device)
    dist.all_reduce(k_tensor, op=dist.ReduceOp.MAX)
    return k_tensor.item()


def pack_for_distributed_training(
    lengths: Sequence[int],
    max_tokens_per_rank: int,
    device: torch.device,
    use_quadratic_cost: bool = True,
) -> list[list[int]]:
    """
    Pack sequences for distributed training with lockstep microbatches.

    This function:
    1. Packs sequences into bins using FFD
    2. Synchronizes K across ranks
    3. Splits bins if needed to reach K
    4. Sorts bins by descending cost for better load balancing

    Args:
        lengths: Sequence lengths for this rank's data
        max_tokens_per_rank: Maximum tokens per microbatch per rank
        device: Torch device for distributed ops
        use_quadratic_cost: Use L^2 as cost proxy

    Returns:
        List of K microbatches, each containing sequence indices.
        Empty list [] indicates a padding microbatch (no real data).
    """
    # Step 1: Local FFD packing
    bins = ffd_pack(lengths, max_tokens_per_rank, use_quadratic_cost)
    local_k = len(bins) if bins else 1  # At least 1 for empty datasets

    # Step 2: Synchronize K across ranks
    global_k = synchronize_microbatch_count(local_k, device)

    # Step 3: Split bins if we have fewer than K
    if len(bins) < global_k:
        bins = split_bins_to_reach_k(bins, global_k, lengths, max_tokens_per_rank, use_quadratic_cost)

    # Step 4: Sort bins by descending cost for better load balancing
    bins.sort(key=lambda b: b.total_cost, reverse=True)

    # Convert to index lists
    result = []
    for bin_ in bins:
        if len(bin_) > 0:
            result.append(bin_.indices)
        else:
            result.append([])  # Padding microbatch

    return result


class DistributedPackingSampler:
    """
    A sampler that yields microbatch indices with lockstep synchronization.

    This sampler ensures all ranks produce exactly K microbatches per epoch,
    preventing FSDP collective mismatches.
    """

    def __init__(
        self,
        dataset: JsonlDataset,
        max_tokens_per_rank: int,
        device: torch.device,
        use_quadratic_cost: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            dataset: Dataset with __getitem__ returning dict with 'seq_len' key
            max_tokens_per_rank: Maximum tokens per microbatch
            device: Torch device for distributed ops
            use_quadratic_cost: Use L^2 as cost proxy
            shuffle: Whether to shuffle the dataset before packing
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.max_tokens_per_rank = max_tokens_per_rank
        self.device = device
        self.use_quadratic_cost = use_quadratic_cost
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._microbatches: list[list[int]] | None = None

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
        self._microbatches = None  # Invalidate cache

    def _compute_microbatches(self) -> list[list[int]]:
        """Compute microbatches with distributed synchronization."""
        n = len(self.dataset)

        # Get all lengths
        lengths = [self.dataset[i]["seq_len"] for i in range(n)]

        # Optionally shuffle (using same seed across ranks for reproducibility)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(n).tolist()
            # Reorder lengths according to shuffle
            shuffled_lengths = [lengths[i] for i in indices]
        else:
            indices = list(range(n))
            shuffled_lengths = lengths

        # Pack with distributed sync
        microbatches = pack_for_distributed_training(
            shuffled_lengths,
            self.max_tokens_per_rank,
            self.device,
            self.use_quadratic_cost,
        )

        # Map back to original dataset indices
        result = []
        for batch in microbatches:
            if batch:
                result.append([indices[i] for i in batch])
            else:
                result.append([])  # Padding batch

        return result

    @property
    def microbatches(self) -> list[list[int]]:
        if self._microbatches is None:
            self._microbatches = self._compute_microbatches()
        return self._microbatches

    def __len__(self) -> int:
        return len(self.microbatches)

    def __iter__(self):
        for batch_indices in self.microbatches:
            yield batch_indices


@dataclass
class PackedBatch:
    """A batch with metadata about whether it's a padding batch."""

    batch: dict | None
    is_padding: bool = False
    indices: list[int] = field(default_factory=list)


@dataclass
class AccumulationWindow:
    """A group of microbatches to accumulate before taking an optimizer step."""

    microbatches: list[PackedBatch]
    num_accumulation_steps: int  # K for this window

    def __len__(self):
        return len(self.microbatches)


def create_distributed_grpo_dataloader(
    dataset: JsonlDataset,
    max_tokens_per_rank: int,
    device: torch.device,
    collate_fn,
    accumulation_steps: int | None = None,
    use_quadratic_cost: bool = True,
    shuffle: bool = False,
    seed: int = 42,
):
    """
    Create a DataLoader with distributed packing for GRPO training.

    This DataLoader ensures all ranks perform the same number of microbatches,
    preventing FSDP collective mismatches.

    For padding batches (empty bins), we yield a batch with a single repeated
    sequence from the dataset to ensure FSDP collectives work correctly.
    The is_padding flag indicates this batch should not contribute to gradients.

    Args:
        dataset: JsonlDataset or similar with 'seq_len' in items
        max_tokens_per_rank: Maximum tokens per microbatch per rank
        device: Torch device for distributed ops
        collate_fn: Collation function for batches
        accumulation_steps: Number of microbatches to accumulate before optimizer step.
                          If None, yields individual microbatches.
                          If set, yields AccumulationWindow objects.
        use_quadratic_cost: Use L^2 as cost proxy
        shuffle: Whether to shuffle before packing
        seed: Random seed

    Returns:
        Generator that yields PackedBatch or AccumulationWindow objects
    """
    sampler = DistributedPackingSampler(
        dataset=dataset,
        max_tokens_per_rank=max_tokens_per_rank,
        device=device,
        use_quadratic_cost=use_quadratic_cost,
        shuffle=shuffle,
        seed=seed,
    )

    # Collect all microbatches
    all_microbatches = []
    for batch_indices in sampler:
        if not batch_indices:
            # Padding batch - use the first item from dataset as a dummy
            # This ensures FSDP collectives still work
            dummy_item = dataset._create_empty_entry()
            batch = collate_fn([dummy_item])
            all_microbatches.append(PackedBatch(batch=batch, is_padding=True, indices=[]))
        else:
            # Real batch
            items = [dataset[i] for i in batch_indices]
            batch = collate_fn(items)
            all_microbatches.append(PackedBatch(batch=batch, is_padding=False, indices=batch_indices))

    # If no accumulation, yield individual microbatches
    if accumulation_steps is None:
        for mb in all_microbatches:
            yield mb
    else:
        # Group into accumulation windows
        for i in range(0, len(all_microbatches), accumulation_steps):
            window = all_microbatches[i : i + accumulation_steps]
            # Handle last window: might have fewer than accumulation_steps
            # Pad with dummy batches if needed to maintain lockstep
            while len(window) < accumulation_steps:
                if len(dataset) > 0:
                    dummy_item = dataset[0]
                    batch = collate_fn([dummy_item])
                    window.append(PackedBatch(batch=batch, is_padding=True, indices=[]))
                else:
                    window.append(PackedBatch(batch=None, is_padding=True, indices=[]))

            yield AccumulationWindow(microbatches=window, num_accumulation_steps=accumulation_steps)
