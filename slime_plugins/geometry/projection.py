"""Low-memory deterministic projections used by geometry observations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch


def stable_seed(name: str, seed: int = 0) -> int:
    """Return a process-independent 31-bit seed for ``name``."""

    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _hash_indices(indices: torch.Tensor, base_seed: int) -> torch.Tensor:
    """Mix non-negative indices into uniformly spread unsigned 32-bit values.

    A plain linear congruential map is especially poor when ``dim`` is a power
    of two: bucket and sign bits become correlated and repeat in a short,
    structured pattern.  This finalizer is the public-domain 32-bit mixer used
    by several integer hash tables.  Masking after each multiplication makes
    the intended unsigned wraparound explicit while retaining fast tensor-only
    execution on CPU and CUDA.
    """

    mask = 0xFFFFFFFF
    values = torch.bitwise_and(indices + int(base_seed), mask)
    values = torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))
    values = torch.bitwise_and(values * 0x7FEB352D, mask)
    values = torch.bitwise_xor(values, torch.bitwise_right_shift(values, 15))
    values = torch.bitwise_and(values * 0x846CA68B, mask)
    return torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))


@torch.no_grad()
def count_sketch_many(
    tensors: Sequence[torch.Tensor],
    dim: int,
    *,
    seed: int,
    name: str,
    chunk_size: int = 1_048_576,
) -> list[torch.Tensor]:
    """Project equal-shaped tensors with one shared CountSketch hash stream.

    The map is generated arithmetically in chunks, so memory is O(``dim`` +
    ``chunk_size``) instead of O(number of parameters * ``dim``).  Independent
    parameter/rank names receive independent hash streams, which makes sketches
    additive across tensor/pipeline-parallel shards. Sharing hashes makes the
    common weight+gradient snapshot materially cheaper without changing either
    projection.
    """

    if dim <= 0:
        raise ValueError("CountSketch dimension must be positive.")
    if chunk_size <= 0:
        raise ValueError("CountSketch chunk_size must be positive.")

    if not tensors:
        return []
    flats = []
    for tensor in tensors:
        value = tensor.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        if hasattr(value, "_local_tensor"):
            value = value._local_tensor
        flats.append(value.reshape(-1))
    numel = flats[0].numel()
    device = flats[0].device
    if any(flat.numel() != numel or flat.device != device for flat in flats[1:]):
        raise ValueError("CountSketch tensors must have the same number of elements and device.")
    results = [torch.zeros(dim, dtype=torch.float32, device=device) for _ in flats]
    base_seed = stable_seed(name, seed)

    for start in range(0, numel, chunk_size):
        stop = min(start + chunk_size, numel)
        indices = torch.arange(start, stop, dtype=torch.int64, device=device)
        hashes = _hash_indices(indices, base_seed)
        buckets = torch.remainder(hashes, dim)
        # Use a high hash bit for the Rademacher sign.  For the usual
        # power-of-two projection sizes this is disjoint from the bucket bits.
        sign_bits = torch.bitwise_and(torch.bitwise_right_shift(hashes, 31), 1)
        signs = sign_bits.to(torch.float32).mul_(2).sub_(1)
        for result, flat in zip(results, flats, strict=True):
            result.scatter_add_(0, buckets, flat[start:stop].to(torch.float32) * signs)
    return results


@torch.no_grad()
def count_sketch(
    tensor: torch.Tensor,
    dim: int,
    *,
    seed: int,
    name: str,
    chunk_size: int = 1_048_576,
) -> torch.Tensor:
    """Project one tensor with deterministic signed hashing."""

    return count_sketch_many([tensor], dim, seed=seed, name=name, chunk_size=chunk_size)[0]
