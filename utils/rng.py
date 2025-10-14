# utils/rng.py

"""
Design goals
--------------
Reproducible experiments: one master seed -> stable substreams per componenets.
Stateless fan-out: same (parent, tag, extras) always yields same child seed.
No global state; explicit np.random.Generator everywhere.
Hashing-based seed derivation avoids correlations between substreams.

Complexity
------------
Seed derivation: O(L) where L is length of concatenated tag/extra bytes.
RNG ops delegate to NumPy Generator (PCG64): O(1) amortized per sample.

Usage
------------
>>> rng = DeterministicRNG(12345, stream="exp")
>>> items_rng = rng.child("items", 0)               #catalog 0
>>> ilp_rng = rng.child("ilp")
>>> g = items_rng.gen                               #np.random.Generator
>>> g.random()
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np

SeedLike = Union[int, np.integer]

def _to_bytes(x: Any) -> bytes:
    """
    Best-effort stable serialization to bytes for hashing.
    Supports ints, floats, strings, bytes, tuples/lists, and numpy scalars.
    """
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, (np.integer,)):
        return int(x).to_bytes(8, "little", signed=False)
    if isinstance(x, int):
        return x.to_bytes(8, "little", signed=False)
    if isinstance(x, float):
        return struct.pack("<d", x)
    if isinstance(x, str):
        return x.encode("utf-8", errors="strict")
    if isinstance(x, (tuple, list)):
        out = bytearray()
        out.extend(struct.pack("<I", len(x)))
        for el in x:
            b = _to_bytes(el)
            out.extend(struct.pack("<I", len(b)))
            out.extend(b)
        return bytes(out)
    # Fallback to repr (stable enough for tags; avoid for large arrays)
    return repr(x).encode("utf-8", errors="ignore")

def hash_uint64(*parts: Any) -> np.uint64:
    """
    Deterministically hash arbitrary parts into a single uint64.
    We use BLAKE2b-128 (fast, high quality) then fold to 64 bits.
    """
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        b = _to_bytes(p)
        h.update(struct.pack("<I", len(b)))
        h.update(b)
    digest = h.digest()
    # Take lower 8 bytes as little-endian uint64
    return np.frombuffer(digest[:8], dtype=np.uint64)[0]

def derive_seed(parent_seed: SeedLike, tag: str, *extras: Any) -> int:
    """
    Derive a child seed from a parent seed and a tag (+ optional extras).
    The output is a python int in [0, 2**64-1], acceptable by np.random.PCG64.
    """
    u = hash_uint64(int(parent_seed), tag, extras)
    return int(u)

def make_generator(seed: SeedLike) ->  np.random.Generator:
    """Create a fresh NumPy Generator (PCG64) from a 64-bit seed."""
    return np.random.Generator(np.random.PCG64(seed))

@dataclass(frozen=True)
class DeterministicRNG:
    """
    Lightweight handle that (a) stores a seed and (b) lazily provides a Generator. 
    -Frozen dataclass so instances are hashable and safe to pass around.
    -Child streams are generated via 'child(tag, *extras)'.
    """
    seed: int
    stream: str = "root"

    @property
    def gen(self) -> np.random.Generator:
        # Lazily instantiate a Generator each access; cheap & side-effect free.
        return make_generator(self.seed)
    
    def child(self, tag: str, *extras: Any) -> "DeterministicRNG":
        """
        Fan out to a deterministic child stream.

        >>> rng = DeterministicRNG(42)
        >>> a = rng.child("items", 0)       #catalog index 0
        """
        return DeterministicRNG(seed=derive_seed(self.seed, tag, *extras), stream=f"{self.stream}/{tag}")
    
    def fork(self, count: int, tag: str = "fork") -> List["DeterministicRNG"]:
        """
        Produce 'count' independent child streams:
        [child(tag, 0), child(tag, 1), ...]
        """
        return [self.child(tag, i) for i in range(count)]
    
    # Convenience wrappers - these call into the underlying generator and 
    # keep method names intuitive for callers.
    def randint(self, low: int, high: int | None = None, size: int | Tuple[int, ...] | None = None) -> np.ndarray:
        return self.gen.integers(low, high=high, size=size)
    
    def choice(self, a: Union[int, Sequence], size: int | Tuple[int, ...] | None = None,
               replace: bool = True, p: np.ndarray | None = None) -> np.ndarray:
        return self.gen.choice(a, size=size, replace=replace, p=p)
    
    def permutation(self, x: Union[int, Sequence]) -> np.ndarray:
        return self.gen.permutation(x)
    
    def shuffle_inplace(self, x: np.ndarray) -> None:
        self.gen.shuffle(x)