"""Seeded RNG streams. Only Random.random() is guaranteed stable across Python versions,
so every helper is built on it (see docs.python.org/3/library/random.html#notes-on-reproducibility)."""
from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


class Rng:
    def __init__(self, key: str, stage: str):
        digest = hashlib.sha256(f"{key}|{stage}".encode()).digest()
        self._r = random.Random(int.from_bytes(digest, "big"))

    def random(self) -> float:
        return self._r.random()

    def randint(self, a: int, b: int) -> int:
        """Uniform integer in [a, b]."""
        return a + min(int(self.random() * (b - a + 1)), b - a)

    def chance(self, p: float) -> bool:
        return self.random() < p

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self.randint(0, len(seq) - 1)]

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        pool = list(seq)
        self.shuffle(pool)
        return pool[:k]

    def weighted(self, items: Sequence[T], weights: Sequence[float]) -> T:
        x = self.random() * sum(weights)
        for item, w in zip(items, weights):
            x -= w
            if x < 0:
                return item
        return items[-1]
