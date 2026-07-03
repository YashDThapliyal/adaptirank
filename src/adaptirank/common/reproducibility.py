"""Deterministic seeding helpers."""

from __future__ import annotations

import random
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed installed numerical runtimes and return what was configured."""

    random.seed(seed)
    np.random.seed(seed)
    seeded: dict[str, Any] = {"python": seed, "numpy": seed, "torch": False}
    if find_spec("torch") is not None:
        torch: Any = import_module("torch")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True
    return seeded
