"""Deterministic seeding helpers."""

from __future__ import annotations

import random
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed installed numerical runtimes and return what was configured."""

    random.seed(seed)
    np.random.seed(seed)
    seeded: dict[str, Any] = {"python": seed, "numpy": seed, "torch": False}
    # On macOS, importing PyTorch after faiss-cpu has initialized its bundled OpenMP runtime
    # aborts the interpreter. Normal dense pipelines call this before loading FAISS. If an
    # embedding/index-only test has already imported FAISS, skip optional torch seeding safely.
    if find_spec("torch") is not None and "faiss" not in sys.modules:
        torch: Any = import_module("torch")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True
    elif find_spec("torch") is not None:
        seeded["torch"] = "skipped_after_faiss_import"
    return seeded
