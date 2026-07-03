import random

import numpy as np

from adaptirank.common.reproducibility import seed_everything


def test_seed_everything_is_deterministic() -> None:
    seed_everything(42)
    first = (random.random(), np.random.random())
    seed_everything(42)
    second = (random.random(), np.random.random())
    assert first == second
