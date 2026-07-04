"""Pointwise and LambdaMART training helpers for M3."""

from __future__ import annotations

import time
from importlib import import_module
from typing import Any

import numpy as np
import polars as pl

from adaptirank.ranking.features import FEATURE_COLUMNS


def feature_matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(pl.col(FEATURE_COLUMNS).cast(pl.Float32)).to_numpy()


def judged(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("judgment_status") == "judged").sort("query_key", "product_key")


def query_groups(frame: pl.DataFrame) -> list[int]:
    return frame.group_by("query_key", maintain_order=True).len().get_column("len").to_list()


def train_pointwise(train: pl.DataFrame, params: dict[str, Any], seed: int) -> tuple[Any, float]:
    ensemble: Any = import_module("sklearn.ensemble")
    model = ensemble.HistGradientBoostingRegressor(random_state=seed, **params)
    started = time.perf_counter()
    model.fit(feature_matrix(train), train.get_column("relevance_grade").to_numpy())
    return model, time.perf_counter() - started


def train_lambdamart(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    params: dict[str, Any],
    *,
    seed: int,
    early_stopping_rounds: int,
) -> tuple[Any, float]:
    lightgbm: Any = import_module("lightgbm")
    model = lightgbm.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7],
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=1,
        **params,
    )
    started = time.perf_counter()
    model.fit(
        feature_matrix(train),
        train.get_column("relevance_grade").to_numpy(),
        group=query_groups(train),
        eval_set=[
            (feature_matrix(validation), validation.get_column("relevance_grade").to_numpy())
        ],
        eval_group=[query_groups(validation)],
        eval_at=[5, 10],
        callbacks=[lightgbm.early_stopping(early_stopping_rounds, verbose=False)],
        feature_name=list(FEATURE_COLUMNS),
    )
    return model, time.perf_counter() - started


def predict(model: Any, frame: pl.DataFrame) -> np.ndarray:
    matrix = feature_matrix(frame)
    if hasattr(model, "booster_"):
        return np.asarray(model.booster_.predict(matrix), dtype=np.float32)
    return np.asarray(model.predict(matrix), dtype=np.float32)
