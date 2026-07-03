"""Canonical domain types shared by future replaceable subsystems."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class RelevanceLabel(StrEnum):
    EXACT = "E"
    SUBSTITUTE = "S"
    COMPLEMENT = "C"
    IRRELEVANT = "I"


class Product(DomainModel):
    product_id: str
    title: str
    description: str | None = None
    brand: str | None = None
    category: str | None = None
    locale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryEvent(DomainModel):
    event_id: str
    query_id: str
    query_text: str
    user_id: str | None = None
    timestamp: datetime
    device: str | None = None
    locale: str
    context: dict[str, Any] = Field(default_factory=dict)


class Candidate(DomainModel):
    query_id: str
    product_id: str
    retrieval_score: float
    rank: int = Field(ge=1)
    source: str
    features: dict[str, float] = Field(default_factory=dict)


class Advertiser(DomainModel):
    advertiser_id: str
    budget: float = Field(ge=0)
    value_per_conversion: float
    risk_tolerance: float
    quality_score: float


class Campaign(DomainModel):
    campaign_id: str
    advertiser_id: str
    product_ids: list[str]
    daily_budget: float = Field(ge=0)
    base_bid: float = Field(ge=0)
    target_categories: list[str]
    target_roi: float | None = None


class AuctionOpportunity(DomainModel):
    opportunity_id: str
    query_event: QueryEvent
    eligible_campaign_ids: list[str]
    candidate_product_ids: list[str]
    timestamp: datetime


class LoggedBanditFeedback(DomainModel):
    event_id: str
    context: np.ndarray
    action: int
    reward: float
    propensity: float = Field(gt=0, le=1)
    position: int | None = Field(default=None, ge=0)
    candidate_set: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
