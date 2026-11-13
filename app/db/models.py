"""ORM models.

Submissions are raw entries, stored verbatim and never edited. Features are the
consolidated, canonical requests; every triaged submission belongs to exactly
one. Keeping the originals means any merge can be undone.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Enum, Float, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.domain.enums import (
    CategoryStatus,
    FeatureStatus,
    ProposalStatus,
    ProposalType,
    SubmissionStatus,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _enum(enum_cls: type[StrEnum]) -> Enum:
    """Store enums by value ("pending"), not by member name ("PENDING")."""
    return Enum(
        enum_cls,
        native_enum=False,
        validate_strings=True,
        values_callable=lambda e: [m.value for m in e],
    )


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[CategoryStatus] = mapped_column(
        _enum(CategoryStatus), default=CategoryStatus.PROPOSED
    )
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    features: Mapped[list["Feature"]] = relationship(back_populates="category")


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    status: Mapped[FeatureStatus] = mapped_column(
        _enum(FeatureStatus), default=FeatureStatus.TRIAGED
    )
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    category: Mapped[Category | None] = relationship(back_populates="features")
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="feature", order_by="Submission.created_at"
    )


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[SubmissionStatus] = mapped_column(
        _enum(SubmissionStatus), default=SubmissionStatus.PENDING
    )
    feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"))
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    feature: Mapped[Feature | None] = relationship(back_populates="submissions")
    proposals: Mapped[list["Proposal"]] = relationship(back_populates="submission")


class Proposal(Base):
    """An AI suggestion awaiting, or holding, a human decision."""

    __tablename__ = "proposals"

    id: Mapped[int] = mapped_column(primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[ProposalType] = mapped_column(_enum(ProposalType))
    status: Mapped[ProposalStatus] = mapped_column(
        _enum(ProposalStatus), default=ProposalStatus.PENDING
    )
    # Taxonomy proposals are not tied to a single submission.
    submission_id: Mapped[int | None] = mapped_column(ForeignKey("submissions.id"))
    feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"))

    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float)
    reasoning: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    ai_event_id: Mapped[str | None] = mapped_column(String(36))

    decision_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    decided_by: Mapped[str | None] = mapped_column(String(100))
    decided_at: Mapped[datetime | None]
    decision_event_id: Mapped[str | None] = mapped_column(String(36))

    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    submission: Mapped[Submission | None] = relationship(back_populates="proposals")
