"""Database tables.

A Submission is a raw request exactly as it was sent in; it is never edited.
A Feature is the consolidated, canonical request. Every triaged submission
belongs to exactly one feature. Because the original submissions are kept, any
merge can be undone.
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


def current_utc_time() -> datetime:
    return datetime.now(UTC)


def enum_stored_by_value(enum_class: type[StrEnum]) -> Enum:
    """A column type that stores an enum's value ("pending"), not its name ("PENDING")."""

    def list_enum_values(enum_type: type[StrEnum]) -> list[str]:
        return [member.value for member in enum_type]

    return Enum(
        enum_class,
        native_enum=False,
        validate_strings=True,
        values_callable=list_enum_values,
    )


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[CategoryStatus] = mapped_column(
        enum_stored_by_value(CategoryStatus), default=CategoryStatus.PROPOSED
    )
    created_at: Mapped[datetime] = mapped_column(default=current_utc_time)

    features: Mapped[list["Feature"]] = relationship(back_populates="category")


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    status: Mapped[FeatureStatus] = mapped_column(
        enum_stored_by_value(FeatureStatus), default=FeatureStatus.TRIAGED
    )
    created_at: Mapped[datetime] = mapped_column(default=current_utc_time)
    updated_at: Mapped[datetime] = mapped_column(default=current_utc_time, onupdate=current_utc_time)
    # Set when a reviewer deletes the feature. Archived features are hidden but
    # kept, so they can be restored.
    archived_at: Mapped[datetime | None]

    category: Mapped[Category | None] = relationship(back_populates="features")
    # Every submission ever attached, including archived ones.
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="feature", order_by="Submission.created_at"
    )

    @property
    def is_archived(self) -> bool:
        return self.archived_at is not None

    @property
    def active_submissions(self) -> list["Submission"]:
        """Submissions that haven't been archived, oldest first."""
        active = []
        for submission in self.submissions:
            if not submission.is_archived:
                active.append(submission)
        return active


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[SubmissionStatus] = mapped_column(
        enum_stored_by_value(SubmissionStatus), default=SubmissionStatus.PENDING
    )
    feature_id: Mapped[int | None] = mapped_column(ForeignKey("features.id"))
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(default=current_utc_time)
    # Set when a reviewer deletes this submission from its feature. Archived
    # submissions don't count towards the feature and aren't used in duplicate
    # search, but are kept so they can be restored.
    archived_at: Mapped[datetime | None]

    feature: Mapped[Feature | None] = relationship(back_populates="submissions")
    proposals: Mapped[list["Proposal"]] = relationship(back_populates="submission")

    @property
    def is_archived(self) -> bool:
        return self.archived_at is not None


class Proposal(Base):
    """An AI suggestion that is waiting for, or has received, a human decision."""

    __tablename__ = "proposals"

    id: Mapped[int] = mapped_column(primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[ProposalType] = mapped_column(enum_stored_by_value(ProposalType))
    status: Mapped[ProposalStatus] = mapped_column(
        enum_stored_by_value(ProposalStatus), default=ProposalStatus.PENDING
    )
    # Taxonomy proposals are not tied to a single submission, so this can be empty.
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

    created_at: Mapped[datetime] = mapped_column(default=current_utc_time)

    submission: Mapped[Submission | None] = relationship(back_populates="proposals")
