"""Pipeline stages as plain functions.

No FastAPI and no decision logging here: the services layer handles both. The
eval runner can call these directly. Model output is validated against the
inputs, so a hallucinated ID never reaches a proposal.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.llm import LLM, LLMResult
from app.ai.prompts import Prompt, load_prompt
from app.ai.vectors import from_blob
from app.db.models import Category, Feature, Submission
from app.domain.enums import CategoryStatus
from app.pipeline.schemas import (
    CategoryAssignment,
    DuplicateJudgment,
    SummaryUpdate,
    TaxonomyProposal,
)

PROMPT_VERSION = "v1"
EXAMPLES_PER_CATEGORY = 3


def prompt_for(name: str) -> Prompt:
    return load_prompt(name, PROMPT_VERSION)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _request_block(title: str, description: str) -> str:
    return f"<request>\nTitle: {title}\nDescription: {description}\n</request>"


# --- Stage 1: duplicate candidates (embedding search) -----------------------


@dataclass(frozen=True)
class Candidate:
    feature_id: int
    similarity: float
    submission_id: int  # the most similar submission within that feature


def find_candidates(
    session: Session,
    embedding: np.ndarray,
    *,
    k: int,
    threshold: float,
    exclude_submission_id: int | None = None,
) -> list[Candidate]:
    """Search every triaged submission, then keep each feature's best match.

    Searching submissions rather than feature summaries means a paraphrase of
    any one past request still matches.
    """
    rows = session.execute(
        select(Submission.id, Submission.feature_id, Submission.embedding).where(
            Submission.feature_id.is_not(None), Submission.embedding.is_not(None)
        )
    ).all()

    query = embedding / (np.linalg.norm(embedding) or 1.0)
    best: dict[int, Candidate] = {}
    for submission_id, feature_id, blob in rows:
        if submission_id == exclude_submission_id:
            continue
        vector = from_blob(blob)
        similarity = float(vector @ query / (np.linalg.norm(vector) or 1.0))
        if similarity < threshold:
            continue
        current = best.get(feature_id)
        if current is None or similarity > current.similarity:
            best[feature_id] = Candidate(feature_id, similarity, submission_id)

    return sorted(best.values(), key=lambda c: c.similarity, reverse=True)[:k]


# --- Stage 2: duplicate judgment (LLM) ---------------------------------------


def judge_duplicate(
    llm: LLM, title: str, description: str, candidates: Sequence[tuple[Candidate, Feature]]
) -> tuple[LLMResult[DuplicateJudgment], str]:
    prompt = prompt_for("duplicate_judge")
    candidate_blocks = "\n".join(
        f'<candidate feature_id="{feature.id}" similarity="{candidate.similarity:.2f}">\n'
        f"Title: {feature.title}\nSummary: {feature.summary}\n</candidate>"
        for candidate, feature in candidates
    )
    user = (
        f"New request:\n{_request_block(title, description)}\n\n"
        f"Candidate features:\n{candidate_blocks}"
    )
    result = llm.structured(system=prompt.text, user=user, output_type=DuplicateJudgment)

    judgment = result.output
    valid_ids = {feature.id for _, feature in candidates}
    if judgment.duplicate_of_feature_id not in valid_ids:
        judgment.duplicate_of_feature_id = None
    judgment.confidence = _clamp(judgment.confidence)
    return result, user


# --- Stage 3: summary update for a merge (LLM) -------------------------------


def propose_summary(
    llm: LLM, feature: Feature, title: str, description: str
) -> tuple[LLMResult[SummaryUpdate], str]:
    prompt = prompt_for("summary")
    user = (
        f"<feature>\nTitle: {feature.title}\nCurrent summary: {feature.summary}\n</feature>\n\n"
        f"New duplicate request:\n{_request_block(title, description)}"
    )
    return llm.structured(system=prompt.text, user=user, output_type=SummaryUpdate), user


# --- Stage 4: categorization (LLM) ------------------------------------------


def active_categories(session: Session) -> list[Category]:
    return list(
        session.scalars(
            select(Category).where(Category.status == CategoryStatus.ACTIVE).order_by(Category.name)
        )
    )


def categorize(
    llm: LLM, session: Session, title: str, description: str
) -> tuple[LLMResult[CategoryAssignment], str]:
    prompt = prompt_for("categorize")
    categories = active_categories(session)

    blocks = []
    for category in categories:
        examples = "\n".join(
            f"  - {feature.title}" for feature in category.features[:EXAMPLES_PER_CATEGORY]
        )
        blocks.append(
            f'<category id="{category.id}" name="{category.name}">\n'
            f"Description: {category.description}\n"
            f"Example requests:\n{examples or '  (none yet)'}\n</category>"
        )
    category_text = "\n".join(blocks) if blocks else "(no categories exist yet)"

    user = f"Categories:\n{category_text}\n\nRequest to categorize:\n{_request_block(title, description)}"
    result = llm.structured(system=prompt.text, user=user, output_type=CategoryAssignment)

    assignment = result.output
    assignment.confidence = _clamp(assignment.confidence)
    valid_ids = {c.id for c in categories}
    if assignment.action == "assign" and assignment.category_id not in valid_ids:
        # The model named a category that doesn't exist. If it also described
        # a new one, treat it as that proposal; otherwise leave no category,
        # with zero confidence, so the reviewer has to pick one.
        assignment.category_id = None
        if assignment.new_category_name:
            assignment.action = "propose_new"
        else:
            assignment.confidence = 0.0
    return result, user


# --- Taxonomy bootstrap (LLM) ------------------------------------------------


def propose_taxonomy(
    llm: LLM, requests: Sequence[tuple[str, str]]
) -> tuple[LLMResult[TaxonomyProposal], str]:
    prompt = prompt_for("taxonomy")
    user = "Feature requests:\n" + "\n".join(
        _request_block(title, description) for title, description in requests
    )
    return llm.structured(system=prompt.text, user=user, output_type=TaxonomyProposal), user
