"""The AI pipeline's stages, as plain functions.

Nothing here writes to the decision log or knows about FastAPI; the services
layer does both. The eval runner calls these functions directly.

Every answer from the model is checked against what it was given, so an ID the
model made up never reaches a proposal.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.llm import LanguageModel, LLMResult
from app.ai.prompts import PromptTemplate, load_prompt_template
from app.ai.vectors import bytes_to_vector, cosine_similarity
from app.db.models import Category, Feature, Submission
from app.domain.enums import CategoryStatus
from app.pipeline.schemas import (
    CategoryAssignment,
    DuplicateJudgment,
    SummaryUpdate,
    TaxonomyProposal,
)

CURRENT_PROMPT_VERSION = "v1"
EXAMPLE_REQUESTS_SHOWN_PER_CATEGORY = 3


def load_current_prompt(name: str) -> PromptTemplate:
    return load_prompt_template(name, CURRENT_PROMPT_VERSION)


def clamp_to_zero_one(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def format_request_for_prompt(title: str, description: str) -> str:
    return f"<request>\nTitle: {title}\nDescription: {description}\n</request>"


# --- Stage 1: find duplicate candidates with embedding search ---------------


@dataclass(frozen=True)
class DuplicateCandidate:
    feature_id: int
    similarity: float
    most_similar_submission_id: int


def find_duplicate_candidates(
    session: Session,
    new_submission_embedding: np.ndarray,
    *,
    max_candidates: int,
    min_similarity: float,
    submission_id_to_skip: int | None = None,
) -> list[DuplicateCandidate]:
    """Find existing features that look similar to a new submission.

    Every triaged submission is compared, not just each feature's summary, so a
    rewording of any one past request still matches. Each feature is listed
    once, using its most similar submission. Results are sorted best first.
    Archived submissions and archived features are left out.
    """
    query = (
        select(Submission.id, Submission.feature_id, Submission.embedding)
        .join(Feature, Feature.id == Submission.feature_id)
        .where(
            Submission.embedding.is_not(None),
            Submission.archived_at.is_(None),
            Feature.archived_at.is_(None),
        )
    )
    triaged_submissions = session.execute(query).all()

    best_candidate_per_feature: dict[int, DuplicateCandidate] = {}
    for submission_id, feature_id, stored_embedding in triaged_submissions:
        if submission_id == submission_id_to_skip:
            continue

        similarity = cosine_similarity(new_submission_embedding, bytes_to_vector(stored_embedding))
        if similarity < min_similarity:
            continue

        current_best = best_candidate_per_feature.get(feature_id)
        if current_best is None or similarity > current_best.similarity:
            best_candidate_per_feature[feature_id] = DuplicateCandidate(
                feature_id=feature_id,
                similarity=similarity,
                most_similar_submission_id=submission_id,
            )

    candidates = list(best_candidate_per_feature.values())
    candidates.sort(key=lambda candidate: candidate.similarity, reverse=True)
    return candidates[:max_candidates]


# --- Stage 2: ask the model whether it is a duplicate -----------------------


def build_duplicate_check_message(
    title: str, description: str, candidates: Sequence[tuple[DuplicateCandidate, Feature]]
) -> str:
    candidate_blocks = []
    for candidate, feature in candidates:
        candidate_blocks.append(
            f'<candidate feature_id="{feature.id}" similarity="{candidate.similarity:.2f}">\n'
            f"Title: {feature.title}\n"
            f"Summary: {feature.summary}\n"
            f"</candidate>"
        )
    return (
        "New request:\n"
        + format_request_for_prompt(title, description)
        + "\n\nCandidate features:\n"
        + "\n".join(candidate_blocks)
    )


def ask_model_if_duplicate(
    model: LanguageModel,
    title: str,
    description: str,
    candidates: Sequence[tuple[DuplicateCandidate, Feature]],
) -> tuple[LLMResult[DuplicateJudgment], str]:
    """Return the model's judgment and the exact message that was sent."""
    prompt = load_current_prompt("duplicate_judge")
    user_message = build_duplicate_check_message(title, description, candidates)
    result = model.request_structured_output(
        system_prompt=prompt.text, user_message=user_message, output_type=DuplicateJudgment
    )

    judgment = result.output
    candidate_feature_ids = {feature.id for _, feature in candidates}
    if judgment.duplicate_of_feature_id not in candidate_feature_ids:
        # Either "not a duplicate" (None), or an ID the model wasn't shown.
        judgment.duplicate_of_feature_id = None
    judgment.confidence = clamp_to_zero_one(judgment.confidence)
    return result, user_message


# --- Stage 3: ask the model for an updated summary after a merge ------------


def build_summary_update_message(feature: Feature, title: str, description: str) -> str:
    return (
        f"<feature>\nTitle: {feature.title}\nCurrent summary: {feature.summary}\n</feature>\n\n"
        "New duplicate request:\n"
        + format_request_for_prompt(title, description)
    )


def ask_model_for_merged_summary(
    model: LanguageModel, feature: Feature, title: str, description: str
) -> tuple[LLMResult[SummaryUpdate], str]:
    """Return the model's updated summary and the exact message that was sent."""
    prompt = load_current_prompt("summary")
    user_message = build_summary_update_message(feature, title, description)
    result = model.request_structured_output(
        system_prompt=prompt.text, user_message=user_message, output_type=SummaryUpdate
    )
    return result, user_message


# --- Stage 4: ask the model for a category ----------------------------------


def list_active_categories(session: Session) -> list[Category]:
    query = (
        select(Category)
        .where(Category.status == CategoryStatus.ACTIVE)
        .order_by(Category.name)
    )
    return list(session.scalars(query))


def build_category_list_for_prompt(categories: Sequence[Category]) -> str:
    if not categories:
        return "(no categories exist yet)"

    category_blocks = []
    for category in categories:
        active_features = [feature for feature in category.features if not feature.is_archived]
        example_features = active_features[:EXAMPLE_REQUESTS_SHOWN_PER_CATEGORY]
        if example_features:
            example_lines = "\n".join(f"  - {feature.title}" for feature in example_features)
        else:
            example_lines = "  (none yet)"
        category_blocks.append(
            f'<category id="{category.id}" name="{category.name}">\n'
            f"Description: {category.description}\n"
            f"Example requests:\n{example_lines}\n"
            f"</category>"
        )
    return "\n".join(category_blocks)


def build_categorize_message(categories: Sequence[Category], title: str, description: str) -> str:
    return (
        "Categories:\n"
        + build_category_list_for_prompt(categories)
        + "\n\nRequest to categorize:\n"
        + format_request_for_prompt(title, description)
    )


def ask_model_for_category(
    model: LanguageModel, session: Session, title: str, description: str
) -> tuple[LLMResult[CategoryAssignment], str]:
    """Return the model's category choice and the exact message that was sent."""
    prompt = load_current_prompt("categorize")
    categories = list_active_categories(session)
    user_message = build_categorize_message(categories, title, description)
    result = model.request_structured_output(
        system_prompt=prompt.text, user_message=user_message, output_type=CategoryAssignment
    )

    assignment = result.output
    assignment.confidence = clamp_to_zero_one(assignment.confidence)

    active_category_ids = {category.id for category in categories}
    model_chose_unknown_category = (
        assignment.action == "assign" and assignment.category_id not in active_category_ids
    )
    if model_chose_unknown_category:
        assignment.category_id = None
        if assignment.new_category_name:
            # It also described a new category, so treat it as that proposal.
            assignment.action = "propose_new"
        else:
            # Nothing usable: zero confidence, so the reviewer has to pick one.
            assignment.confidence = 0.0
    return result, user_message


# --- Taxonomy bootstrap: ask the model to propose a set of categories -------


def build_taxonomy_message(requests: Sequence[tuple[str, str]]) -> str:
    request_blocks = []
    for title, description in requests:
        request_blocks.append(format_request_for_prompt(title, description))
    return "Feature requests:\n" + "\n".join(request_blocks)


def ask_model_for_taxonomy(
    model: LanguageModel, requests: Sequence[tuple[str, str]]
) -> tuple[LLMResult[TaxonomyProposal], str]:
    """Return the model's proposed categories and the exact message that was sent."""
    prompt = load_current_prompt("taxonomy")
    user_message = build_taxonomy_message(requests)
    result = model.request_structured_output(
        system_prompt=prompt.text, user_message=user_message, output_type=TaxonomyProposal
    )
    return result, user_message
