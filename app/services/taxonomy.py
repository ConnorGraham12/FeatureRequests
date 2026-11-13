"""Taxonomy bootstrap: the AI proposes a set of categories and a human approves them.

    request_taxonomy_proposal   ask the AI for categories, stored as a pending proposal
    approve_taxonomy            create the categories the reviewer kept (edited or not)
    reject_taxonomy             create nothing
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Category, Proposal
from app.domain.enums import CategoryStatus, Decision, ProposalStatus, ProposalType
from app.domain.proposals import InvalidDecisionError, ensure_proposal_is_pending
from app.observability.correlation import bind_correlation_id, generate_correlation_id
from app.observability.decision_log import Actor, EventType
from app.pipeline import stages
from app.services.context import TriageContext, record_decision_on_proposal
from app.services.review import NotFoundError


def request_taxonomy_proposal(
    session: Session, context: TriageContext, requests: Sequence[tuple[str, str]]
) -> Proposal:
    """Ask the model to propose categories for these (title, description) requests.

    Raises LLMError if the model call fails.
    """
    with bind_correlation_id(generate_correlation_id()) as correlation_id:
        prompt = stages.load_current_prompt("taxonomy")
        result, user_message = stages.ask_model_for_taxonomy(context.language_model, requests)

        proposed_categories = []
        for category in result.output.categories:
            proposed_categories.append({"name": category.name, "description": category.description})

        proposed_event = context.write_ai_decision_event(
            EventType.TAXONOMY_PROPOSED,
            result,
            prompt,
            user_message,
            parent_event_id=None,
            payload={"request_count": len(requests), "categories": proposed_categories},
            reasoning=result.output.reasoning,
        )
        proposal = Proposal(
            correlation_id=correlation_id,
            type=ProposalType.TAXONOMY,
            payload={"categories": proposed_categories},
            reasoning=result.output.reasoning,
            model=result.model,
            prompt_version=prompt.versioned_name,
            ai_event_id=proposed_event.event_id,
        )
        session.add(proposal)
        session.commit()
        return proposal


def load_pending_taxonomy_proposal(session: Session, proposal_id: int) -> Proposal:
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.type is not ProposalType.TAXONOMY:
        raise NotFoundError(f"No taxonomy proposal with id {proposal_id}")
    ensure_proposal_is_pending(proposal.status)
    return proposal


def remove_blank_categories(categories: list[dict]) -> list[dict]:
    """Trim names and descriptions, dropping any row where either is empty."""
    cleaned = []
    for category in categories:
        name = category.get("name", "").strip()
        description = category.get("description", "").strip()
        if name and description:
            cleaned.append({"name": name, "description": description})
    return cleaned


def approve_taxonomy(
    session: Session, context: TriageContext, proposal_id: int, kept_categories: list[dict]
) -> list[Category]:
    """Create the categories the reviewer kept from the AI's proposal.

    If the kept list is exactly what the AI proposed, this is a plain approval;
    otherwise it is recorded as an edit. A category whose name already exists is
    skipped rather than duplicated. Returns the categories that were created.
    """
    proposal = load_pending_taxonomy_proposal(session, proposal_id)
    kept_categories = remove_blank_categories(kept_categories)
    if not kept_categories:
        raise InvalidDecisionError("Keep at least one category, or reject the proposal")

    reviewer_changed_something = kept_categories != proposal.payload["categories"]
    if reviewer_changed_something:
        decision = Decision.EDIT
        new_status = ProposalStatus.EDITED
        decision_details = {"categories": kept_categories}
    else:
        decision = Decision.APPROVE
        new_status = ProposalStatus.APPROVED
        decision_details = None

    with bind_correlation_id(proposal.correlation_id):
        record_decision_on_proposal(proposal, new_status, context.reviewer_name, decision_details)
        log_payload = {"decision": decision}
        if decision_details:
            log_payload["categories"] = kept_categories
        decision_event = context.decision_log.write_event(
            EventType.TAXONOMY_DECISION,
            Actor.for_human_reviewer(context.reviewer_name),
            parent_event_id=proposal.ai_event_id,
            payload=log_payload,
        )
        proposal.decision_event_id = decision_event.event_id

        created_categories = create_categories_not_already_present(
            session, context, kept_categories, proposal.id, decision_event.event_id
        )
        session.commit()
        return created_categories


def create_categories_not_already_present(
    session: Session,
    context: TriageContext,
    categories: list[dict],
    source_proposal_id: int,
    parent_event_id: str,
) -> list[Category]:
    existing_names = set(session.scalars(select(Category.name)))
    created_categories = []
    for item in categories:
        if item["name"] in existing_names:
            continue

        category = Category(name=item["name"], description=item["description"], status=CategoryStatus.ACTIVE)
        session.add(category)
        session.flush()  # assigns category.id
        existing_names.add(item["name"])
        created_categories.append(category)

        context.decision_log.write_event(
            EventType.CATEGORY_CREATED,
            Actor.for_human_reviewer(context.reviewer_name),
            parent_event_id=parent_event_id,
            payload={
                "category_id": category.id,
                "name": category.name,
                "description": category.description,
                "source_proposal_id": source_proposal_id,
            },
        )
    return created_categories


def reject_taxonomy(session: Session, context: TriageContext, proposal_id: int) -> None:
    proposal = load_pending_taxonomy_proposal(session, proposal_id)
    with bind_correlation_id(proposal.correlation_id):
        record_decision_on_proposal(proposal, ProposalStatus.REJECTED, context.reviewer_name, None)
        decision_event = context.decision_log.write_event(
            EventType.TAXONOMY_DECISION,
            Actor.for_human_reviewer(context.reviewer_name),
            parent_event_id=proposal.ai_event_id,
            payload={"decision": Decision.REJECT},
        )
        proposal.decision_event_id = decision_event.event_id
        session.commit()
