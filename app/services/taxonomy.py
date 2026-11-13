"""Taxonomy bootstrap: the AI proposes a category set, a human approves it."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Category, Proposal
from app.domain.enums import CategoryStatus, Decision, ProposalType
from app.domain.proposals import validate_decision
from app.observability.correlation import correlation_scope, new_correlation_id
from app.observability.decision_log import Actor, EventType
from app.pipeline import stages
from app.services.context import TriageContext, mark_decided
from app.services.review import NotFoundError


def propose(session: Session, ctx: TriageContext, requests: Sequence[tuple[str, str]]) -> Proposal:
    """Ask the model for a taxonomy. Raises LLMError if the call fails."""
    with correlation_scope(new_correlation_id()) as correlation_id:
        prompt = stages.prompt_for("taxonomy")
        result, user = stages.propose_taxonomy(ctx.llm, requests)
        categories = [c.model_dump() for c in result.output.categories]
        event = ctx.log_ai(
            EventType.TAXONOMY_PROPOSED,
            result,
            prompt,
            user,
            parent_event_id=None,
            payload={"request_count": len(requests), "categories": categories},
            reasoning=result.output.reasoning,
        )
        proposal = Proposal(
            correlation_id=correlation_id,
            type=ProposalType.TAXONOMY,
            payload={"categories": categories},
            reasoning=result.output.reasoning,
            model=result.model,
            prompt_version=prompt.id,
            ai_event_id=event.event_id,
        )
        session.add(proposal)
        session.commit()
        return proposal


def decide(
    session: Session,
    ctx: TriageContext,
    proposal_id: int,
    decision: Decision,
    categories: list[dict] | None = None,
) -> list[Category]:
    """Approve the proposed taxonomy as-is, approve an edited version, or reject it.

    Categories whose name already exists are skipped, not duplicated.
    """
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.type is not ProposalType.TAXONOMY:
        raise NotFoundError(f"No taxonomy proposal with id {proposal_id}")

    cleaned = [
        {"name": c["name"].strip(), "description": c["description"].strip()}
        for c in categories or []
        if c.get("name", "").strip() and c.get("description", "").strip()
    ]
    if decision is Decision.EDIT and cleaned == proposal.payload["categories"]:
        decision = Decision.APPROVE  # "edited" with no changes is a plain approval
    decision_payload = {"categories": cleaned} if decision is Decision.EDIT else None
    status = validate_decision(proposal.type, proposal.status, decision, decision_payload)

    with correlation_scope(proposal.correlation_id):
        mark_decided(proposal, status, ctx.reviewer, decision_payload)
        event = ctx.log.record(
            EventType.TAXONOMY_DECISION,
            Actor.human(ctx.reviewer),
            parent_event_id=proposal.ai_event_id,
            payload={"decision": decision, **(decision_payload or {})},
        )
        proposal.decision_event_id = event.event_id

        created: list[Category] = []
        if decision is not Decision.REJECT:
            chosen = cleaned if decision is Decision.EDIT else proposal.payload["categories"]
            existing = set(session.scalars(select(Category.name)))
            for item in chosen:
                if item["name"] in existing:
                    continue
                category = Category(
                    name=item["name"], description=item["description"], status=CategoryStatus.ACTIVE
                )
                session.add(category)
                session.flush()
                existing.add(item["name"])
                created.append(category)
                ctx.log.record(
                    EventType.CATEGORY_CREATED,
                    Actor.human(ctx.reviewer),
                    parent_event_id=event.event_id,
                    payload={
                        "category_id": category.id,
                        "name": category.name,
                        "description": category.description,
                        "source_proposal_id": proposal.id,
                    },
                )
        session.commit()
        return created
