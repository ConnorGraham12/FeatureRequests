"""Human-gate rules for AI proposals. Pure logic, no I/O.

AI output never changes state directly: it becomes a proposal, and only a human
decision validated here may move it out of PENDING.
"""

from collections.abc import Mapping
from typing import Any

from app.domain.enums import Decision, ProposalStatus, ProposalType


class ProposalAlreadyDecidedError(ValueError):
    pass


class InvalidDecisionError(ValueError):
    pass


_DECISION_TO_STATUS = {
    Decision.APPROVE: ProposalStatus.APPROVED,
    Decision.REJECT: ProposalStatus.REJECTED,
    Decision.EDIT: ProposalStatus.EDITED,
}

# Fields the reviewer must supply for a given (proposal type, decision).
# A rejected category must still be saved, with the reviewer's chosen category.
_REQUIRED_FIELDS: dict[tuple[ProposalType, Decision], tuple[str, ...]] = {
    (ProposalType.CATEGORY, Decision.REJECT): ("category_id",),
    (ProposalType.CATEGORY, Decision.EDIT): ("category_id",),
    (ProposalType.SUMMARY, Decision.EDIT): ("summary",),
    (ProposalType.NEW_CATEGORY, Decision.EDIT): ("name", "description"),
    (ProposalType.NEW_CATEGORY, Decision.REJECT): ("category_id",),
    (ProposalType.TAXONOMY, Decision.EDIT): ("categories",),
}


def validate_decision(
    proposal_type: ProposalType,
    current_status: ProposalStatus,
    decision: Decision,
    decision_payload: Mapping[str, Any] | None = None,
) -> ProposalStatus:
    """Check a human decision is allowed and return the proposal's new status."""
    if current_status is not ProposalStatus.PENDING:
        raise ProposalAlreadyDecidedError(f"Proposal is already {current_status}")

    payload = decision_payload or {}
    missing = [
        field
        for field in _REQUIRED_FIELDS.get((proposal_type, decision), ())
        if payload.get(field) in (None, "", [])
    ]
    if missing:
        raise InvalidDecisionError(
            f"{decision} on a {proposal_type} proposal requires: {', '.join(missing)}"
        )
    return _DECISION_TO_STATUS[decision]
