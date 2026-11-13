import pytest

from app.domain.enums import Decision, ProposalStatus, ProposalType
from app.domain.proposals import (
    InvalidDecisionError,
    ProposalAlreadyDecidedError,
    validate_decision,
)


def test_approve_pending_proposal():
    status = validate_decision(ProposalType.DUPLICATE, ProposalStatus.PENDING, Decision.APPROVE)
    assert status is ProposalStatus.APPROVED


def test_decided_proposal_cannot_be_decided_again():
    with pytest.raises(ProposalAlreadyDecidedError):
        validate_decision(ProposalType.DUPLICATE, ProposalStatus.APPROVED, Decision.REJECT)


def test_rejected_category_requires_reviewer_category():
    with pytest.raises(InvalidDecisionError, match="category_id"):
        validate_decision(ProposalType.CATEGORY, ProposalStatus.PENDING, Decision.REJECT)

    status = validate_decision(
        ProposalType.CATEGORY, ProposalStatus.PENDING, Decision.REJECT, {"category_id": 3}
    )
    assert status is ProposalStatus.REJECTED


def test_rejected_merge_needs_no_payload():
    status = validate_decision(ProposalType.DUPLICATE, ProposalStatus.PENDING, Decision.REJECT)
    assert status is ProposalStatus.REJECTED


def test_edited_summary_requires_text():
    with pytest.raises(InvalidDecisionError):
        validate_decision(ProposalType.SUMMARY, ProposalStatus.PENDING, Decision.EDIT, {"summary": ""})
