import pytest

from app.domain.enums import ProposalStatus
from app.domain.proposals import (
    InvalidDecisionError,
    ProposalAlreadyDecidedError,
    ensure_category_is_chosen,
    ensure_proposal_is_pending,
    ensure_text_is_present,
)


def test_pending_proposal_can_be_decided():
    ensure_proposal_is_pending(ProposalStatus.PENDING)


@pytest.mark.parametrize(
    "status", [ProposalStatus.APPROVED, ProposalStatus.REJECTED, ProposalStatus.EDITED]
)
def test_decided_proposal_cannot_be_decided_again(status):
    with pytest.raises(ProposalAlreadyDecidedError):
        ensure_proposal_is_pending(status)


def test_text_must_not_be_blank():
    assert ensure_text_is_present("  My summary ", "A summary") == "My summary"
    with pytest.raises(InvalidDecisionError, match="A summary is required"):
        ensure_text_is_present("   ", "A summary")


def test_a_category_must_be_chosen():
    assert ensure_category_is_chosen(3) == 3
    with pytest.raises(InvalidDecisionError, match="Choose a category"):
        ensure_category_is_chosen(None)
