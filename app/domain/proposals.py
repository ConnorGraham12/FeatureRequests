"""Rules for human decisions on AI proposals. Pure logic, no database or I/O.

The AI never changes state directly: what it suggests is stored as a proposal,
and only a human decision may move that proposal out of PENDING.
"""

from app.domain.enums import ProposalStatus


class ProposalAlreadyDecidedError(ValueError):
    """Raised when someone tries to decide a proposal a second time."""


class InvalidDecisionError(ValueError):
    """Raised when a decision is missing information or isn't allowed."""


def ensure_proposal_is_pending(current_status: ProposalStatus) -> None:
    """Raise unless the proposal is still waiting for a decision."""
    if current_status is not ProposalStatus.PENDING:
        raise ProposalAlreadyDecidedError(f"This proposal was already {current_status}")


def ensure_text_is_present(value: str | None, field_description: str) -> str:
    """Return the trimmed text, or raise if it is empty."""
    trimmed = (value or "").strip()
    if not trimmed:
        raise InvalidDecisionError(f"{field_description} is required")
    return trimmed


def ensure_category_is_chosen(category_id: int | None) -> int:
    """Return the category ID, or raise if the reviewer didn't choose one."""
    if category_id is None:
        raise InvalidDecisionError("Choose a category")
    return category_id
