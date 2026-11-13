"""Correlation and run IDs carried through a unit of work via contextvars.

A correlation ID is created when a submission is received and stored on the
submission, so a human review that happens days later can re-enter the same
correlation scope. A run ID is set only during eval runs.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)


def new_correlation_id() -> str:
    return str(uuid.uuid4())


def is_valid_correlation_id(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, TypeError):
        return False
    return True


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def get_run_id() -> str | None:
    return _run_id.get()


@contextmanager
def correlation_scope(correlation_id: str) -> Iterator[str]:
    """Bind `correlation_id` for the duration of the block."""
    token = _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.reset(token)


@contextmanager
def run_scope(run_id: str) -> Iterator[str]:
    """Bind an eval `run_id` for the duration of the block."""
    token = _run_id.set(run_id)
    try:
        yield run_id
    finally:
        _run_id.reset(token)
