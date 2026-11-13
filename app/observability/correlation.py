"""Correlation IDs and eval run IDs for the current unit of work.

A correlation ID is created when a submission is received and stored on the
submission. Every later decision about that submission, even one made days
later, binds the same ID again so all of its log events can be found together.

An eval run ID is only set while the eval runner is working.

Both are stored in context variables, so each request (or thread) sees its own.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_current_eval_run_id: ContextVar[str | None] = ContextVar("eval_run_id", default=None)


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def is_valid_correlation_id(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, TypeError):
        return False
    return True


def get_current_correlation_id() -> str | None:
    return _current_correlation_id.get()


def get_current_eval_run_id() -> str | None:
    return _current_eval_run_id.get()


@contextmanager
def bind_correlation_id(correlation_id: str) -> Iterator[str]:
    """Make `correlation_id` the current one until the `with` block ends."""
    token = _current_correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _current_correlation_id.reset(token)


@contextmanager
def bind_eval_run_id(run_id: str) -> Iterator[str]:
    """Make `run_id` the current eval run ID until the `with` block ends."""
    token = _current_eval_run_id.set(run_id)
    try:
        yield run_id
    finally:
        _current_eval_run_id.reset(token)
