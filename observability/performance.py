"""Reusable observability timing helpers."""

from __future__ import annotations

from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from observability.tracer import trace_stage

P = ParamSpec("P")
R = TypeVar("R")


def traced_stage(stage_name: str):
    """Decorate a synchronous operation so it records one shared stage timer."""
    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with trace_stage(stage_name):
                return function(*args, **kwargs)
        return wrapped
    return decorator
