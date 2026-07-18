"""Minimal trace context used until the Phase F observability work."""

from __future__ import annotations

from typing import Any
from uuid import uuid4


class TraceContext:
    def __init__(self) -> None:
        self.trace_id = str(uuid4())
        self.stages: list[dict[str, Any]] = []

    def record_stage(
        self, name: str, details: dict[str, Any] | None = None
    ) -> None:
        self.stages.append({"name": name, "details": details or {}})
