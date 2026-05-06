"""Minimal placeholder interaction to satisfy VeRL's multi-turn config validator."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl.interactions.base import BaseInteraction


class NoOpInteraction(BaseInteraction):
    async def start_interaction(self, instance_id: str | None = None, **kwargs) -> str:
        return instance_id or str(uuid4())

    async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs):
        return True, "", 0.0, {}

    async def calculate_score(self) -> float:
        return 0.0

    async def finalize_interaction(self) -> None:
        return None
