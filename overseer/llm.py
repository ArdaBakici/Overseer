"""Thin wrapper around the Anthropic client with Overseer's defaults."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from .config import LLMConfig

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = anthropic.AsyncAnthropic() if cfg.enabled else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    async def create(self, *, system: str, messages: list[dict[str, Any]], effort: str | None,
                     tools: list[dict[str, Any]] | None = None, max_tokens: int | None = None):
        assert self.client is not None
        params: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.cfg.adaptive_thinking:
            params["thinking"] = {"type": "adaptive"}
        if effort:
            params["output_config"] = {"effort": effort}
        if tools:
            params["tools"] = tools
            # Agent loops resend a growing history; cache the prefix.
            params["cache_control"] = {"type": "ephemeral"}
        if self.cfg.fallbacks:
            # On a safety-classifier refusal, the API re-runs the request on a fallback model.
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
            return await self.client.beta.messages.create(**params)
        return await self.client.messages.create(**params)

    async def text(self, *, system: str, prompt: str, effort: str | None, max_tokens: int = 4000) -> str:
        resp = await self.create(system=system, messages=[{"role": "user", "content": prompt}],
                                 effort=effort, max_tokens=max_tokens)
        if resp.stop_reason == "refusal":
            return "(the model declined to analyse this)"
        return response_text(resp)


def response_text(resp: Any) -> str:
    return "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
