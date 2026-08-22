"""Shared AsyncOpenAI chat.completions.create with SDK kwarg guards."""

from __future__ import annotations

import logging
from typing import Any

from xiaozhi_bridge.llm_params import (
    parse_unexpected_sdk_kwarg,
    relocate_sdk_kwarg_to_extra_body,
    strip_sdk_kwarg,
)

logger = logging.getLogger(__name__)


async def sdk_create_completion(client: Any, kwargs: dict[str, Any]) -> Any:
    """Call create(); on SDK TypeError relocate or strip the bad kwarg once."""
    try:
        return await client.chat.completions.create(**kwargs)
    except TypeError as e:
        param = parse_unexpected_sdk_kwarg(e)
        if not param:
            raise
        relocated = relocate_sdk_kwarg_to_extra_body(kwargs, param)
        if relocated:
            logger.warning("LLM SDK rejected %s — retrying via extra_body", param)
            return await client.chat.completions.create(**relocated)
        stripped = strip_sdk_kwarg(kwargs, param)
        logger.warning("LLM SDK rejected %s — retrying without it", param)
        return await client.chat.completions.create(**stripped)
