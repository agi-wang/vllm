# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /v1/systemone`` for a decision pooling model.

Clef encodes the record off the event loop, sends span metadata through
``PoolingParams.extra_kwargs``, and formats the worker's joint-head logits.
The other families send one prefill per prompt (Lev choice sends two) and
read the published head for that family. vLLM batches the prefills. There
is no token generation, so prefix cache is skipped. Request bodies are not
logged. Images and videos are rejected.
"""

from __future__ import annotations

import asyncio
import uuid
from functools import cache
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.entrypoints.serve.utils.api_utils import (
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.model_executor.models.clef_schema import (
    answers_from_logits,
    encode_record,
    validate_systemone_request,
)
from vllm.model_executor.models.decision_heads import (
    SYSTEMONE_ARCHITECTURES,
    DecisionPrompt,
    answers_from_prompt_scores,
    build_prompts,
)
from vllm.pooling_params import PoolingParams

logger = init_logger(__name__)

router = APIRouter()


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    state: Any
    questions: dict[str, Any]
    images: list[Any] | None = None
    videos: list[Any] | None = None


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def _error(exc: Exception) -> JSONResponse:
    response = create_error_response(exc)
    return JSONResponse(content=response.model_dump(), status_code=response.error.code)


def _family(model_config: Any) -> str | None:
    for architecture in model_config.architectures or []:
        family = SYSTEMONE_ARCHITECTURES.get(architecture)
        if family is not None:
            return family
    return None


def _temperature_kwargs(model_config: Any, family: str) -> dict[str, Any]:
    """Published temperatures. Missing files keep the default of 1."""
    model = model_config.model
    revision = model_config.revision
    if family == "laya":
        return _laya_temperature_kwargs(model, revision)
    if family == "lev":
        return {"temperatures": _lev_temperatures(model, revision)}
    if family == "kev":
        return {"temperatures": _kev_temperatures(model, revision)}
    return {}


@cache
def _laya_temperature_kwargs(model: str, revision: str | None) -> dict[str, Any]:
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    blob = get_hf_file_to_dict("rl_agent_config.json", model, revision)
    if not blob:
        blob = get_hf_file_to_dict("julia_config.json", model, revision)
    blob = blob or {}
    by_options = blob.get("temperature_by_options")
    by_type = blob.get("temperature")
    return {
        "laya_by_options": by_options if isinstance(by_options, dict) else None,
        "laya_by_type": by_type if isinstance(by_type, list) else None,
    }


@cache
def _lev_temperatures(model: str, revision: str | None) -> dict[str, float] | None:
    """Mode A buckets from ``calibration.json``, named ``choice.small``."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    blob = get_hf_file_to_dict("calibration.json", model, revision) or {}
    raw = blob.get("temperatures") if isinstance(blob, dict) else None
    if not isinstance(raw, dict):
        return None
    table: dict[str, float] = {}
    for name, value in raw.items():
        parts = str(name).split(":")
        if len(parts) >= 2 and parts[1] == "A":
            key = ".".join([parts[0], *parts[2:]])
        else:
            continue
        try:
            table[key] = float(value)
        except (TypeError, ValueError):
            continue
    return table or None


@cache
def _kev_temperatures(model: str, revision: str | None) -> dict[str, float] | None:
    """One scalar from ``head.pt`` applies to choice, score, and noul."""
    from pathlib import Path

    import torch

    from vllm.transformers_utils.repo_utils import (
        _try_download_from_hf_hub,
        try_get_local_file,
    )

    local = try_get_local_file(model, "head.pt", revision=revision)
    path = local if isinstance(local, Path) and local.is_file() else None
    if path is None:
        path = _try_download_from_hf_hub(model, "head.pt", revision)
    if path is None:
        return None
    blob = torch.load(path, map_location="cpu", weights_only=True)
    value = blob.get("temperature") if isinstance(blob, dict) else None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = float(value.reshape(-1)[0])
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return {"choice": number, "score": number, "noul": number}


def _score_rows(data: Any) -> list[float]:
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    return [float(value) for value in data]


async def _encode_one(
    engine: Any,
    family: str,
    prompt: DecisionPrompt,
) -> list[float]:
    pooling_params = PoolingParams(
        task="token_classify",
        use_activation=False,
        skip_reading_prefix_cache=True,
        extra_kwargs={"decision": prompt.extra},
    )
    request_id = f"{family}-{uuid.uuid4().hex}"
    final = None
    async for item in engine.encode(
        {"prompt_token_ids": list(prompt.token_ids)},
        pooling_params,
        request_id,
    ):
        if item.finished:
            final = item
    if final is None or final.outputs is None or final.outputs.data is None:
        raise RuntimeError(f"{family} pooling produced no scores")
    return _score_rows(final.outputs.data)


async def _encode_decision_family(
    engine: Any,
    payload: dict[str, Any],
    family: str,
) -> tuple[dict[str, Any], int]:
    if family == "diffusiongemma":
        raise NotImplementedError(
            "DiffusionGemma decisions use the structured-read example server"
        )
    style = getattr(engine.model_config.hf_config, "laya_prompt_style", "laya")
    temperature_kwargs = _temperature_kwargs(engine.model_config, family)
    executor = getattr(engine.renderer, "_executor", None)

    def _encode() -> list[DecisionPrompt]:
        tokenizer = engine.renderer.get_tokenizer()
        return build_prompts(
            family,
            tokenizer,
            payload,
            max_length=engine.model_config.max_model_len,
            laya_style=style,
            **temperature_kwargs,
        )

    prompts = await asyncio.get_running_loop().run_in_executor(executor, _encode)
    rows = await asyncio.gather(
        *(_encode_one(engine, family, prompt) for prompt in prompts)
    )
    answers = answers_from_prompt_scores(prompts, rows, payload["questions"])
    return answers, sum(len(prompt.token_ids) for prompt in prompts)


@router.post(
    "/v1/systemone",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": dict},
        HTTPStatus.NOT_FOUND.value: {"model": dict},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": dict},
    },
)
@with_cancellation
async def create_systemone(request: SystemOneRequest, raw_request: Request):
    engine = raw_request.app.state.engine_client
    payload = request.model_dump()
    try:
        validate_systemone_request(payload)
        served = engine.model_config.served_model_name
        if payload["model"] != served:
            raise ValueError(
                f"model {payload['model']!r} is not served here; use {served!r}"
            )
        family = _family(engine.model_config)
        if family is None:
            raise RuntimeError("this model does not serve /v1/systemone")
        if family != "clef":
            if engine.model_config.runner_type != "pooling":
                raise RuntimeError("/v1/systemone requires the pooling runner")
            answers, input_tokens = await _encode_decision_family(
                engine, payload, family
            )
            return JSONResponse(
                content={
                    "model": payload["model"],
                    "answers": answers,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                    },
                }
            )
        if engine.model_config.runner_type != "pooling":
            raise RuntimeError("Clef /v1/systemone requires the pooling runner")
        executor = getattr(engine.renderer, "_executor", None)

        def _encode():
            tokenizer = engine.renderer.get_tokenizer()
            return encode_record(
                tokenizer,
                payload,
                max_length=engine.model_config.max_model_len,
            )

        encoded = await asyncio.get_running_loop().run_in_executor(executor, _encode)
        pooling_params = PoolingParams(
            task="token_classify",
            use_activation=False,
            skip_reading_prefix_cache=True,
            extra_kwargs={
                "clef_questions": [
                    question.as_extra() for question in encoded.questions
                ]
            },
        )
        request_id = f"clef-{uuid.uuid4().hex}"
        final = None
        async for item in engine.encode(
            {"prompt_token_ids": list(encoded.input_ids)},
            pooling_params,
            request_id,
        ):
            if item.finished:
                final = item
        if final is None or final.outputs is None or final.outputs.data is None:
            raise RuntimeError("Clef pooling produced no logits")
        answers = answers_from_logits(
            payload["questions"],
            encoded,
            final.outputs.data,
        )
        input_tokens = len(encoded.input_ids)
    except Exception as exc:
        logger.debug(
            "systemone request failed (%s): %s",
            type(exc).__name__,
            exc,
        )
        return _error(exc)
    return JSONResponse(
        content={
            "model": payload["model"],
            "answers": answers,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
            },
        }
    )
