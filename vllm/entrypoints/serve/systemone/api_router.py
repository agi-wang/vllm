# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /v1/systemone`` for a Clef pooling model.

The route encodes the record off the event loop, sends token ids plus span
metadata through ``PoolingParams.extra_kwargs``, and formats the worker's
concatenated logits. vLLM batches the prefills. The worker scores every
sequence that finishes in a step with one published joint-head call.
Request bodies are not logged.
"""

from __future__ import annotations

import asyncio
import uuid
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
                "input_tokens": len(encoded.input_ids),
                "output_tokens": 0,
            },
        }
    )
