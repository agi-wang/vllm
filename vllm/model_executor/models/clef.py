# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clef decision model on the Qwen3.5 prefill path.

Clef-flash is a Qwen3.5 backbone plus ``joint_head.safetensors``. The head
is the published batched joint schema forward: it mean-pools question and
option spans from the full prefill hidden states and mixes them with
output-embedding vectors. It does not decode tokens, so KV cache does not
apply. Concurrent requests are batched prefills. Every sequence that
finishes in one step is scored by a single head call.

Serve the published checkpoint with::

    vllm serve Cloudflare/clef-flash --served-model-name clef-flash

``joint_head_config.json`` selects ``ClefForDecision`` and the pooling runner.
Score requests with ``POST /v1/systemone``. Text only. Tensor parallel size
must be 1 because the head gathers every option token from the untied
output embedding on one rank. Compare probabilities with the Transformers
reference before replacing it: Qwen3.5's linear-attention prefill in vLLM
is not bit-identical to Transformers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models.clef_schema import (
    EncodedRecord,
    JointSchemaHead,
    collate_finished,
    record_from_question_dicts,
)
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.tasks import PoolingTask
from vllm.transformers_utils.repo_utils import (
    _try_download_from_hf_hub,
    try_get_local_file,
)
from vllm.v1.pool.metadata import PoolingMetadata

logger = init_logger(__name__)

_WARMUP_QUESTION = {
    "question_type": 0,
    "question_span": [0, 1],
    "option_spans": [[0, 1]],
}


def _joint_head_file(model: str, filename: str, revision: str | None) -> Path | None:
    local = try_get_local_file(model, filename, revision=revision)
    if isinstance(local, Path) and local.is_file():
        return local
    return _try_download_from_hf_hub(model, filename, revision)


class ClefSchemaPooler(Pooler):
    """Score every sequence that finishes in this step with one head call."""

    def __init__(self, score_fn: Callable[..., list[list[torch.Tensor]]]) -> None:
        super().__init__()
        self.pooling = AllPool()
        # A bound method must not be registered as a child module.
        self.score_fn = score_fn

    def get_supported_tasks(self) -> set[PoolingTask]:
        # token_classify is the supported pooling task that returns logits
        # without the token-embed L2 normalize. Only /v1/systemone supplies
        # the span metadata this head needs.
        return {"token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[torch.Tensor | None]:
        sequences = self.pooling(hidden_states, pooling_metadata)
        token_rows = pooling_metadata.get_prompt_token_ids_cpu()
        outputs: list[torch.Tensor | None] = [None] * len(sequences)
        finished: list[tuple[torch.Tensor, torch.Tensor, EncodedRecord]] = []
        finished_slots: list[int] = []
        for index, (hidden, token_ids_cpu, params) in enumerate(
            zip(sequences, token_rows, pooling_metadata.pooling_params)
        ):
            if hidden is None:
                continue
            extra = params.extra_kwargs or {}
            questions = extra.get("clef_questions")
            if not questions:
                # Engine warmup builds a prompt of zeros and has no schema.
                if token_ids_cpu is None or int(token_ids_cpu.abs().sum()) != 0:
                    raise RuntimeError(
                        "Clef scoring requires clef_questions in "
                        "PoolingParams.extra_kwargs. Use POST /v1/systemone."
                    )
                if hidden.shape[0] < 1:
                    outputs[index] = hidden.new_zeros((1,))
                    continue
                questions = [_WARMUP_QUESTION]
            if token_ids_cpu is None:
                raise RuntimeError("Clef scoring requires prompt token ids")
            finished.append(
                (hidden, token_ids_cpu, record_from_question_dicts(questions))
            )
            finished_slots.append(index)
        if not finished:
            return outputs
        hidden_batch, input_ids, attention_mask, records = collate_finished(finished)
        scored = self.score_fn(hidden_batch, input_ids, attention_mask, records)
        for slot, logits in zip(finished_slots, scored):
            outputs[slot] = torch.cat(logits, dim=0)
        return outputs


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ClefForDecision(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 prefill plus the Clef joint schema head."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        head_config = self._read_head_config(
            model_config.model,
            model_config.revision,
        )
        self.joint_head = JointSchemaHead(**head_config)
        self.joint_head.to(dtype=model_config.dtype)
        self.pooler = ClefSchemaPooler(self._score_questions)

    @staticmethod
    def _read_head_config(model: str, revision: str | None) -> dict[str, Any]:
        path = _joint_head_file(model, "joint_head_config.json", revision)
        if path is None:
            raise ValueError(
                f"{model} has no joint_head_config.json. "
                "ClefForDecision needs the published joint schema head."
            )
        config = json.loads(path.read_text())
        allowed = {
            "hidden_size",
            "width",
            "routing_layers",
            "layers",
            "heads",
            "feedforward",
            "dropout",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"joint_head_config.json has unknown keys: {sorted(unknown)}"
            )
        return config

    def _output_embedding_weight(self) -> torch.Tensor:
        lm_head = self.language_model.lm_head
        if getattr(lm_head, "tp_size", 1) != 1:
            raise RuntimeError(
                "Clef joint head requires --tensor-parallel-size 1 so option "
                "tokens can index the full untied output embedding."
            )
        weight = lm_head.weight
        if weight is None:
            raise RuntimeError("Clef output embedding weight is not loaded")
        return weight

    def _score_questions(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
    ) -> list[list[torch.Tensor]]:
        hidden = hidden_states
        head_dtype = self.joint_head.hidden_norm.weight.dtype
        if hidden.dtype != head_dtype:
            hidden = hidden.to(dtype=head_dtype)
        return self.joint_head(
            hidden,
            input_ids,
            attention_mask,
            records,
            self._output_embedding_weight(),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        self._load_joint_head()
        loaded.update(
            f"joint_head.{name}" for name, _ in self.joint_head.named_parameters()
        )
        return loaded

    def _load_joint_head(self) -> None:
        model_config = self.model_config
        path = _joint_head_file(
            model_config.model,
            "joint_head.safetensors",
            model_config.revision,
        )
        if path is None:
            raise ValueError(f"{model_config.model} has no joint_head.safetensors.")
        from safetensors.torch import load_file

        state = load_file(str(path))
        reference = next(self.joint_head.parameters())
        cast_state = {
            key: value.to(device=reference.device, dtype=reference.dtype)
            for key, value in state.items()
        }
        self.joint_head.load_state_dict(cast_state, strict=True)
        logger.info("Loaded Clef joint schema head from %s", path)
