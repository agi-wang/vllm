# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision 2.0 Lux and Nox on the Qwen3.5 text backbone.

The published package stores the text weights in ``backbone/`` and the
shared candidate head in ``decision_head.safetensors``. The head stays in
fp32, matching its checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.decision_heads import CandidateHead
from vllm.model_executor.models.decision_runtime import DecisionPooler
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM

logger = init_logger(__name__)


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class Decision2ForDecision(Qwen3_5ForCausalLM):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = int(vllm_config.model_config.hf_text_config.hidden_size)
        self.candidate_head = CandidateHead(hidden)
        self.pooler = DecisionPooler(self._score)

    def _score(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        del token_ids
        positions = [int(position) for position in extra["candidate_positions"]]
        candidates = hidden[positions].unsqueeze(0)
        query = hidden[int(extra["query_position"])].unsqueeze(0)
        return self.candidate_head(candidates, query)[0]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def prefixed():
            for name, tensor in weights:
                if name.startswith(("embed_tokens.", "layers.", "norm.")):
                    name = "model." + name
                if name.startswith("decision_head."):
                    continue
                yield name, tensor

        loaded = super().load_weights(prefixed())
        self._load_candidate_head()
        loaded.update(
            f"candidate_head.{name}"
            for name, _ in self.candidate_head.named_parameters()
        )
        return loaded

    def _load_candidate_head(self) -> None:
        from pathlib import Path

        from safetensors.torch import load_file

        from vllm.transformers_utils.repo_utils import (
            _try_download_from_hf_hub,
            try_get_local_file,
        )

        model = self.model_config.model
        revision = self.model_config.revision
        local = try_get_local_file(
            model, "decision_head.safetensors", revision=revision
        )
        path = local if isinstance(local, Path) and local.is_file() else None
        if path is None:
            path = _try_download_from_hf_hub(
                model, "decision_head.safetensors", revision
            )
        if path is None:
            raise ValueError(f"{model} has no decision_head.safetensors")
        state = load_file(str(path))
        reference = next(self.candidate_head.parameters())
        cast = {key: value.to(device=reference.device) for key, value in state.items()}
        self.candidate_head.load_state_dict(cast, strict=True)
        logger.info("Loaded Decision 2.0 candidate head from %s", path)
