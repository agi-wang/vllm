# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kev pointer head on a Qwen3.5 text backbone.

``head.pt`` stores ``q`` and ``k`` projections. The question is the last
token. Each option is read at ``<|box_end|>``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.decision_heads import KevPointerHead
from vllm.model_executor.models.decision_runtime import DecisionPooler
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM

logger = init_logger(__name__)


def _json_file(model: str, filename: str, revision: str | None) -> dict[str, Any]:
    from vllm.transformers_utils.repo_utils import (
        _try_download_from_hf_hub,
        try_get_local_file,
    )

    local = try_get_local_file(model, filename, revision=revision)
    path = local if isinstance(local, Path) and local.is_file() else None
    if path is None:
        path = _try_download_from_hf_hub(model, filename, revision)
    if path is None:
        return {}
    return json.loads(path.read_text())


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class KevForDecision(Qwen3_5ForCausalLM):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = int(vllm_config.model_config.hf_text_config.hidden_size)
        training = _json_file(
            vllm_config.model_config.model,
            "training_config.json",
            vllm_config.model_config.revision,
        )
        head_dim = int(training.get("args", {}).get("head_dim", 256))
        self.pointer_head = KevPointerHead(hidden, head_dim)
        self.pooler = DecisionPooler(self._score)

    def _score(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        del token_ids
        return self.pointer_head(
            hidden,
            int(extra["pointer"]),
            [int(position) for position in extra["markers"]],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        self._load_pointer_head()
        loaded.update(
            f"pointer_head.{name}" for name, _ in self.pointer_head.named_parameters()
        )
        return loaded

    def _load_pointer_head(self) -> None:
        from pathlib import Path

        from vllm.transformers_utils.repo_utils import (
            _try_download_from_hf_hub,
            try_get_local_file,
        )

        model = self.model_config.model
        revision = self.model_config.revision
        local = try_get_local_file(model, "head.pt", revision=revision)
        path = local if isinstance(local, Path) and local.is_file() else None
        if path is None:
            path = _try_download_from_hf_hub(model, "head.pt", revision)
        if path is None:
            raise ValueError(f"{model} has no head.pt")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        head = blob["head"]
        head_dim = int(head["q.weight"].shape[0])
        if head_dim != self.pointer_head.head_dim:
            raise ValueError(
                f"Kev head_dim is {head_dim}, config says {self.pointer_head.head_dim}"
            )
        state = {
            "q.weight": head["q.weight"],
            "q.bias": head["q.bias"],
            "k.weight": head["k.weight"],
            "k.bias": head["k.bias"],
        }
        reference = self.pointer_head.q.weight
        cast = {
            key: value.to(device=reference.device, dtype=reference.dtype)
            for key, value in state.items()
        }
        self.pointer_head.load_state_dict(cast, strict=True)
        logger.info("Loaded Kev pointer head from %s", path)
