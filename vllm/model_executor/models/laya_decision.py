# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Laya and Julia on the ModernBERT pooling path.

Both checkpoints are an encoder plus the published decision head. The root
``model.safetensors`` stores the encoder under ``encoder.`` and the head
under ``head.``, ``type_emb``, ``scorer``, and ``act_head``. Julia uses the
same tensors and a shorter option line.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.decision_heads import LayaDecisionHead
from vllm.model_executor.models.decision_runtime import DecisionPooler
from vllm.model_executor.models.interfaces_base import (
    attn_type,
    default_pooling_type,
)
from vllm.model_executor.models.modernbert import ModernBertModel
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.repo_utils import (
    _try_download_from_hf_hub,
    try_get_local_file,
)

logger = init_logger(__name__)


def _json_file(model: str, filename: str, revision: str | None) -> dict[str, Any]:
    local = try_get_local_file(model, filename, revision=revision)
    path = local if isinstance(local, Path) and local.is_file() else None
    if path is None:
        path = _try_download_from_hf_hub(model, filename, revision)
    if path is None:
        return {}
    return json.loads(path.read_text())


@attn_type("encoder_only")
@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class LayaForDecision(torch.nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.model = ModernBertModel(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config
        decision = _json_file(config.model, "rl_agent_config.json", config.revision)
        if not decision:
            decision = _json_file(config.model, "julia_config.json", config.revision)
        hidden = int(config.hf_config.hidden_size)
        self.decision_head = LayaDecisionHead(
            hidden,
            head_layers=int(decision.get("head_layers", 2)),
        )
        self.pooler = DecisionPooler(self._score)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def _score(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        del token_ids
        markers = [int(position) for position in extra["markers"]]
        device = hidden.device
        return self.decision_head(
            hidden.unsqueeze(0),
            torch.ones(1, hidden.shape[0], dtype=torch.bool, device=device),
            torch.tensor([markers], dtype=torch.long, device=device),
            torch.ones(1, len(markers), dtype=torch.bool, device=device),
            torch.tensor([int(extra["qtype"])], dtype=torch.long, device=device),
        )[0]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        encoder: list[tuple[str, torch.Tensor]] = []
        head: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name.startswith("encoder."):
                encoder.append((name[len("encoder.") :], tensor))
            elif name.split(".", 1)[0] in {
                "head",
                "type_emb",
                "scorer",
                "act_head",
                "temperature",
            }:
                head[name] = tensor
        loaded = self.model.load_weights(encoder)
        reference = self.decision_head.type_emb.weight
        cast = {
            key: value.to(device=reference.device, dtype=reference.dtype)
            if key != "temperature"
            else value.to(device=reference.device)
            for key, value in head.items()
        }
        missing, unexpected = self.decision_head.load_state_dict(cast, strict=False)
        if unexpected or any(not name.startswith("act_head.") for name in missing):
            # act_head is unused by the answer. Every other head tensor is required.
            required_missing = [
                name for name in missing if not name.startswith("act_head.")
            ]
            if required_missing or unexpected:
                raise ValueError(
                    "Laya decision head did not match the checkpoint: "
                    f"missing {required_missing}, unexpected {unexpected}"
                )
        loaded.update(f"decision_head.{name}" for name in cast)
        logger.info("Loaded Laya decision head (%d tensors)", len(cast))
        return loaded
