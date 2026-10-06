# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Label-logit decision models: Lev, Nimble, and OpenJev.

The answer is not a new tensor. It is the backbone's next-token logits at
the label ids, read at the last prompt token. Lev averages two option
orders. OpenJev uses the vision-capable Qwen3.5 backbone; this route still
rejects image bytes so a dropped image cannot change the decision.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.decision_heads import label_token_scores
from vllm.model_executor.models.decision_runtime import DecisionPooler
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class _LabelReadout:
    def _score(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        del token_ids
        logits = self.compute_logits(hidden[-1:])
        if logits is None:
            raise RuntimeError("label readout produced no logits")
        return label_token_scores(logits, [int(token) for token in extra["label_ids"]])


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class LabelDecisionModel(_LabelReadout, Qwen3_5ForCausalLM):
    """Lev and Nimble. The weights are a Qwen3.5 text backbone."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = DecisionPooler(self._score)


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class OpenJevForDecision(_LabelReadout, Qwen3_5ForConditionalGeneration):
    """OpenJev text readout on the Qwen3.5 vision backbone."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = DecisionPooler(self._score)
