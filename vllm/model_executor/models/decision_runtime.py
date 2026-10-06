# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool finished prefills with a decision readout.

The backbone forward is what vLLM batches. Each sequence that finishes in
a step is scored by the family's published head. A request without decision
metadata is refused, except the all-zero warmup prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

ScoreOne = Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]


class DecisionPooler(Pooler):
    def __init__(self, score_one: ScoreOne) -> None:
        super().__init__()
        self.pooling = AllPool()
        self.score_one = score_one

    def get_supported_tasks(self) -> set[PoolingTask]:
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
        for index, (hidden, token_ids, params) in enumerate(
            zip(sequences, token_rows, pooling_metadata.pooling_params)
        ):
            if hidden is None:
                continue
            extra = (params.extra_kwargs or {}).get("decision")
            if not extra:
                if token_ids is None or int(token_ids.abs().sum()) != 0:
                    raise RuntimeError(
                        "Decision scoring requires decision metadata in "
                        "PoolingParams.extra_kwargs. Use POST /v1/systemone."
                    )
                outputs[index] = hidden.new_zeros((1,))
                continue
            if token_ids is None:
                raise RuntimeError("Decision scoring requires prompt token ids")
            outputs[index] = self.score_one(hidden, token_ids, extra)
        return outputs
