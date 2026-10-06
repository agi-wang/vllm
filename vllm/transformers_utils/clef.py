# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture selection for Cloudflare Clef checkpoints.

Published Clef releases keep ``architectures: ["Qwen3_5ForConditionalGeneration"]``
and store the decision head beside the backbone in ``joint_head_config.json``
and ``joint_head.safetensors``.
"""

CLEF_ARCHITECTURE = "ClefForDecision"

_CLEF_BACKBONES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)


def clef_architecture_override(
    architectures: list[str] | None,
) -> list[str] | None:
    """Return ``["ClefForDecision"]`` when a Qwen3.5 config should load the head.

    ``None`` means the caller should leave ``architectures`` unchanged.
    An explicit ``ClefForDecision`` entry is left alone. A later Hugging Face
    override can still replace the result.
    """
    if not architectures or architectures == [CLEF_ARCHITECTURE]:
        return None
    if len(architectures) == 1 and architectures[0] in _CLEF_BACKBONES:
        return [CLEF_ARCHITECTURE]
    return None
