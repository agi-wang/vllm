# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pick a decision architecture from the files a checkpoint publishes.

Clef is selected from ``joint_head_config.json`` in the caller. Laya and
Decision 2.0 publish a wrapper ``config.json`` whose ``model_type`` is not a
Transformers class, so the encoder or backbone config is loaded instead.
"""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig

from vllm.model_executor.models.decision_heads import (
    architecture_for_files,
    laya_prompt_style,
)
from vllm.transformers_utils.repo_utils import (
    file_or_path_exists,
    get_hf_file_to_dict,
)

# Filenames that distinguish the hot decision checkpoints. A normal model
# that has none of these keeps the architecture in its own config.
DECISION_MARKER_FILES = (
    "encoder/config.json",
    "rl_agent_config.json",
    "julia_config.json",
    "backbone/config.json",
    "decision_head.safetensors",
    "head.pt",
    "training_config.json",
    "lev_release.json",
    "schema_config.json",
)

# Wrapper repos: the file that actually describes the backbone.
NESTED_BACKBONE_CONFIG = {
    "LayaTypedDecisions": "encoder/config.json",
    "Decision2Model": "backbone/config.json",
}


def marker_names(model: str, revision: str | None) -> set[str]:
    """Return the decision marker files present on this model."""
    return {
        name
        for name in DECISION_MARKER_FILES
        if file_or_path_exists(model, name, revision)
    }


def detect_decision(
    model: str,
    revision: str | None,
) -> tuple[str | None, set[str]]:
    """Return ``(architecture, marker files)`` for a decision checkpoint."""
    names = marker_names(model, revision)
    schema_task = None
    if "schema_config.json" in names:
        blob = get_hf_file_to_dict("schema_config.json", model, revision) or {}
        task = blob.get("task") if isinstance(blob, dict) else None
        if isinstance(task, str):
            schema_task = task
    return architecture_for_files(names, schema_task), names


def load_nested_backbone_config(
    model: str,
    revision: str | None,
    architecture: str,
    names: set[str],
) -> tuple[dict[str, Any], PreTrainedConfig]:
    """Load the encoder or backbone config and stamp the decision class.

    The returned config is what vLLM already knows how to build. The wrapper
    at the repo root is not.
    """
    filename = NESTED_BACKBONE_CONFIG[architecture]
    config_dict = get_hf_file_to_dict(filename, model, revision)
    model_type = (
        config_dict.get("model_type") if isinstance(config_dict, dict) else None
    )
    if not isinstance(config_dict, dict) or not isinstance(model_type, str):
        raise ValueError(
            f"{model} selects {architecture} but {filename} could not be read"
        )
    config = _config_class(model_type).from_dict(dict(config_dict))
    config._name_or_path = str(model)
    config.update({"architectures": [architecture]})
    if architecture == "LayaTypedDecisions":
        config.laya_prompt_style = laya_prompt_style(names)
    return config_dict, config


def _config_class(model_type: str) -> type[PreTrainedConfig]:
    from vllm.transformers_utils.config import (  # noqa: PLC0415
        _CONFIG_REGISTRY,
        _register_config_class,
    )

    if model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[model_type]
        _register_config_class(model_type, config_class)
        return config_class
    from transformers.models.auto.configuration_auto import (  # noqa: PLC0415
        CONFIG_MAPPING,
    )

    try:
        return CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(
            f"No config class for decision backbone model_type {model_type!r}"
        ) from exc
