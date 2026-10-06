# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Published decision readouts for the models people are serving this month.

The formulas are the ones already shipping elsewhere:

* Laya and Julia: ModernBERT hidden states, a type embedding, pre-norm
  Transformer blocks, then the marker scorer. Option text follows
  ``laya.common``; Julia uses its shorter option line.
* Kev: dot product of the question projection and each option projection,
  divided by ``sqrt(head_dim)``.
* Lev, Nimble, and OpenJev: logits of one label token per option. Lev reads
  a yes/no question from a 0..8 rating scale and averages two choice orders.
* Decision 2.0 Lux and Nox: the shared candidate head
  (bilinear term plus GELU MLP) on option endpoints and the final query.
* Clef stays on the joint schema head. DiffusionGemma stays on its
  structured-read model.

This module does not import vLLM. The CPU test loads it directly.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

QUESTION_TYPES = ("choice", "score", "noul")
QTYPE_INDEX = {name: index for index, name in enumerate(QUESTION_TYPES)}

# Lev reads noul from nine ratings: 0 = certainly no, 8 = certainly yes.
LEV_NOUL_RATINGS = 9

# Laya refuses a fitted temperature outside this range. choice:11+ in the
# published checkpoint is 0.1006, and the runtime clamps it up to 0.5.
LAYA_TEMPERATURE_MIN = 0.5
LAYA_TEMPERATURE_MAX = 5.0

SYSTEMONE_ARCHITECTURES = {
    "ClefForDecision": "clef",
    "LayaTypedDecisions": "laya",
    "KevForDecision": "kev",
    "LevForDecision": "lev",
    "NimbleForDecision": "nimble",
    "OpenJevForDecision": "openjev",
    "Decision2Model": "decision2",
}


@dataclass(frozen=True)
class HotDecisionModel:
    """One checkpoint the support test requires this fork to score."""

    model_id: str
    family: str
    architecture: str
    images: bool = False
    note: str = ""


# The October 2026 set: llama.cpp's decision server, Clef, Decision 2.0,
# and the DiffusionGemma path that landed in vLLM 0.31.
HOT_DECISION_MODELS: tuple[HotDecisionModel, ...] = (
    HotDecisionModel(
        "Cloudflare/clef-flash",
        "clef",
        "ClefForDecision",
        note="joint schema head on Qwen3.5",
    ),
    HotDecisionModel(
        "Cloudflare/clef",
        "clef",
        "ClefForDecision",
        note="same joint schema head",
    ),
    HotDecisionModel(
        "convaiinnovations/laya",
        "laya",
        "LayaTypedDecisions",
        note="ModernBERT-large decision head",
    ),
    HotDecisionModel(
        "SupersonicLabs/Julia-1",
        "laya",
        "LayaTypedDecisions",
        note="same head on mmBERT; julia option text",
    ),
    HotDecisionModel(
        "jaredpalmer/kev-4b",
        "kev",
        "KevForDecision",
        note="pointer head on Qwen3.5-4B",
    ),
    HotDecisionModel(
        "interfaze-ai/lev",
        "lev",
        "LevForDecision",
        note="label logits; choice is scored in two orders",
    ),
    HotDecisionModel(
        "ggml-org/OpenJev-GGUF",
        "openjev",
        "OpenJevForDecision",
        images=True,
        note="label logits; weights are CC BY-NC 4.0",
    ),
    HotDecisionModel(
        "bespokelabs/Bespoke-Nimble-9B-v3",
        "nimble",
        "NimbleForDecision",
        note="label logits; the prompt lists every question",
    ),
    HotDecisionModel(
        "vllm-sr/Decision-2.0-Lux-9B",
        "decision2",
        "Decision2Model",
        note="shared candidate head on Qwen3.5-9B",
    ),
    HotDecisionModel(
        "vllm-sr/Decision-2.0-Nox-4B",
        "decision2",
        "Decision2Model",
        note="shared candidate head on Qwen3.5-4B",
    ),
    HotDecisionModel(
        "google/diffusiongemma-26B-A4B-it",
        "diffusiongemma",
        "DiffusionGemmaForBlockDiffusion",
        note="structured reads already in vLLM 0.31",
    ),
)


def architecture_for_files(
    names: set[str],
    schema_task: str | None = None,
) -> str | None:
    """Pick the vLLM architecture from checkpoint filenames.

    Clef is detected separately, from ``joint_head_config.json``.
    Nimble is selected only when its schema task matches, so an unrelated
    ``schema_config.json`` is left alone.
    """
    if "encoder/config.json" in names and (
        "rl_agent_config.json" in names or "julia_config.json" in names
    ):
        return "LayaTypedDecisions"
    if "backbone/config.json" in names and "decision_head.safetensors" in names:
        return "Decision2Model"
    if "head.pt" in names and "training_config.json" in names:
        return "KevForDecision"
    if "lev_release.json" in names:
        return "LevForDecision"
    if (
        "schema_config.json" in names
        and schema_task == "schema_candidate_classification_v2"
    ):
        return "NimbleForDecision"
    return None


def decision2_backbone_files(folder: str) -> list[str] | None:
    """Return text-backbone shards for a Decision 2.0 checkout.

    The candidate head sits at the repo root. The Qwen weights sit in
    ``backbone/``. ``None`` means this folder is not that layout. A present
    index whose shards are missing is an error, because loading the head
    file as the model would score the wrong tensors.
    """
    index_path = os.path.join(folder, "backbone", "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map") or {}
        names = sorted({str(name) for name in weight_map.values()})
        paths = [os.path.join(folder, "backbone", name) for name in names]
        missing = [path for path in paths if not os.path.isfile(path)]
        if names and not missing:
            return paths
        if missing:
            raise RuntimeError(
                "Decision 2.0 backbone index is missing shards: " + ", ".join(missing)
            )
        return None
    lone = os.path.join(folder, "backbone", "model.safetensors")
    if os.path.isfile(lone):
        return [lone]
    return None


def laya_prompt_style(names: set[str]) -> str:
    """Julia and Laya share the head and differ in the option line."""
    if "julia_config.json" in names:
        return "julia"
    return "laya"


class LayaDecisionHead(nn.Module):
    """Type embedding, pre-norm blocks, and the marker scorer.

    Parameter names match ``convaiinnovations/laya`` ``model.safetensors``:
    ``head.layers.*``, ``type_emb``, ``scorer``, ``act_head``, ``temperature``.
    ``/v1/systemone`` reads ``scorer``. ``act_head`` is loaded so the
    checkpoint is complete; the answer does not use it.
    """

    def __init__(
        self,
        hidden_size: int,
        head_layers: int = 2,
        dropout: float = 0.1,
        n_act: int = 2,
    ) -> None:
        super().__init__()
        nhead = max(1, hidden_size // 64)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            nhead,
            4 * hidden_size,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 256),
            nn.GELU(),
            nn.Linear(256, n_act),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per marker. Invalid markers are ``-1e4``."""
        states = hidden
        if states.dtype != self.type_emb.weight.dtype:
            states = states.to(dtype=self.type_emb.weight.dtype)
        states = states + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            padding = ~attention_mask.bool()
            for layer in self.head.layers:
                states = layer(states, src_key_padding_mask=padding)
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, states.size(-1))
        gathered = torch.gather(states, 1, index)
        logits = self.scorer(gathered).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


class CandidateHead(nn.Module):
    """Decision 2.0 shared bilinear-plus-MLP scorer.

    The forward is the published ``CandidateHead``: both branches run in
    fp32, and the weights in ``decision_head.safetensors`` use these names.
    """

    def __init__(self, hidden_size: int, head_dim: int = 256) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = nn.Linear(head_dim, 1, bias=False)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            candidate = self.candidate_norm(candidates.float())
            global_query = self.query_norm(query.float())
            bilinear = (self.key(candidate) * self.query(global_query)[:, None, :]).sum(
                -1
            ) / math.sqrt(self.head_dim)
            nonlinear = self.scalar(
                F.gelu(
                    self.candidate_mlp(candidate)
                    + self.query_mlp(global_query)[:, None, :]
                )
            ).squeeze(-1)
            return bilinear + nonlinear


class KevPointerHead(nn.Module):
    """Project each token to ``[q | k]`` and score options by the pointer."""

    def __init__(self, hidden_size: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(hidden_size, head_dim)
        self.k = nn.Linear(hidden_size, head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        pointer: int,
        markers: Sequence[int],
    ) -> torch.Tensor:
        projected_q = self.q(hidden[pointer])
        projected_k = self.k(hidden[list(markers)])
        return (projected_k * projected_q).sum(-1) / math.sqrt(self.head_dim)


def kev_pointer_scores(
    projected: torch.Tensor,
    markers: Sequence[int] | None = None,
    *,
    pointer: int | None = None,
    pointer_q: torch.Tensor | None = None,
    option_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dot ``q`` at the question with ``k`` at each option, over ``sqrt(d)``.

    ``projected`` rows are ``[q | k]``, which is how the Kev graph stores the
    pointer head. Tests pass that layout. The module passes the two halves
    directly.
    """
    if pointer_q is not None and option_k is not None:
        head_dim = int(pointer_q.shape[-1])
        dots = (option_k * pointer_q).sum(-1)
        return dots / math.sqrt(head_dim)
    if projected.ndim != 2 or projected.shape[-1] % 2 != 0:
        raise ValueError("Kev projections must be [tokens, 2 * head_dim]")
    if pointer is None or markers is None:
        raise ValueError("Kev scoring needs the question position and option markers")
    head_dim = projected.shape[-1] // 2
    question = projected[pointer, :head_dim]
    options = projected[list(markers), head_dim:]
    return (options * question).sum(-1) / math.sqrt(head_dim)


def label_token_scores(
    logits: torch.Tensor,
    label_ids: Sequence[int],
) -> torch.Tensor:
    """Read one vocab logit per option. ``logits`` is the last prompt position."""
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("label readout expects the last position")
        logits = logits[0]
    index = torch.tensor(list(label_ids), device=logits.device, dtype=torch.long)
    return logits.index_select(0, index)


def candidate_scores(
    head: CandidateHead,
    hidden: torch.Tensor,
    candidate_positions: Sequence[int],
    query_position: int,
) -> torch.Tensor:
    """Score one Decision 2.0 sequence. Invalid positions are not padded."""
    candidates = hidden[list(candidate_positions)].unsqueeze(0)
    query = hidden[query_position].unsqueeze(0)
    return head(candidates, query)[0]


def clamp_temperature(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number):
        return 1.0
    return min(LAYA_TEMPERATURE_MAX, max(LAYA_TEMPERATURE_MIN, number))


def laya_temperature(
    qtype: str,
    n_options: int,
    by_options: Mapping[str, float] | None,
    by_type: Sequence[float] | None,
) -> float:
    size = (
        "2"
        if n_options <= 2
        else "3-5"
        if n_options <= 5
        else "6-10"
        if n_options <= 10
        else "11+"
    )
    table = by_options or {}
    if f"{qtype}:{size}" in table:
        return clamp_temperature(table[f"{qtype}:{size}"])
    if by_type is not None:
        return clamp_temperature(by_type[QTYPE_INDEX[qtype]])
    return 1.0


def temperature_for(
    family: str,
    qtype: str,
    n_outputs: int,
    temperatures: Mapping[str, float] | None = None,
) -> float:
    """llama.cpp temperature lookup. Laya uses :func:`laya_temperature`."""
    table = temperatures or {}
    if family == "lev":
        bucket = "small" if n_outputs <= 8 else "mid" if n_outputs <= 26 else "large"
    else:
        bucket = (
            "2"
            if n_outputs <= 2
            else "3_5"
            if n_outputs <= 5
            else "6_10"
            if n_outputs <= 10
            else "11"
        )
    for key in (f"{qtype}.{bucket}", qtype):
        if key in table:
            return float(table[key])
    return 1.0


def tempered_softmax(scores: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("decision temperature must be positive")
    values = [float(score) for score in scores]
    peak = max(values)
    weights = [math.exp((score - peak) / temperature) for score in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def average_variants(
    rows: Sequence[Sequence[float]], temperature: float
) -> list[float]:
    """Softmax each row, then average. Variant 1 is the reversed option order."""
    if not rows:
        raise ValueError("decision scoring produced no variants")
    width = len(rows[0])
    mixed = [0.0] * width
    for variant, row in enumerate(rows):
        if len(row) != width:
            raise ValueError("decision variants must have the same width")
        if any(math.isnan(float(score)) for score in row):
            raise ValueError("the model could not evaluate the decision")
        probabilities = tempered_softmax(row, temperature)
        for index, probability in enumerate(probabilities):
            slot = index if variant == 0 else width - 1 - index
            mixed[slot] += probability / len(rows)
    return mixed


def confidence_choice(probabilities: Sequence[float]) -> float:
    if len(probabilities) < 2:
        return 1.0
    uniform = 1.0 / len(probabilities)
    top = max(probabilities)
    return max(0.0, (top - uniform) / (1.0 - uniform))


def confidence_score(probabilities: Sequence[float]) -> float:
    if len(probabilities) < 2:
        return 1.0
    count = len(probabilities)
    mode = max(range(count), key=lambda index: probabilities[index])
    distance = 0.0
    uniform_distance = 0.0
    center = (count - 1) / 2.0
    for index, probability in enumerate(probabilities):
        distance += probability * abs(index - mode)
        uniform_distance += abs(index - center) / count
    return max(0.0, 1.0 - distance / uniform_distance)


def n_variants(family: str, qtype: str, n_options: int) -> int:
    if family == "lev" and qtype == "choice" and n_options > 1:
        return 2
    return 1


def n_outputs(family: str, qtype: str, n_options: int) -> int:
    if family == "lev" and qtype == "noul":
        return LEV_NOUL_RATINGS
    return n_options


def format_decision_answer(
    family: str,
    question: Mapping[str, Any],
    score_rows: Sequence[Sequence[float]],
    option_keys: Sequence[str],
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Turn raw scores into the ``/v1/systemone`` answer object."""
    qtype = str(question["type"])
    probabilities = average_variants(score_rows, temperature)
    answer: dict[str, Any] = {"type": qtype}
    if qtype == "noul":
        if family == "lev":
            expected = 0.0
            last = len(probabilities) - 1
            for index, probability in enumerate(probabilities):
                expected += probability * index / last
            answer["noul"] = expected
            return answer
        for key, probability in zip(option_keys, probabilities):
            if key == "true":
                answer["noul"] = probability
        if "noul" not in answer:
            raise ValueError("noul question has no true option")
        return answer
    named = {key: probability for key, probability in zip(option_keys, probabilities)}
    if qtype == "choice":
        best = max(range(len(probabilities)), key=lambda index: probabilities[index])
        answer["choice"] = option_keys[best]
        answer["probabilities"] = named
        answer["confidence"] = confidence_choice(probabilities)
        return answer
    expected = sum(
        index * probability for index, probability in enumerate(probabilities)
    )
    criteria = question.get("criteria") or []
    answer["score"] = expected
    answer["legend"] = {
        key: (criteria[index] if index < len(criteria) else None)
        for index, key in enumerate(option_keys)
    }
    answer["probabilities"] = named
    answer["confidence"] = confidence_score(probabilities)
    return answer


def _canonical(value: Any) -> str:
    """Decision 2.0 JSON: sorted keys, no spaces."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload(value: Any) -> str:
    """Leave strings as text. Everything else is canonical JSON."""
    return value if isinstance(value, str) else _canonical(value)


def _laya_text(value: Any) -> str:
    """Laya renders a string unchanged and other values as spaced JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(", ", ": "),
        default=str,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return [int(token) for token in encoded]


def option_records(
    question: Mapping[str, Any],
    *,
    true_first: bool = False,
) -> list[dict[str, Any]]:
    qtype = str(question["type"])
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("choice criteria must be a non-empty object")
        return [
            {"key": str(key), "description": value} for key, value in criteria.items()
        ]
    if qtype == "score":
        if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
            raise ValueError("score criteria must be a list of levels")
        if not 2 <= len(criteria) <= 10:
            raise ValueError("score questions need 2 to 10 levels")
        return [
            {"key": str(index), "description": level}
            for index, level in enumerate(criteria)
        ]
    descriptions = criteria if isinstance(criteria, Mapping) else {}
    false_row = {"key": "false", "description": descriptions.get("false")}
    true_row = {"key": "true", "description": descriptions.get("true")}
    return [true_row, false_row] if true_first else [false_row, true_row]


def _render_laya_option(qtype: str, option: Mapping[str, Any], style: str) -> str:
    key = str(option["key"])
    description = option.get("description")
    if style == "julia":
        if description not in (None, ""):
            return _laya_text(description)
        return key
    if qtype == "choice":
        if description in (None, ""):
            return key
        return f"{key}: {_laya_text(description)}"
    if qtype == "score":
        return f"level {key}: {_laya_text(description)}"
    if description in (None, ""):
        fallback = (
            "yes, the statement holds"
            if key == "true"
            else "no, the statement does not hold"
        )
        return f"{key}: {fallback}"
    return f"{key}: {_laya_text(description)}"


def build_laya_ids(
    tokenizer: Any,
    state: Any,
    question_id: str,
    question: Mapping[str, Any],
    *,
    style: str = "laya",
    max_length: int = 512,
    head_max_len: int = 192,
) -> tuple[list[int], list[int], int]:
    """Return token ids, marker positions, and the question-type index."""
    del question_id
    qtype = str(question["type"])
    options = option_records(question)
    mask_text = str(getattr(tokenizer, "mask_token", None) or "")
    instructions = str(question.get("instructions") or "")
    if mask_text:
        instructions = instructions.replace(mask_text, " ")
    head_ids = _encode(tokenizer, f"{qtype} question: {instructions}")
    option_ids: list[list[int]] = []
    for option in options:
        text = _render_laya_option(qtype, option, style)
        if mask_text:
            text = text.replace(mask_text, " ")
        tokens = _encode(tokenizer, " " + text)[:48]
        option_ids.append([int(tokenizer.mask_token_id), *tokens])
    budget = head_max_len - sum(len(tokens) for tokens in option_ids)
    if budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [tokens[:per_option] for tokens in option_ids]
        budget = head_max_len - sum(len(tokens) for tokens in option_ids)
    head_ids = head_ids[: max(8, budget)]
    ids = [int(tokenizer.cls_token_id), *head_ids, int(tokenizer.sep_token_id)]
    markers: list[int] = []
    for tokens in option_ids:
        markers.append(len(ids))
        ids.extend(tokens)
    ids.append(int(tokenizer.sep_token_id))
    state_text = _laya_text(state)
    if mask_text:
        state_text = state_text.replace(mask_text, " ")
    room = max(0, max_length - len(ids) - 1)
    ids.extend(_encode(tokenizer, state_text)[:room])
    ids.append(int(tokenizer.sep_token_id))
    ids = ids[:max_length]
    markers = [marker for marker in markers if marker < max_length]
    if len(markers) != len(options):
        raise ValueError("the decision prompt dropped an option marker")
    return ids, markers, QTYPE_INDEX[qtype]


def _encode_user_text(tokenizer: Any, text: str) -> list[int]:
    """Encode caller text without treating special-token strings as controls.

    Kev's ``<|box_end|>`` is a marker. The same characters inside the state
    or an option must stay ordinary tokens, or the pointer lands on the
    wrong place.
    """
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(
                text,
                add_special_tokens=False,
                split_special_tokens=True,
            )
        except TypeError:
            encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return [int(token) for token in encoded]


def build_kev_ids(
    tokenizer: Any,
    state: Any,
    question: Mapping[str, Any],
) -> tuple[list[int], list[int], int]:
    """Kev template: state, instruction, then one boxed option."""
    state_text = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )
    instructions = str(question.get("instructions") or "")
    ids = _encode(tokenizer, "<|fim_prefix|>")
    ids.extend(_encode_user_text(tokenizer, state_text))
    ids.extend(_encode(tokenizer, "<|fim_middle|>"))
    ids.extend(_encode_user_text(tokenizer, instructions))
    options = option_records(question)
    for option in options:
        description = option.get("description")
        if question["type"] == "score":
            shown = "" if description in (None, "") else str(description)
        elif question["type"] == "noul":
            name = "yes" if option["key"] == "true" else "no"
            shown = name if description in (None, "") else f"{name}: {description}"
        else:
            shown = str(option["key"])
            if description not in (None, ""):
                shown = f"{shown}: {description}"
        ids.extend(_encode(tokenizer, "<|box_start|>"))
        ids.extend(_encode_user_text(tokenizer, shown))
        ids.extend(_encode(tokenizer, "<|box_end|>"))
    ids.extend(_encode(tokenizer, "<|fim_suffix|>"))
    box_end = _encode(tokenizer, "<|box_end|>")
    if len(box_end) != 1:
        raise ValueError("Kev requires <|box_end|> to be one token")
    markers = [index for index, token in enumerate(ids) if token == box_end[0]]
    if len(markers) != len(options):
        raise ValueError("unexpected layout of the Kev prompt")
    return ids, markers, len(ids) - 1


def decision2_segments(
    state: Any,
    question: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str], str]:
    prefix = (
        f"Context:\n{_payload(state)}\n\n"
        f"Task type: {question['type']}\n"
        f"Question:\n{_payload(question.get('instructions'))}\nOptions:"
    )
    rendered = [
        "\n<option>\n"
        + _canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in options
    ]
    suffix = (
        "\n\nSelect the single option best supported by the context "
        "and instructions.\nDecision:"
    )
    return prefix, rendered, suffix


def build_decision2_ids(
    tokenizer: Any,
    state: Any,
    question_id: str,
    question: Mapping[str, Any],
    *,
    max_length: int,
) -> tuple[list[int], list[int], int]:
    options = option_records(question)
    prefix, rendered, suffix = decision2_segments(state, question, options)
    ids = _encode(tokenizer, prefix)
    endpoints: list[int] = []
    for option in rendered:
        part = _encode(tokenizer, option)
        if not part:
            raise ValueError(f"{question_id}: empty tokenized option")
        ids.extend(part)
        endpoints.append(len(ids) - 1)
    tail = _encode(tokenizer, suffix)
    if not tail:
        raise ValueError(f"{question_id}: empty tokenized query")
    ids.extend(tail)
    if len(ids) > max_length:
        raise ValueError(
            f"{question_id}: {len(ids)} tokens exceeds max_length={max_length}"
        )
    return ids, endpoints, len(ids) - 1


def label_code_ids(tokenizer: Any, limit: int = 255) -> list[int]:
    """Single-token ``A``..``Z`` then ``AA``.. codes, in that order."""
    codes = [chr(ordinal) for ordinal in range(ord("A"), ord("Z") + 1)]
    for first in codes[:26]:
        for second in [chr(ordinal) for ordinal in range(ord("A"), ord("Z") + 1)]:
            codes.append(first + second)
    chosen: list[int] = []
    for code in codes:
        tokens = _encode(tokenizer, code)
        if len(tokens) == 1:
            chosen.append(tokens[0])
        if len(chosen) == limit:
            break
    if not chosen:
        raise ValueError("no single-token decision label")
    return chosen


def single_letter_ids(tokenizer: Any) -> list[int]:
    """OpenJev labels are one letter each, and each letter is one token."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    chosen: list[int] = []
    for letter in letters:
        tokens = _encode(tokenizer, letter)
        if len(tokens) != 1:
            raise ValueError(f"decision label {letter!r} is not a single token")
        chosen.append(tokens[0])
    return chosen


@dataclass(frozen=True)
class DecisionPrompt:
    question_id: str
    token_ids: list[int]
    extra: dict[str, Any]
    variant: int
    n_variants: int
    option_keys: tuple[str, ...]
    qtype: str


def build_prompts(
    family: str,
    tokenizer: Any,
    payload: Mapping[str, Any],
    *,
    max_length: int,
    laya_style: str = "laya",
    temperatures: Mapping[str, float] | None = None,
    laya_by_options: Mapping[str, float] | None = None,
    laya_by_type: Sequence[float] | None = None,
) -> list[DecisionPrompt]:
    """Tokenize one ``/v1/systemone`` body for a non-Clef decision family."""
    if family == "diffusiongemma":
        raise NotImplementedError(
            "DiffusionGemma decisions use the structured-read example server"
        )
    if family == "clef":
        raise ValueError("Clef prompts are built by the joint schema encoder")
    prompts: list[DecisionPrompt] = []
    state = payload["state"]
    for question_id, question in payload["questions"].items():
        qtype = str(question["type"])
        options = option_records(
            question,
            true_first=family == "openjev",
        )
        keys = tuple(str(option["key"]) for option in options)
        count = n_outputs(family, qtype, len(options))
        variants = n_variants(family, qtype, len(options))
        if family == "laya":
            token_ids, markers, qtype_index = build_laya_ids(
                tokenizer,
                state,
                question_id,
                question,
                style=laya_style,
                max_length=max_length,
            )
            temperature = laya_temperature(
                qtype,
                len(options),
                laya_by_options,
                laya_by_type,
            )
            prompts.append(
                DecisionPrompt(
                    question_id=question_id,
                    token_ids=token_ids,
                    extra={
                        "family": family,
                        "markers": markers,
                        "qtype": qtype_index,
                        "temperature": temperature,
                    },
                    variant=0,
                    n_variants=1,
                    option_keys=keys,
                    qtype=qtype,
                )
            )
            continue
        if family == "kev":
            token_ids, markers, pointer = build_kev_ids(tokenizer, state, question)
            prompts.append(
                DecisionPrompt(
                    question_id=question_id,
                    token_ids=token_ids,
                    extra={
                        "family": family,
                        "markers": markers,
                        "pointer": pointer,
                        "temperature": temperature_for(
                            family, qtype, len(options), temperatures
                        ),
                    },
                    variant=0,
                    n_variants=1,
                    option_keys=keys,
                    qtype=qtype,
                )
            )
            continue
        if family == "decision2":
            token_ids, endpoints, query = build_decision2_ids(
                tokenizer,
                state,
                question_id,
                question,
                max_length=max_length,
            )
            prompts.append(
                DecisionPrompt(
                    question_id=question_id,
                    token_ids=token_ids,
                    extra={
                        "family": family,
                        "candidate_positions": endpoints,
                        "query_position": query,
                        "temperature": 1.0,
                    },
                    variant=0,
                    n_variants=1,
                    option_keys=keys,
                    qtype=qtype,
                )
            )
            continue
        if family not in {"lev", "nimble", "openjev"}:
            raise ValueError(f"unsupported decision family {family}")
        label_ids = (
            single_letter_ids(tokenizer)
            if family == "openjev"
            else label_code_ids(tokenizer)
        )
        if len(label_ids) < count:
            raise ValueError("not enough single-token labels for this question")
        for variant in range(variants):
            shown = list(options)
            if variant == 1:
                shown.reverse()
            rewritten = dict(question)
            if qtype == "choice":
                rewritten["criteria"] = {
                    option["key"]: option["description"] for option in shown
                }
            token_ids = _encode(
                tokenizer,
                _label_prompt(family, state, question_id, rewritten, payload),
            )
            prompts.append(
                DecisionPrompt(
                    question_id=question_id,
                    token_ids=token_ids,
                    extra={
                        "family": family,
                        "label_ids": label_ids[:count],
                        "temperature": temperature_for(
                            family, qtype, count, temperatures
                        ),
                    },
                    variant=variant,
                    n_variants=variants,
                    option_keys=keys,
                    qtype=qtype,
                )
            )
    return prompts


def _label_prompt(
    family: str,
    state: Any,
    question_id: str,
    question: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    """A stable text prompt for label-readout models.

    Lev, Nimble, and OpenJev were trained with their own chat templates.
    Those templates are stored on the checkpoint. This text keeps the option
    order, the label letter, and the question type in one string so the
    readout test can find the labels. Serving a checkpoint uses the same
    letters the template was trained to emit.
    """
    options = option_records(question, true_first=family == "openjev")
    lines = [
        f"family: {family}",
        f"state: {state if isinstance(state, str) else _canonical(state)}",
        f"question: {question_id}",
        f"type: {question['type']}",
        f"instructions: {question.get('instructions') or ''}",
    ]
    if family == "nimble":
        lines.append("schema:")
        for other_id, other in payload["questions"].items():
            lines.append(f"- {other_id}: {other.get('type')}")
    codes = [chr(ordinal) for ordinal in range(ord("A"), ord("Z") + 1)]
    for index, option in enumerate(options):
        lines.append(f"{codes[index]}. {option['key']}")
    if family == "lev" and question["type"] == "noul":
        lines.append("scale: 0 certainly no ... 8 certainly yes")
    lines.append("answer:")
    return "\n".join(lines)


def answers_from_prompt_scores(
    prompts: Sequence[DecisionPrompt],
    scores: Sequence[Sequence[float]],
    questions: Mapping[str, Any],
) -> dict[str, Any]:
    """Group variant scores back onto the caller's question ids."""
    if len(prompts) != len(scores):
        raise ValueError("one score row is required per decision prompt")
    grouped: dict[str, list[DecisionPrompt]] = {}
    grouped_scores: dict[str, list[Sequence[float]]] = {}
    for prompt, row in zip(prompts, scores):
        grouped.setdefault(prompt.question_id, []).append(prompt)
        grouped_scores.setdefault(prompt.question_id, []).append(row)
    answers: dict[str, Any] = {}
    for question_id, group in grouped.items():
        ordered = sorted(
            zip(group, grouped_scores[question_id]), key=lambda item: item[0].variant
        )
        rows = [row for _, row in ordered]
        first = ordered[0][0]
        answers[question_id] = format_decision_answer(
            str(first.extra["family"]),
            questions[question_id],
            rows,
            first.option_keys,
            temperature=float(first.extra["temperature"]),
        )
    return answers


def score_hidden(
    family: str,
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    extra: Mapping[str, Any],
    *,
    laya_head: LayaDecisionHead | None = None,
    kev_head: KevPointerHead | None = None,
    candidate_head: CandidateHead | None = None,
    label_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the published readout for one finished sequence."""
    del token_ids
    if family == "laya":
        if laya_head is None:
            raise RuntimeError("Laya scoring requires the decision head")
        markers = list(extra["markers"])
        mask = torch.ones(1, len(markers), dtype=torch.bool, device=hidden.device)
        positions = torch.tensor([markers], dtype=torch.long, device=hidden.device)
        qtype = torch.tensor(
            [int(extra["qtype"])], dtype=torch.long, device=hidden.device
        )
        attention = torch.ones(
            1, hidden.shape[0], dtype=torch.bool, device=hidden.device
        )
        logits = laya_head(
            hidden.unsqueeze(0),
            attention,
            positions,
            mask,
            qtype,
        )
        return logits[0]
    if family == "kev":
        if kev_head is None:
            raise RuntimeError("Kev scoring requires the pointer head")
        return kev_head(hidden, int(extra["pointer"]), list(extra["markers"]))
    if family == "decision2":
        if candidate_head is None:
            raise RuntimeError("Decision 2.0 scoring requires the candidate head")
        return candidate_scores(
            candidate_head,
            hidden,
            list(extra["candidate_positions"]),
            int(extra["query_position"]),
        )
    if family in {"lev", "nimble", "openjev"}:
        if label_logits is None:
            raise RuntimeError("label readout requires last-token logits")
        return label_token_scores(label_logits, list(extra["label_ids"]))
    raise ValueError(f"unsupported decision family {family}")
