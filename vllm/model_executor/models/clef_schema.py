# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The prompt format, joint schema head, and SystemOne answer mapping are
# adapted from Cloudflare's Apache-2.0 clef-flash ``joint_schema_model.py``.
"""Text encoding and the Clef joint schema head.

One forward pass scores every typed question. There is no token generation.
``noul``, ``choice``, and ``score`` questions each receive one logit per
allowed option. The HTTP layer softmaxes those logits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]

    def as_extra(self) -> dict[str, Any]:
        """JSON-safe span metadata for ``PoolingParams.extra_kwargs``."""
        return {
            "question_type": self.question_type,
            "question_span": [self.question_span[0], self.question_span[1]],
            "option_spans": [[start, end] for start, end in self.option_spans],
        }


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str


def _tokens(tokenizer: Any, text: str) -> list[int]:
    """Tokenize the way the published encoder does.

    clef-flash calls ``tokenizer(text, add_special_tokens=False)`` and reads
    ``input_ids``. Objects that are not callable keep the ``encode`` path so
    a test double can stand in for that call.
    """
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        ids = getattr(encoded, "input_ids", None)
        if ids is None:
            ids = encoded["input_ids"]
        return [int(token) for token in ids]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        return [int(token) for token in encoded.ids]
    return [int(token) for token in encoded]


def record_from_question_dicts(questions: list[dict[str, Any]]) -> EncodedRecord:
    """Rebuild the span objects ``JointSchemaHead.forward`` indexes."""
    encoded = tuple(
        EncodedQuestion(
            question_id="",
            question_type=int(question["question_type"]),
            question_span=(
                int(question["question_span"][0]),
                int(question["question_span"][1]),
            ),
            option_spans=tuple(
                (int(start), int(end)) for start, end in question["option_spans"]
            ),
            option_ids=tuple("" for _ in question["option_spans"]),
        )
        for question in questions
    )
    return EncodedRecord(input_ids=(), questions=encoded, record_id="")


def collate_finished(
    sequences: list[tuple[torch.Tensor, torch.Tensor, EncodedRecord]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[EncodedRecord]]:
    """Pad finished sequences into one batch for the published head.

    Zeros sit past the attention mask. The head slices each row with
    ``attention_mask.sum()`` before it reads a span, so the pad is unused.
    """
    if not sequences:
        raise ValueError("at least one finished sequence is required")
    lengths = [int(hidden.shape[0]) for hidden, _, _ in sequences]
    width = max(lengths)
    hidden_size = int(sequences[0][0].shape[-1])
    device = sequences[0][0].device
    hidden_states = sequences[0][0].new_zeros((len(sequences), width, hidden_size))
    input_ids = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros(
        (len(sequences), width),
        dtype=torch.long,
        device=device,
    )
    records: list[EncodedRecord] = []
    for index, (hidden, tokens, record) in enumerate(sequences):
        length = lengths[index]
        if int(tokens.shape[0]) != length:
            raise ValueError(
                f"token ids ({int(tokens.shape[0])}) do not match "
                f"hidden states ({length})"
            )
        hidden_states[index, :length] = hidden
        input_ids[index, :length] = tokens.to(device=device, dtype=torch.long)
        attention_mask[index, :length] = 1
        records.append(record)
    return hidden_states, input_ids, attention_mask, records


def validate_systemone_request(request: dict[str, Any]) -> None:
    """Reject a body this build cannot score.

    Images and videos are rejected here. The published checkpoint can see
    them through the Transformers processor; this vLLM path scores text only,
    so a silent drop would change the decision.
    """
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if request.get("images") or request.get("videos"):
        raise NotImplementedError(
            "this Clef build scores text only; images and videos are not accepted"
        )
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"{question_id}: question must be an object")
        if question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
) -> EncodedRecord:
    """Tokenize one text record and record the spans the joint head pools."""
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    question_items = list(record["questions"].items())
    for question_index, (question_id, question) in enumerate(question_items):
        field_header = (
            f"\nFIELD {question_index + 1}\n"
            f"ID: {question_id}\n"
            f"TYPE: {question['type']}\n"
            "INSTRUCTION: "
        )
        schema_ids.extend(_tokens(tokenizer, field_header))
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        options = question_options(question)
        for option_index, (option_id, description) in enumerate(options):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        (
            "\n<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
        ),
    )
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            "schema requires "
            f"{fixed_length} tokens before state; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=(
                question.question_span[0] + schema_offset,
                question.question_span[1] + schema_offset,
            ),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in question.option_spans
            ),
            option_ids=question.option_ids,
        )
        for question in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
    )


def systemone_answer(
    question: dict[str, Any], probabilities: dict[str, float]
) -> dict[str, Any]:
    """Convert per-option probabilities for one question into a SystemOne answer."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {
                option: round(probabilities[option], 4) for option in options
            },
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(
            sum(index * probabilities[level] for index, level in enumerate(levels)),
            4,
        ),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


def answers_from_logits(
    questions: dict[str, Any],
    encoded: EncodedRecord,
    logits: torch.Tensor,
) -> dict[str, Any]:
    """Split one concatenated logit vector into SystemOne answers."""
    flat = logits.detach().float().cpu().reshape(-1)
    offset = 0
    answers: dict[str, Any] = {}
    for question in encoded.questions:
        count = len(question.option_ids)
        piece = flat[offset : offset + count]
        if piece.numel() != count:
            raise RuntimeError(
                "Clef returned "
                f"{flat.numel()} logits for "
                f"{sum(len(item.option_ids) for item in encoded.questions)} options"
            )
        offset += count
        probabilities = dict(
            zip(
                question.option_ids,
                torch.softmax(piece, dim=-1).tolist(),
                strict=True,
            )
        )
        answers[question.question_id] = systemone_answer(
            questions[question.question_id],
            probabilities,
        )
    if offset != flat.numel():
        raise RuntimeError(
            f"Clef returned {flat.numel()} logits; the schema uses {offset}"
        )
    return answers


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    """Joint schema head. Parameter names match ``joint_head.safetensors``."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(
            hidden_size, width, bias=False
        )
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width,
                    heads=heads,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """Score a padded batch. This is the published clef-flash joint head."""
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [
                    self._mean_span(sequence_hidden, question.question_span)
                    for question in record.questions
                ]
            )
            type_ids = torch.tensor(
                [question.question_type for question in record.questions],
                device=hidden_states.device,
            )
            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts = []
            for question in record.questions:
                context_vectors = torch.stack(
                    [
                        self._mean_span(sequence_hidden, span)
                        for span in question.option_spans
                    ]
                )
                lexical_vectors = []
                for start, end in question.option_spans:
                    token_ids = input_ids[batch_index, start:end]
                    lexical_vectors.append(
                        output_embedding_weight[token_ids].mean(dim=0)
                    )
                lexical = torch.stack(lexical_vectors)
                option_contexts.append(context_vectors)
                lexical_options.append(lexical)
                option_counts.append(len(question.option_spans))

            option_queries = []
            for question_index, (context_vectors, lexical) in enumerate(
                zip(option_contexts, lexical_options)
            ):
                option_queries.append(
                    self.option_context_projection(context_vectors)
                    + self.option_lexical_projection(lexical)
                    + self.option_question_projection(
                        question_vectors[question_index]
                    ).unsqueeze(0)
                )
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            routed_options = routed_options[0]
            split_options = list(torch.split(routed_options, option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                routing_weights = torch.softmax(
                    torch.matmul(options, field) / math.sqrt(options.shape[-1]),
                    dim=0,
                )
                option_summaries.append(
                    torch.sum(routing_weights.unsqueeze(-1) * options, dim=0)
                )
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            )
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field, question, lexical, routed in zip(
                fields,
                record.questions,
                lexical_options,
                split_options,
            ):
                anchor = functional.normalize(
                    question_vectors[len(record_logits)] + global_vector,
                    dim=-1,
                )
                lexical_anchor = functional.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(lexical_anchor, anchor)
                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
                features = torch.cat(
                    [
                        repeated_field,
                        options,
                        repeated_field * options,
                        torch.abs(repeated_field - options),
                    ],
                    dim=-1,
                )
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual
                record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
            results.append(record_logits)
        return results

    def score_record(
        self,
        sequence_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        questions: list[dict[str, Any]],
        output_embedding_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Score one unpadded sequence through the published batched forward."""
        hidden, ids, mask, records = collate_finished(
            [
                (
                    sequence_hidden,
                    token_ids,
                    record_from_question_dicts(questions),
                )
            ]
        )
        return self.forward(
            hidden,
            ids,
            mask,
            records,
            output_embedding_weight,
        )[0]
