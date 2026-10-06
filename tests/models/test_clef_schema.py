# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


clef_arch = _load("clef_arch", _ROOT / "vllm/transformers_utils/clef.py")
clef_schema = _load("clef_schema", _ROOT / "vllm/model_executor/models/clef_schema.py")


class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _record():
    return {
        "id": "sample",
        "model": "clef-flash",
        "state": {"host": "a"},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"technical": "Engineering", "billing": "Accounts"},
            },
            "outage": {"type": "noul", "instructions": "Is there an outage?"},
            "urgency": {
                "type": "score",
                "instructions": "How soon?",
                "criteria": ["Later", "Today", "Now"],
            },
        },
    }


def test_architecture_override_selects_clef_for_published_backbone():
    override = clef_arch.clef_architecture_override
    assert override(["Qwen3_5ForConditionalGeneration"]) == ["ClefForDecision"]
    assert override(["Qwen3_5MoeForConditionalGeneration"]) == ["ClefForDecision"]
    assert override(["ClefForDecision"]) is None
    assert override(["LlamaForCausalLM"]) is None
    assert override(None) is None


def test_encode_record_shifts_spans_past_state():
    encoded = clef_schema.encode_record(_CharTokenizer(), _record(), max_length=4096)
    prefix = clef_schema._tokens(
        _CharTokenizer(),
        f"<|im_start|>system\n{clef_schema.SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nSTATE:\n",
    )
    state = clef_schema._tokens(
        _CharTokenizer(), clef_schema.render(_record()["state"])
    )
    offset = len(prefix) + len(state)
    schema_mark = clef_schema._tokens(_CharTokenizer(), "\n\nSCHEMA FIELDS:\n")
    assert encoded.input_ids[offset : offset + len(schema_mark)] == tuple(schema_mark)
    assert [question.question_id for question in encoded.questions] == [
        "department",
        "outage",
        "urgency",
    ]
    assert encoded.questions[0].option_ids == ("billing", "technical")
    assert encoded.questions[1].option_ids == ("true", "false")
    assert encoded.questions[2].option_ids == ("0", "1", "2")
    for question in encoded.questions:
        assert question.question_span[0] >= offset
        for start, end in question.option_spans:
            assert offset <= start < end <= len(encoded.input_ids)


def test_validate_rejects_media_and_empty_questions():
    with pytest.raises(NotImplementedError):
        clef_schema.validate_systemone_request({**_record(), "images": ["x"]})
    with pytest.raises(ValueError, match="at least one question"):
        clef_schema.validate_systemone_request({**_record(), "questions": {}})


def test_systemone_answers_match_option_order():
    questions = _record()["questions"]
    choice = clef_schema.systemone_answer(
        questions["department"],
        {"billing": 0.2, "technical": 0.8},
    )
    assert choice["choice"] == "technical"
    assert list(choice["probabilities"]) == ["technical", "billing"]
    noul = clef_schema.systemone_answer(
        questions["outage"],
        {"true": 0.8363, "false": 0.1637},
    )
    assert noul == {"type": "noul", "noul": 0.8363}
    score = clef_schema.systemone_answer(
        questions["urgency"],
        {"0": 0.1, "1": 0.2, "2": 0.7},
    )
    assert score["score"] == 1.6
    assert score["legend"]["1"] == "Today"


def test_joint_head_returns_one_logit_vector_per_question():
    torch.manual_seed(0)
    head = clef_schema.JointSchemaHead(
        hidden_size=8,
        width=8,
        routing_layers=1,
        layers=1,
        heads=2,
        feedforward=16,
    ).eval()
    names = set(head.state_dict())
    assert "hidden_norm.weight" in names
    assert "prior_logit_scale" in names
    assert "evidence_layers.0.attention.in_proj_weight" in names
    assert "layers.0.linear1.weight" in names
    assert "residual_scorer.0.weight" in names

    encoded = clef_schema.encode_record(_CharTokenizer(), _record(), max_length=4096)
    hidden = torch.randn(len(encoded.input_ids), 8)
    embed = torch.randn(512, 8)
    token_ids = torch.tensor(encoded.input_ids)
    logits = head.score_record(
        hidden,
        token_ids,
        [question.as_extra() for question in encoded.questions],
        embed,
    )
    assert [tuple(item.shape) for item in logits] == [(2,), (2,), (3,)]
    flat = torch.cat(logits)
    answers = clef_schema.answers_from_logits(_record()["questions"], encoded, flat)
    assert set(answers) == {"department", "outage", "urgency"}
    assert answers["department"]["type"] == "choice"
    assert abs(sum(answers["department"]["probabilities"].values()) - 1) < 1e-3
    again = head.score_record(
        hidden,
        token_ids,
        [question.as_extra() for question in encoded.questions],
        embed,
    )
    assert torch.equal(torch.cat(logits), torch.cat(again))
