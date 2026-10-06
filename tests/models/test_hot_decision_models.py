# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the decision heads this fork scores.

The module under test does not import vLLM. Architecture registration is
checked from source so this file can run with ``--noconftest``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heads = _load(
    "decision_heads_under_test",
    _ROOT / "vllm/model_executor/models/decision_heads.py",
)


class _CharTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        del add_special_tokens, kwargs
        return [ord(char) for char in text]


class _KevTokenizer:
    """Control strings are one token. User text is split, even the same strings."""

    specials = {
        "<|fim_prefix|>": 1,
        "<|fim_middle|>": 2,
        "<|box_start|>": 3,
        "<|box_end|>": 4,
        "<|fim_suffix|>": 5,
    }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ):
        del add_special_tokens
        if split_special_tokens:
            return [100 + ord(char) for char in text]
        ids: list[int] = []
        index = 0
        keys = sorted(self.specials, key=len, reverse=True)
        while index < len(text):
            matched = False
            for key in keys:
                if text.startswith(key, index):
                    ids.append(self.specials[key])
                    index += len(key)
                    matched = True
                    break
            if not matched:
                ids.append(100 + ord(text[index]))
                index += 1
        return ids


def _questions():
    return {
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
    }


def _manual_candidate(head, candidates: torch.Tensor, query: torch.Tensor):
    candidate = head.candidate_norm(candidates.float())
    global_query = head.query_norm(query.float())
    bilinear = (head.key(candidate) * head.query(global_query)[:, None, :]).sum(
        -1
    ) / math.sqrt(head.head_dim)
    nonlinear = head.scalar(
        F.gelu(head.candidate_mlp(candidate) + head.query_mlp(global_query)[:, None, :])
    ).squeeze(-1)
    return bilinear + nonlinear


def test_hot_catalog_covers_the_october_models_and_leaves_diffusiongemma_out():
    catalog = heads.HOT_DECISION_MODELS
    assert len(catalog) == 11
    families = {item.family for item in catalog}
    assert families == {
        "clef",
        "laya",
        "kev",
        "lev",
        "openjev",
        "nimble",
        "decision2",
        "diffusiongemma",
    }
    architectures = {item.architecture for item in catalog}
    assert architectures == set(heads.SYSTEMONE_ARCHITECTURES) | {
        "DiffusionGemmaForBlockDiffusion"
    }
    assert "DiffusionGemmaForBlockDiffusion" not in heads.SYSTEMONE_ARCHITECTURES
    for item in catalog:
        if item.family == "diffusiongemma":
            continue
        assert heads.SYSTEMONE_ARCHITECTURES[item.architecture] == item.family
    openjev = next(item for item in catalog if item.family == "openjev")
    assert openjev.images is True


def test_marker_files_select_one_architecture():
    select = heads.architecture_for_files
    assert (
        select({"encoder/config.json", "rl_agent_config.json"}) == "LayaTypedDecisions"
    )
    assert select({"encoder/config.json", "julia_config.json"}) == "LayaTypedDecisions"
    assert heads.laya_prompt_style({"julia_config.json"}) == "julia"
    assert heads.laya_prompt_style({"rl_agent_config.json"}) == "laya"
    assert (
        select({"backbone/config.json", "decision_head.safetensors"})
        == "Decision2Model"
    )
    assert select({"head.pt", "training_config.json"}) == "KevForDecision"
    assert select({"lev_release.json"}) == "LevForDecision"
    assert (
        select(
            {"schema_config.json"},
            schema_task="schema_candidate_classification_v2",
        )
        == "NimbleForDecision"
    )
    assert select({"schema_config.json"}, schema_task="something_else") is None
    assert select(set()) is None


def test_laya_head_matches_published_names_and_masks_missing_markers():
    torch.manual_seed(0)
    named = heads.LayaDecisionHead(64, head_layers=2)
    keys = set(named.state_dict())
    for name in (
        "type_emb.weight",
        "temperature",
        "scorer.0.weight",
        "scorer.1.weight",
        "scorer.3.weight",
        "act_head.0.weight",
        "act_head.2.weight",
        "head.layers.0.linear1.weight",
        "head.layers.0.linear2.weight",
        "head.layers.0.self_attn.in_proj_weight",
        "head.layers.0.self_attn.out_proj.weight",
        "head.layers.0.norm1.weight",
        "head.layers.1.norm2.weight",
    ):
        assert name in keys

    head = heads.LayaDecisionHead(32, head_layers=0)
    head.eval()
    with torch.no_grad():
        head.type_emb.weight.zero_()
        head.type_emb.weight[0, 0] = 3
        head.type_emb.weight[1, 0] = -3
        head.scorer[1].weight.zero_()
        head.scorer[1].weight[0, 0] = 1
        head.scorer[1].bias.zero_()
        head.scorer[3].weight.zero_()
        head.scorer[3].weight[0, 0] = 1
        head.scorer[3].bias.zero_()
    hidden = torch.zeros(2, 4, 32)
    markers = torch.tensor([[1, 2], [1, 2]])
    mask = torch.tensor([[True, False], [True, True]])
    attention = torch.ones(2, 4, dtype=torch.bool)
    first = head(hidden, attention, markers, mask, torch.tensor([0, 0]))
    second = head(hidden, attention, markers, mask, torch.tensor([1, 0]))
    assert first.shape == (2, 2)
    assert first[0, 1].item() == -1e4
    assert first[0, 0].item() != second[0, 0].item()
    assert first[1, 0].item() == second[1, 0].item()


def test_candidate_head_matches_the_bilinear_plus_mlp_formula():
    torch.manual_seed(1)
    head = heads.CandidateHead(8, head_dim=4)
    head.eval()
    candidates = torch.randn(2, 3, 8)
    query = torch.randn(2, 8)
    scored = head(candidates, query)
    expected = _manual_candidate(head, candidates, query)
    assert scored.shape == (2, 3)
    assert torch.allclose(scored, expected)

    hidden = torch.randn(6, 8)
    row = heads.candidate_scores(head, hidden, [1, 4], 5)
    again = _manual_candidate(
        head,
        hidden[[1, 4]].unsqueeze(0),
        hidden[5].unsqueeze(0),
    )[0]
    assert torch.allclose(row, again)


def test_kev_pointer_is_a_scaled_dot_and_user_text_is_not_a_marker():
    projected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [3.0, 4.0, 9.0, 9.0],
        ]
    )
    scores = heads.kev_pointer_scores(projected, [0, 1], pointer=2)
    question = projected[2, :2]
    options = projected[[0, 1], 2:]
    expected = (options * question).sum(-1) / math.sqrt(2)
    assert torch.allclose(scores, expected)

    torch.manual_seed(2)
    pointer = heads.KevPointerHead(5, 4)
    pointer.eval()
    hidden = torch.randn(6, 5)
    markers = [1, 3]
    direct = pointer(hidden, 5, markers)
    split = heads.kev_pointer_scores(
        projected.new_zeros((1, 1)),
        pointer_q=pointer.q(hidden[5]),
        option_k=pointer.k(hidden[markers]),
    )
    assert torch.allclose(direct, split)

    ids, found, last = heads.build_kev_ids(
        _KevTokenizer(),
        "see <|box_end|> now",
        _questions()["department"],
    )
    assert found == [index for index, token in enumerate(ids) if token == 4]
    assert len(found) == 2
    assert last == len(ids) - 1
    assert ids[last] == 5


def test_label_readout_and_lev_reverse_variant():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    picked = heads.label_token_scores(logits, [3, 1])
    assert torch.allclose(picked, torch.tensor([0.4, 0.2]))

    mixed = heads.average_variants([[2.0, 0.0], [0.0, 2.0]], 1.0)
    forward = heads.tempered_softmax([2.0, 0.0], 1.0)
    assert mixed == forward
    forgotten = [
        (
            heads.tempered_softmax([2.0, 0.0], 1.0)[index]
            + heads.tempered_softmax([0.0, 2.0], 1.0)[index]
        )
        / 2
        for index in range(2)
    ]
    assert mixed != forgotten

    nine = [0.0] * 8 + [10.0]
    answer = heads.format_decision_answer(
        "lev",
        {"type": "noul", "instructions": "outage"},
        [nine],
        ("false", "true"),
        temperature=1.0,
    )
    probabilities = heads.tempered_softmax(nine, 1.0)
    expected = sum(
        index * probability for index, probability in enumerate(probabilities)
    )
    expected /= len(probabilities) - 1
    assert answer["noul"] == expected
    assert answer["noul"] > 0.99

    other = heads.format_decision_answer(
        "openjev",
        {"type": "noul", "instructions": "outage"},
        [[0.0, 2.0]],
        ("true", "false"),
        temperature=1.0,
    )
    assert other["noul"] == heads.tempered_softmax([0.0, 2.0], 1.0)[0]


def test_laya_temperature_clamp_and_prompt_styles():
    assert heads.clamp_temperature(0.1006) == 0.5
    assert heads.clamp_temperature(True) == 1.0
    assert heads.clamp_temperature(9) == 5.0
    assert heads.laya_temperature("choice", 12, {"choice:11+": 0.10058}, None) == 0.5
    assert heads.n_variants("lev", "choice", 2) == 2
    assert heads.n_variants("lev", "noul", 2) == 1
    assert heads.n_outputs("lev", "noul", 2) == 9

    payload = {"state": {"host": "a"}, "questions": _questions()}
    prompts = heads.build_prompts(
        "lev",
        _CharTokenizer(),
        payload,
        max_length=4096,
        temperatures={"choice.small": 0.7, "noul.mid": 1.4},
    )
    department = [prompt for prompt in prompts if prompt.question_id == "department"]
    assert [prompt.variant for prompt in department] == [0, 1]
    first = "".join(chr(token) for token in department[0].token_ids)
    second = "".join(chr(token) for token in department[1].token_ids)
    assert "A. technical" in first and "B. billing" in first
    assert "A. billing" in second and "B. technical" in second
    assert department[0].extra["temperature"] == 0.7
    outage = next(prompt for prompt in prompts if prompt.question_id == "outage")
    assert len(outage.extra["label_ids"]) == 9
    assert outage.extra["temperature"] == 1.4

    julia_ids, _, _ = heads.build_laya_ids(
        _CharTokenizer(),
        "state",
        "department",
        _questions()["department"],
        style="julia",
    )
    laya_ids, markers, qtype = heads.build_laya_ids(
        _CharTokenizer(),
        "state",
        "department",
        _questions()["department"],
        style="laya",
    )
    julia_text = "".join(chr(token) for token in julia_ids)
    laya_text = "".join(chr(token) for token in laya_ids)
    assert "Engineering" in julia_text
    assert "technical: Engineering" not in julia_text
    assert "technical: Engineering" in laya_text
    assert markers == [index for index, token in enumerate(laya_ids) if token == 3]
    assert qtype == 0
    assert laya_ids[0] == 1


def test_decision2_backbone_files_skip_the_root_head(tmp_path: Path):
    root = tmp_path
    (root / "decision_head.safetensors").write_bytes(b"head")
    backbone = root / "backbone"
    backbone.mkdir()
    shard = backbone / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
        }
    }
    (backbone / "model.safetensors.index.json").write_text(json.dumps(index))
    found = heads.decision2_backbone_files(str(root))
    assert found == [str(shard)]
    assert heads.decision2_backbone_files(str(root / "missing-dir-ok")) is None
    (backbone / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.norm.weight": "missing.safetensors"}}'
    )
    try:
        heads.decision2_backbone_files(str(root))
    except RuntimeError as exc:
        assert "missing shards" in str(exc)
    else:
        raise AssertionError("a Decision 2.0 index with a missing shard must fail")


def test_decision2_prompt_matches_the_published_segments():
    prefix, options, suffix = heads.decision2_segments(
        {"b": 1, "a": 2},
        {"type": "choice", "instructions": "Which?"},
        [{"key": "technical", "description": "Engineering"}],
    )
    assert prefix == (
        'Context:\n{"a":2,"b":1}\n\nTask type: choice\nQuestion:\nWhich?\nOptions:'
    )
    assert options == [
        '\n<option>\n{"description":"Engineering","key":"technical"}\n</option>'
    ]
    assert suffix == (
        "\n\nSelect the single option best supported by the context "
        "and instructions.\nDecision:"
    )
    ids, endpoints, query = heads.build_decision2_ids(
        _CharTokenizer(),
        "state",
        "department",
        _questions()["department"],
        max_length=4096,
    )
    assert endpoints
    assert query == len(ids) - 1
    try:
        heads.build_decision2_ids(
            _CharTokenizer(),
            "state",
            "department",
            _questions()["department"],
            max_length=8,
        )
    except ValueError as exc:
        assert "no truncation" not in str(exc)
        assert "exceeds max_length" in str(exc)
    else:
        raise AssertionError("long Decision 2.0 prompts must not be truncated")

    try:
        heads.build_prompts("clef", _CharTokenizer(), {}, max_length=8)
    except ValueError:
        pass
    else:
        raise AssertionError("Clef prompts stay on the joint encoder")
    try:
        heads.build_prompts("diffusiongemma", _CharTokenizer(), {}, max_length=8)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("DiffusionGemma is not a systemone family")


def test_registered_architectures_and_ci_cover_the_same_models():
    registry = (_ROOT / "vllm/model_executor/models/registry.py").read_text()
    config = (_ROOT / "vllm/transformers_utils/config.py").read_text()
    loader = (_ROOT / "vllm/model_executor/model_loader/default_loader.py").read_text()
    router = (_ROOT / "vllm/entrypoints/serve/systemone/api_router.py").read_text()
    workflow = (_ROOT / ".github/workflows/clef.yml").read_text()
    for item in heads.HOT_DECISION_MODELS:
        assert item.architecture in registry
    for architecture in heads.SYSTEMONE_ARCHITECTURES:
        assert architecture in registry
    assert "detect_decision" in config
    assert "encoder/config.json" in config or "NESTED_BACKBONE_CONFIG" in config
    assert "decision2_backbone_files" in loader
    assert "backbone" in loader and "decision_head.safetensors" in loader
    assert "clef_questions" in router
    assert "build_prompts" in router
    assert "skip_reading_prefix_cache=True" in router
    assert "tests/models/test_hot_decision_models.py" in workflow

    classes = {
        "laya_decision.py": {"LayaForDecision"},
        "decision2.py": {"Decision2ForDecision"},
        "kev_decision.py": {"KevForDecision"},
        "label_decision.py": {"LabelDecisionModel", "OpenJevForDecision"},
        "decision_runtime.py": {"DecisionPooler"},
    }
    models = _ROOT / "vllm/model_executor/models"
    for filename, expected in classes.items():
        path = models / filename
        found = {
            node.name
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef)
        }
        assert expected <= found
        text = path.read_text()
        assert "DecisionPooler" in text
        if filename != "decision_runtime.py":
            assert "is_pooling_model = True" in text
