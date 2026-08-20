from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("validate_contract.py")
SPEC = importlib.util.spec_from_file_location("validate_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
WORKSPACE = Path("/dev/shm/kimi-k3")


@pytest.fixture
def config():
    return json.loads((WORKSPACE / "moonshot" / "kimi-k3" / "config.json").read_text())


def test_exact_schedule_and_boundaries(config):
    result = MODULE.validate_config(config)
    assert len(result["kda_zero_based"]) == 69
    assert len(result["mla_zero_based"]) == 24
    assert result["boundaries"] == {0: "kda_dense", 1: "kda_moe", 2: "kda_moe", 3: "mla_moe", 91: "mla_moe", 92: "mla_moe"}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("num_hidden_layers",), 92),
        (("num_experts",), 895),
        (("num_experts_per_token",), 15),
        (("routed_expert_hidden_size",), 3583),
    ],
)
def test_rejects_invalid_scalar_invariants(config, path, value):
    invalid = copy.deepcopy(config)
    invalid["text_config"][path[0]] = value
    with pytest.raises(MODULE.ContractError):
        MODULE.validate_config(invalid)


def test_rejects_overlapping_layer_schedule(config):
    invalid = copy.deepcopy(config)
    invalid["text_config"]["linear_attn_config"]["kda_layers"].append(4)
    with pytest.raises(MODULE.ContractError, match="disjoint partition"):
        MODULE.validate_config(invalid)


def test_source_map_schema():
    assert MODULE.validate_source_map(WORKSPACE / "zml" / "docs" / "kimi_k3" / "source-map.yaml") == 11


def test_memory_formulas():
    values = MODULE.memory_estimates()["bytes"]
    assert values["kda_recurrent_state_per_batch_layer"] == 3_145_728
    assert values["mla_cache_per_batch_layer_token"] == 1_152
