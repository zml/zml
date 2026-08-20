from __future__ import annotations

import torch

from export_prefix4_reference import diagnostic_head


def test_diagnostic_head_selects_sources_and_greedy_token():
    hidden = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    blocks = torch.tensor([
        [[1.0, 0.0]],
        [[0.0, 1.0]],
    ])
    weights = {
        "output_attn_res_norm": torch.ones(2),
        "output_attn_res_proj": torch.tensor([[1.0, 0.0]]),
        "final_norm": torch.ones(2),
        "lm_head": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ]),
    }
    result = diagnostic_head(hidden, blocks, weights)
    assert result["output_attn_res.candidates"].shape == (2, 2, 2)
    assert result["output_attn_res.weights"].shape == (2, 2)
    assert torch.allclose(result["output_attn_res.weights"].sum(-1), torch.ones(2))
    assert result["logits"].shape == (1, 2, 3)
    assert result["greedy_token"].tolist() == [1]


def test_diagnostic_head_preserves_hidden_shape():
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    blocks = torch.zeros(3, 1, 8)
    weights = {
        "output_attn_res_norm": torch.ones(8),
        "output_attn_res_proj": torch.ones(1, 8),
        "final_norm": torch.ones(8),
        "lm_head": torch.eye(8),
    }
    result = diagnostic_head(hidden, blocks, weights)
    assert result["output_attn_res.out"].shape == hidden.shape
    assert result["final_norm.out"].shape == hidden.shape
    assert result["logits"].shape == (1, 3, 8)
