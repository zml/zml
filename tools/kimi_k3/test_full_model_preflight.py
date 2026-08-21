from __future__ import annotations

import copy
import unittest

import full_model_preflight as preflight


def kimi_config() -> dict:
    mla = [*range(4, 93, 4), 93]
    kda = [layer for layer in range(1, 94) if layer not in mla]
    return {
        "model_type": "kimi_k3",
        "text_config": {
            "num_hidden_layers": 93,
            "hidden_size": 7168,
            "first_k_dense_replace": 1,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "attn_res_block_size": 12,
            "linear_attn_config": {
                "kda_layers": kda,
                "full_attn_layers": mla,
            },
        },
    }


class FullModelPreflightTests(unittest.TestCase):
    def test_config_freezes_complete_one_based_schedule(self):
        kda, mla = preflight.validate_config(kimi_config())
        self.assertEqual(len(kda), 69)
        self.assertEqual(len(mla), 24)
        self.assertEqual(kda | mla, set(range(93)))
        self.assertIn(0, kda)
        self.assertIn(3, mla)
        self.assertIn(92, mla)

    def test_config_rejects_schedule_gap(self):
        config = copy.deepcopy(kimi_config())
        config["text_config"]["linear_attn_config"]["kda_layers"].pop()
        with self.assertRaisesRegex(preflight.PreflightError, "schedule"):
            preflight.validate_config(config)

    def test_dense_and_moe_layer_ownership_contracts(self):
        dense = preflight.expected_nonexpert_suffixes("kda_dense")
        preflight.validate_layer_inventory(0, "kda_dense", dense, {})

        moe = preflight.expected_nonexpert_suffixes("kda_moe")
        expert_counts = {
            component: preflight.EXPERT_COUNT
            for component in preflight.EXPERT_COMPONENTS
        }
        preflight.validate_layer_inventory(1, "kda_moe", moe, expert_counts)

        broken = dict(expert_counts)
        broken["w2.weight_scale"] -= 1
        with self.assertRaisesRegex(preflight.PreflightError, "incomplete expert"):
            preflight.validate_layer_inventory(1, "kda_moe", moe, broken)

    def test_expert_partitions_are_balanced_and_complete(self):
        sizes = preflight.partition_sizes(preflight.EXPERT_COUNT, 3)
        self.assertEqual(sizes, [299, 299, 298])
        self.assertEqual(sum(sizes), preflight.EXPERT_COUNT)

    def test_tensor_parallel_selection_respects_contract_dimensions(self):
        for devices, tensor_parallel in (
            (1, 1),
            (8, 8),
            (16, 16),
            (24, 8),
            (32, 32),
        ):
            with self.subTest(devices=devices):
                self.assertEqual(
                    preflight.choose_tensor_parallel(devices), tensor_parallel
                )
                self.assertEqual(devices % tensor_parallel, 0)
                self.assertTrue(
                    all(
                        dim % tensor_parallel == 0
                        for dim in preflight.TP_DIMS
                    )
                )


if __name__ == "__main__":
    unittest.main()
