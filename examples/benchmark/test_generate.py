"""Independent examples for the benchmark generator's offset oracle.

Run: python3 -m unittest discover -s examples/benchmark -p 'test_*.py'
"""

import unittest

from generate import (
    LAYOUTS,
    MODEL,
    TWO,
    Strategy,
    bindings_for,
    expected_offsets,
    shape_partitions,
)


class GeneratorTests(unittest.TestCase):
    def test_reordered_asymmetric_matrix(self):
        # [16, 24] f32: row stride 96, half-row width 48 bytes.
        bindings = {'model': ('link_y',), 'data': ('link_x',)}
        self.assertEqual(
            expected_offsets(LAYOUTS[2], bindings, (16, 24), ('model', 'data')),
            [0, 768, 48, 816],
        )

    def test_last_axis_partition(self):
        self.assertEqual(
            expected_offsets(LAYOUTS[1], {'model': ('link_x',)}, (3, 24), (None, 'model')),
            [0, 12, 24, 36, 48, 60, 72, 84],
        )

    def test_shared_physical_axis_is_consumed_in_tensor_order(self):
        bindings = {'model': ('link_x',), 'data': ('link_x',)}
        self.assertEqual(
            expected_offsets(LAYOUTS[1], bindings, (16, 24), ('data', 'model')),
            [0, 192, 384, 576, 768, 960, 1152, 1344],
        )

    def test_incompatible_shapes_and_meshes(self):
        all_axes = bindings_for(LAYOUTS[-1], Strategy('all', mode='all'))
        self.assertIsNone(expected_offsets(LAYOUTS[-1], all_axes, (16,), ('model',)))
        missing = Strategy('missing', (('model', ('link_y',)),))
        self.assertIsNone(bindings_for(LAYOUTS[1], missing))

    def test_folded_axes_and_reversed_device_ids(self):
        layout = LAYOUTS[-1]
        bindings = bindings_for(layout, Strategy('fold', mode='fold'))
        self.assertEqual(
            expected_offsets(layout, bindings, (64,), ('model',)),
            list(range(0, 256, 8)),
        )

    def test_shape_partition_variety(self):
        cases = list(shape_partitions(TWO))
        self.assertEqual(len(cases), len(set(cases)))
        self.assertIn(((24, 56), ('model', 'data')), cases)
        self.assertIn(((24, 56), ('data', 'model')), cases)
        self.assertIn(((16, 24, 40), (None, None, 'model')), cases)
        self.assertIn(((64,), (None,)), cases)
        self.assertIn(((64,), ('model',)), list(shape_partitions(MODEL)))


if __name__ == '__main__':
    unittest.main()
