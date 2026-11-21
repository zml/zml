# Tenstorrent cluster descriptors

tt-umd cluster descriptors (YAML) for the platform's device-free mode
(`CreateOptions.tenstorrent = .{ .runtime = .fake, .cluster_descriptor = ... }`):
the plugin builds its topology from the file instead of probing cards, so a
machine without cards can compile ahead of time and run on the host.

Copied unedited from xla-private
`xla/backends/tenstorrent/topology/fixtures/`, where their provenance is
recorded:

| File | Devices | Default view | Origin |
|---|---|---|---|
| `blackhole_P150.yaml` | 1 | `[1]` | tt-umd `descriptors/cluster_descriptor_examples/` (Apache-2.0, Tenstorrent Inc.) |
| `box_2xp150_capture.yaml` | 2 | `[2]` line, 2 links | captured on the `tt` box (two p150a, one QSFP-DD cable) |
| `3xp150_line.yaml` | 3 | `[3]` line | synthetic, written for the xla-private topology tests |
| `blackhole_4xP150.yaml` | 4 | `[4]` ring | tt-umd `descriptors/cluster_descriptor_examples/` |
| `blackhole_8xP150.yaml` | 8 | `[8]` ring | tt-umd `descriptors/cluster_descriptor_examples/` |
