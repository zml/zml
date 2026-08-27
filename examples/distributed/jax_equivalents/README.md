# JAX study examples expressed with the proposed ZML interface

These files are one-to-one interface previews based on the JAX programs in the
workspace root. They preserve the important shapes, placements, operations,
and observations from those programs.

They intentionally have no `BUILD.bazel` file and are not expected to compile
until the public distributed ZML implementation exists.

| JAX source | ZML preview | Main concept |
| --- | --- | --- |
| `gpu_example.py` | `gpu_example.zig` | Initialize, inspect devices, barrier. |
| `gpu_mesh.py` | `gpu_mesh.zig` | Two-dimensional sharding and global sum. |
| `gpu_matmul.py` | `gpu_matmul.zig` | Both matrix dimensions sharded. |
| `gpu_matmul_replicat.py` | `gpu_matmul_replicated.zig` | Replicated right-hand matrix. |
| `gpu_shard_map.py` | `gpu_shard_map.zig` | Shard-local work and explicit reduction. |

## Concept mapping

| JAX | Proposed ZML |
| --- | --- |
| `jax.distributed.initialize(...)` | `zml.Platform.init(..., .distributed = ...)` |
| `jax.process_index()` | `platform.processIndex()` |
| `jax.devices()` | `platform.globalDevices()` |
| `jax.local_devices()` | `platform.addressableDevices()` |
| `Mesh(..., ("host", "gpu"))` | logical axes bound to `.network` and `.link` |
| `P("host", "gpu")` | first shape axis `.host`, second `.gpu` |
| `P(("host", "gpu"), None)` | one `.data` axis bound to both physical axes |
| `P()` | `zml.Sharding.replicated` |
| `jax.device_put` | `zml.Buffer.fromSlice` |
| `jax.jit` | `platform.compileFn` |
| `array.addressable_shards` | `buffer.shards()` |
| `jax.shard_map` | `zml.ops.manualComputation` |
| `jax.lax.psum` | `zml.ops.allReduceAxes` |
| `block_until_ready()` | `callOpts(..., .{ .wait = true })` |
| `sync_global_devices(name)` | `platform.barrier(io, name)` |

`gpu_shard_map.py` currently creates an `(8, 16)` input on four devices.
Therefore each device receives `(2, 16)`, or 32 values—not four rows as one
source comment states. The ZML preview follows the actual expression and
expects a partial sum of 32 and a global sum of 128.
