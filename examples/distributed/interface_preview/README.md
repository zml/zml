# Distributed ZML interface preview

These files show what distributed ZML applications could look like after the
multi-host implementation in `multihost.md` is complete.

They are intentionally **not buildable**:

- this directory has no `BUILD.bazel` file;
- several APIs do not exist yet;
- the examples are interface sketches, not tests;
- names and signatures may change during implementation.

The working low-level bootstrap remains in `../distributed_info`. These
previews show the intended application-facing layer after that bootstrap is
owned by `zml.Platform`.

## Files

| File | Use case |
| --- | --- |
| `common.zig` | Shared CLI parsing and final `Platform.init` bootstrap. |
| `distributed_info.zig` | Process, global-device, and local-device discovery. |
| `data_parallel_matmul.zig` | Batch sharding over every host and GPU. |
| `manual_all_reduce.zig` | Partition IDs and a named-axis all-reduce. |
| `data_parallel_train_step.zig` | Synchronous data-parallel gradient averaging. |
| `hybrid_parallel_inference.zig` | Host data parallelism plus local tensor parallelism. |
| `distributed_checkpoint.zig` | Atomic rank-sharded checkpoint save and restore. |

## Intended lifecycle

Every process runs the same program:

```text
parse job configuration
        ↓
zml.Platform.init(..., .distributed = ...)
        ↓
register logical mesh and physical-axis bindings
        ↓
compile one global SPMD program
        ↓
create only this process's addressable buffer shards
        ↓
execute in the same order on every process
        ↓
inspect local shards / checkpoint / synchronize
        ↓
Platform.deinit performs coordinated shutdown
```

Tensor payloads move through PJRT/XLA/NCCL. The distributed coordinator is
only the control plane for startup, metadata, barriers, health, and shutdown.

## Future launch shape

After real Bazel targets replace these previews, both hosts would run the same
target with a different process index and the same namespace:

```bash
# Host 0
bazel run //examples/distributed/data_parallel_matmul -- \
    100.80.27.10:8910 0 2 zml-run-001

# Host 1
bazel run //examples/distributed/data_parallel_matmul -- \
    100.80.27.10:8910 1 2 zml-run-001
```

`common.zig`, `distributed_info.zig`, `data_parallel_matmul.zig`, and
`manual_all_reduce.zig` closely follow the core API proposed in
`multihost.md`. Process-local buffer creation, named collectives, training
conveniences, and checkpoint management are later interface candidates.

Runnable counterparts to the five JAX programs in the workspace root are in
`../gpu_example`, `../gpu_mesh`, `../gpu_matmul`,
`../gpu_matmul_replicated`, and `../gpu_shard_map`.
