# Reading partitioned HLO graphs

For `*.after_spmd_partitioner.hlo.pb`, hlo2json automatically reads the sibling
`*.before_optimizations.hlo.pb` when present. The normal CLI and Python adapter
both use this behavior. Keep the two files together.

Select a node to inspect its attributes and output metadata:

- `tensor_shape` is the local shape from the selected dump.
- `pre-SPMD shape` comes from the sibling's entry computation. Parameters match
  by parameter number. Intermediate values match a unique instruction name using
  `metadata.op_name`; this is operation provenance, not proof that the values are
  identical. Generated intermediate values may refer to the same original op.
- `pre-SPMD sharding` describes that reference value, not an inferred sharding of
  the local tensor. Tile dimensions include trailing subgroup dimensions; the
  replication factor counts explicitly replicated subgroup dimensions.
- `recorded sharding` describes the selected instruction. Missing sharding is
  unknown, not replicated. Shape equality does not imply replication: the inputs
  of all-reduces may hold partial sums.
- Payload sizes count logical tensor bytes, excluding layout padding and protocol
  overhead. Unsupported element types and dynamic shapes have no byte estimate.
- Ring send/receive estimates assume an ideal ring within each recorded group.
  They are per device, per invocation, and exclude protocol overhead. The dump
  does not establish the runtime algorithm or measured network traffic.

Collectives expose channels, dimensions, device-group descriptions and reduction
root opcodes. Copies retain the `copy` label even if their historical name starts
with `custom-call`. Real custom calls show their target.

## Inspected Llama layer

`module_2797.llama_layer.after_spmd_partitioner.hlo.pb` has two partitions, with
collective groups containing devices 0 and 1:

| Operations | Local input | Local output | Input payload/device | Ideal ring send/device |
| --- | --- | --- | --- | --- |
| Two all-gathers (channels 1, 2) | BF16[1,16,4,128] | BF16[1,16,8,128] | 16 KiB each | 16 KiB each |
| Two sum all-reduces (channels 3, 4) | BF16[16,4096] | BF16[16,4096] | 128 KiB each | 128 KiB each |

The ideal-ring total is 288 KiB sent and 288 KiB received per device per layer
invocation. This is an analytical estimate, not measured wire traffic.

The sibling before-optimization HLO retains global parameter shapes and explicit
shardings. For example, parameter 1 changes from BF16[6144,4096] to
BF16[3072,4096], with tile dimensions [2,1] and no replication. Most intermediate
shardings are absent after partitioning. The sibling `module.mlir` additionally
retains named meshes, but the converter does not parse MLIR or infer physical
network topology from those names.

The two remaining actual custom calls target `__gpu$xla.gpu.triton` (attention
kernels). Many other `custom-call.*` names are copies descended from sharding
constraints, not communication operations.

## Compiler implementation context versus runtime decisions

The converter also automatically discovers, for the same dump stem:

- `<stem>/topology.pb`: PJRT GPU grouping, GPU model and target PCIe capability.
  This is not the physical NCCL interconnect graph. In particular, PJRT topology
  partitions are distinct from the module's SPMD partition count.
- `<stem>.*gpu_after_optimizations.hlo.pb`: a unique final GPU dump. Collectives
  are correlated by opcode and nonzero channel ID. Multiple matches or multiple
  final dumps are left ambiguous. The actual backend-config JSON is displayed;
  `DEFAULT` is not interpreted as a specific NCCL algorithm.
- `<stem>.thunk_sequence_after_thunk_passes.txt`: module-wide collective thunk
  lines, without pretending to associate an unlabelled thunk with a channel.

Absent optional files are skipped; malformed optional files produce diagnostics.
Arbitrary HLO filenames without a recognized dump-stage suffix are not searched.

For the inspected layer, the final HLO has only two all-reduces (channels 3 and 4),
with synchronous, non-pipelined collective backend configs using `DEFAULT`.
Neither gather has a same-opcode/channel counterpart in that final dump, and the
thunk sequence also contains only two all-reduces. Thus the earlier 288 KiB total
is a model of the post-SPMD stage, **not** the final executable. An ideal-ring
model of just those two final BF16 all-reduces would send 256 KiB/device.

NCCL selects algorithms (e.g. Ring/Tree) and protocols (e.g. Simple/LL/LL128) at
runtime, subject to environment/configuration constraints and tuning. These
choices are not stored in the supplied compiler dumps. Runtime evidence must be
captured on the machine executing the model, for example with:

```sh
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL,NET \
NCCL_DEBUG_FILE='/tmp/nccl.%h.%p.log' \
NCCL_TOPO_DUMP_FILE=/tmp/nccl-topology.xml \
  <your-model-command>
```

Use a distinct topology filename for each process when launching multiple
processes. Tuning diagnostics can report tables/candidates rather than an
unambiguous choice for each operation; a runtime/profiler trace may be needed to
identify a particular call's selected kernel. XLA channel IDs are not NCCL
channel IDs, so runtime records cannot safely be joined on that number alone.
The viewer does not automatically attribute arbitrary runtime logs to graph
nodes. No such logs or NCCL topology XML were supplied with this dump family.

References: [NCCL logging](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/logging.html)
and [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
