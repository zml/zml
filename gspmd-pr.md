# Why the isolated GSPMD correctness fix is required

## PR summary

This PR fixes two independent correctness assumptions in ZML's existing GSPMD
path:

1. An XLA tile assignment was serialized in canonical device order instead of
   the order defined by the tensor's placement.
2. `manualComputation` changed from global shapes to local shard shapes inline
   in the global function, without the complete manual-sharding boundary used
   by JAX/XLA.

These are core GSPMD issues. Distributed execution exposed them because a
host/GPU mesh has multiple physical axes, but neither issue is caused by the
distributed runtime. Both can be reproduced without adding distributed
platform, buffer, or execution APIs.


The changes do not alter ZML's public API or its Shardy path.

## The two different views of a tensor

GSPMD must keep three related concepts consistent:

- **Global shape:** the logical tensor seen by the model, such as `[16, 64]`.
- **Local shape:** the part seen by one partition, such as `[8, 64]`.
- **Placement:** the mapping from each logical tile and replica to an XLA
  partition.

Knowing only that a tensor has four shards is insufficient. XLA must also know
which partition owns each tile and which partitions are replicas of the same
tile. A manual computation must additionally say exactly where the global view
ends and the local view begins.

The previous implementation handled simple layouts where these relationships
could be inferred. It did not encode them correctly for every valid placement.

## Fix 1: derive tile-assignment order from `Placement`

### What the old code did

`gspmdShardingAttrForShape` correctly calculated the tile dimensions, but then
emitted the assignment IDs in canonical physical-device order:

```text
0,1,2,3
```

Canonical order describes how the physical mesh is traversed. It does not
necessarily describe how a particular tensor uses that mesh. Tensor dimensions
may bind to physical axes in another order, and unused physical axes become
replica axes.

### Concrete 2×2 example

Consider a physical mesh whose canonical order is host-major:

| Partition | Host/data coordinate | GPU/model coordinate |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |

Now place a tensor as follows:

- its hidden dimension is sharded over the GPU/model axis;
- the host/data axis replicates each hidden tile.

The resulting XLA tile shape is:

```text
[1, 2, 2]
```

This means:

```text
unsharded feature × two hidden tiles × two replicas
```

XLA flattens the assignment in tile-major order. The partitions must therefore
be grouped by hidden tile:

| Hidden tile | Replica partitions |
| ---: | --- |
| 0 | `0,2` |
| 1 | `1,3` |

The correct serialized assignment is:

```text
0,2,1,3
```

The old assignment, `0,1,2,3`, told XLA that partitions 0 and 1 were replicas
of hidden tile 0. They are not: partition 1 has the other GPU/model coordinate.
Likewise, it incorrectly grouped partitions 2 and 3 as replicas of hidden tile
1.

This is not a cosmetic ordering difference. The assignment is part of the
tensor's semantics.

### What the fix does

The fix keeps the existing tile-shape calculation. It changes only how the
assignment list is constructed:

1. Build the tensor's `Placement`.
2. For each physical device, use the placement axis plans and device
   coordinates to calculate its logical tile index.
3. Place the XLA partition ordinal into that tile's replica slots.
4. Serialize the resulting tile-major permutation.

The implementation still uses one temporary assignment buffer, just as the old
implementation did. It does not add another device-sized allocation.

The focused regression verifies the previously missing case:

```text
{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}
```

### What happens without this fix

For a non-trivial physical mesh, XLA can receive a valid-looking but incorrect
`mhlo.sharding` attribute. Possible consequences include:

- a partition consuming the wrong input tile;
- two different tiles being described as replicas;
- duplicated or missing rows, columns, or attention heads;
- communication being inserted for the wrong partition groups;
- a numerically incorrect result without a compiler or runtime error.

The silent case is the most dangerous. The attribute is syntactically valid,
so there is no requirement for XLA to reject it.

## Fix 2: make `manualComputation` a real GSPMD boundary

### What `manualComputation` means

Outside `manualComputation`, ZML operations describe global tensors and GSPMD
decides how to partition them. Inside `manualComputation`, the body deliberately
works on the local shard already assigned to one partition.

For example:

```text
global value: tensor<16x64xf32>
local value:  tensor<8x64xf32>
```

The compiler must not treat operations on the local value as another global
computation that still needs automatic partitioning.

### What the old lowering emitted

The previous lowering put the local-shape body directly in the public global
function:

```text
global input
  -> SPMDFullToShardShape
  -> inline local-shape operations
  -> SPMDShardToFullShape
  -> global output
```

The shape-changing custom calls were present, but two pieces of the protocol
were missing:

- an explicit global sharding constraint before entering manual mode and a
  manual constraint before leaving it;
- a private local-shape function that structurally isolates the manual body
  from the automatically partitioned global function.

The body therefore used local tensor types while remaining inside a function
whose surrounding semantics were global and automatically partitioned.

### What the corrected lowering emits

The corrected lowering follows the same boundary structure used by JAX/XLA:

```text
global input
  -> Sharding(global placement)
  -> SPMDFullToShardShape {manual}
  -> call private @manual_computation.impl_N(local types)
  -> Sharding {manual}
  -> SPMDShardToFullShape(global placement)
  -> global output
```

The private function has an explicitly local signature:

```text
func.func private @manual_computation.impl_N(
    tensor<8x64xf32>
) -> tensor<8x64xf32>
```

Each element has one responsibility:

- `Sharding(global placement)` fixes the layout on the global side.
- `SPMDFullToShardShape {manual}` converts to the per-partition shape.
- The private function contains only local-shape operations.
- `Sharding {manual}` says the returned value is already managed manually.
- `SPMDShardToFullShape(global placement)` restores the global view and the
  expected output placement.

This prevents the automatic partitioner from having to infer whether the body
is global or local. It also makes the generated program compatible with the
manual-sharding convention exercised by JAX.

### Allocation impact

The minimized implementation keeps the arrays already present in the old
GSPMD lowering:

- local input and output shapes;
- local input values;
- output sharding attributes;
- global result values and types;
- returned tensors.

The existing global result arrays are moved earlier and reused for the private
function return and call. Block arguments are added directly, and the unique
private-function name is formatted into a stack buffer. The private-function
scaffolding therefore adds no explicit arena-backed slice allocation. Building
the required input sharding attribute still uses the existing attribute
builder and its scratch storage.

### What happens without this fix

The inline form can continue to work for a simple elementwise body, but it
relies on the partitioner inferring intent from surrounding shapes and custom
calls. For more demanding bodies or another XLA version, possible outcomes are:

- local operations being interpreted or rewritten in the global context;
- a local dimension being partitioned again;
- incompatible local/global shape propagation;
- per-shard custom calls receiving an unexpected shape or placement;
- compiler verification or partitioning failures;
- output resharding that does not match the declared global result;
- behavior changing after an apparently unrelated optimization or XLA update.

The old program compiling successfully was not proof that the manual boundary
was fully specified. It showed only that the simple case was inferable.

## Why did it work before?

Several properties of the previous examples masked the defects.

### One physical axis

On a one-dimensional two-GPU mesh, canonical device order and tensor tile order
are normally identical:

```text
canonical: 0,1
placement: 0,1
```

There is no second physical axis that can become a replica axis or be bound in
a different order. The tile-ordering bug has nothing visible to permute.

### Aligned axis order

Even on a multi-axis mesh, the old assignment is correct when tensor axes are
used in the same order as the physical traversal. The problem appears only when
placement order and canonical order differ, such as model sharding on the inner
GPU axis with replication on the outer host axis.

### Dense device IDs

Single-host PJRT devices are normally numbered `0,1,...`. In those cases a PJRT
device ID looks identical to an XLA partition ordinal, hiding another unsafe
assumption in the old serialization.

### Simple shard-local operations

The studied manual body used shape-preserving elementwise operations. Those
operations need no resharding and no explicit collective, so the inline local
region was easy for GSPMD to accept. A debug print or simple multiply can appear
correct even though the global/local protocol is incomplete.

### Shardy uses another lowering

ZML's Shardy path emits an `sdy.manual_computation` region. It does not use this
GSPMD custom-call sequence. Tests or examples using the default Shardy
partitioner therefore do not exercise the faulty GSPMD branch.

### Symmetric data can hide a wrong assignment

If replicated inputs contain identical values, swapping replica membership may
produce the same visible numbers. Reductions and symmetric operations can mask
the error further. A placement error can therefore survive smoke tests until
different shards contain meaningfully different data.

In short, the old code worked when physical order, logical order, IDs, shapes,
and compiler inference all happened to agree. A two-axis host/GPU topology
removed those coincidences.

## Why this belongs in an isolated PR

The distributed work is a consumer of these GSPMD semantics, not their
implementation source. This PR changes only:

- `zml/Sharding.zig`: placement-derived tile assignment and its focused test;
- `zml/ops.zig`: complete GSPMD manual-computation boundaries.

It deliberately does not change:

- `Platform` or PJRT client creation;
- distributed coordination or lifecycle;
- buffers or execution;
- collectives;
- examples or launch scripts;
- the public ZML API;
- the Shardy lowering.

Keeping this separate allows reviewers to evaluate two precise invariants:

1. Does every emitted tile assignment agree with `Placement`?
2. Does every GSPMD manual body have explicit global-to-local and
   local-to-global boundaries?

The distributed PR can then rely on those invariants instead of carrying a
second copy of the same core correctness fix.

## Verification

The isolated branch has been verified remotely on CUDA Linux:

- Zig formatting passed.
- `bazel test //zml:test` passed with CUDA enabled.
- `bazel build //examples/sharding` passed.
- The sharding example completed on two GPUs with `--partitioner=gspmd`.
- Generated MLIR contained private `manual_computation.impl_N` functions.
- Generated MLIR contained `SPMDFullToShardShape`, manual `Sharding`, and
  `SPMDShardToFullShape` boundaries.
- Both local shards executed and the final output preview was produced.

## Review conclusion

Without these changes, simple GSPMD examples may continue to pass, but ZML
cannot reliably claim that a logical placement is the placement XLA executes,
or that a `manualComputation` body is unambiguously per-partition.

The fix replaces those coincidental assumptions with explicit invariants while
leaving public APIs and distributed runtime code unchanged.
