# Simplifying Dimension Handling with Tagged Tensors

In most frameworks, axes are anonymous indices: `x.transpose(1, 2)` or
`matmul(a, b)` and you hope the ranks line up. ZML lets you **name** axes with
tags so ops, reshapes, and sharding refer to meaning instead of positions.

## Shapes with tags

Pass enum literals (or other tag sources) when building a `Shape`:

```zig
const s = zml.Shape.init(.{ .batch = 32, .seq = 128, .d = 4096 }, .f16);
```

Or attach tags to an existing tensor:

```zig
const x = input.flatten().convert(.f32).withTags(.{.d});
```

Query by name:

```zig
const seq_len = x.dim(.seq);
const axis = x.axis(.d);
```

## Ops that use tags

Contract / reduce on a named axis instead of a raw index:

```zig
// From examples/mnist — linear layer
return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
```

Tags also show up when partitioning for multi-device:

```zig
shape.withPartitioning(.{ .batch = .data, .hidden = .model });
```

Logical mesh axes (`.data`, `.model`, …) are registered on the `Platform` /
`Sharding` mesh; see `examples/sharding` and `zml/Sharding.zig`.

## Why bother

- Fewer off-by-one axis bugs when ranks grow (attention, MoE, KV cache).
- Broadcast and contract rules can check matching tags.
- Sharding binds **logical** tags to **mesh** axes without hard-coding dim indices.

## Related

- [ZML Concepts](../learn/concepts.md) — `Shape` / `Slice` / `Buffer` / `Tensor`
- [Writing your first model](./write_first_model.md)
- MNIST example: `examples/mnist/mnist.zig`
