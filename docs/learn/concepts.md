
# ZML Concepts

## Model lifecycle

ZML is an inference stack that helps running Machine Learning (ML) models, and
particulary Neural Networks (NN).

The lifecycle of a model is implemented in the following steps:

1. Open the model file and read the shapes of the weights, but don't load the weights 
   yet.

2. Using the loaded shapes and optional metadata, instantiate a model struct
   with `Tensor`s, representing the shape and layout of each layer of the NN.

3. Compile the model struct and it's `forward` function into an accelerator
   specific executable. The `forward` function describes the mathematical
   operations corresponding to the model inference.

4. Load the model weights from disk, onto the accelerator memory.

5. Load some user inputs, and copy them to the accelerator.

6. Call the executable using the weights and the user inputs.

7. Fetch the returned model output from accelerator into host memory, and
   finally present it to the user.

8. When all user inputs have been processed, free the executable resources, weights and
   inputs.


**Some details:**

Note that the compilation and weight loading steps are both bottlenecks to your
model startup time, but they can be done in parallel thanks to Zig's `std.Io` interface.

The accelerator is typically a GPU, but can be another chip, or even the CPU
itself, churning vector instructions.


## Tensor Bros.

In ZML, we leverage Zig's static type system to differentiate between a few
concepts, hence we not only have a `Tensor` to work with, like other ML
frameworks, but also `Buffer`, `Slice`, and `Shape`.

Let's explain all that.

* `Shape`: _describes_ a multi-dimension array.
    - `Shape.init(.{16}, .f32)` represents a vector of 16 floats of 32 bits
      precision.
    - `Shape.init(.{512, 1024}, .f16)` represents a matrix of `512*1024` floats
      of 16 bits precision, i.e. a `[512][1024]f16` array.

    A `Shape` is only **metadata**, it doesn't point to or own any memory. The
    `Shape` struct can also represent a regular number, aka a scalar:
    `Shape.init(.{}, .i32)` represents a 32-bit signed integer.

* `Slice`: _is_ the combination of a `Shape` and raw bytes (bytes that are **on the CPU**).
  - can own the underlying memory - but can also accomodate non-owned memory.

* `Buffer`: _is_ a multi-dimension array, whose memory is allocated **on an
  accelerator**.
    - contains a handle that the ZML runtime can use to convert it into a
      physical address, but there is no guarantee this address is visible from
      the CPU.
    - can be created from a `Slice` by calling `Buffer.fromSlice(...)`. 

* `Tensor`: is a mathematical object representing an intermediary result of a
  computation or an input to an executable (including weights).
  - is basically a `Shape` with an attached MLIR value when in the context of a compilation.


## The model struct

The model struct is the Zig code that describes your Neural Network (NN).
Let's look a the following model architecture:

![Multilayer perceptrons](https://raw.githubusercontent.com/zml/zml.github.io/refs/heads/main/docs-assets/perceptron.png)

This is how we can describe it in a Zig struct:

```zig
const Model = struct {
    input_layer: zml.Tensor,
    output_layer: zml.Tensor,

    pub fn forward(self: Model, input: zml.Tensor) zml.Tensor {
      const hidden = self.input_layer.matmul(input);
      const output = self.output_layer.matmul(hidden);
      return output;
    }
}
```

NNs are generally seen as a composition of smaller NNs, which are split into
layers. ZML makes it easy to mirror this structure in your code.

```zig
const Model = struct {
    input_layer: MyOtherLayer,
    output_layer: MyLastLayer,

    pub fn forward(self: Model, input: zml.Tensor) zml.Tensor {
      const hidden = self.input_layer.forward(input);
      const output = self.output_layer.forward(hidden);
      return output;
    }
}
```

`zml.nn` module provides a number of well-known layers to more easily bootstrap
models.

Since the `Model` struct contains `Tensor`s, it is only ever useful during the
compilation stage, but not during inference. If we want to represent the model
with actual `Buffer`s, we can use the `zml.Bufferize(Model)`, which is a mirror
struct of `Model` but with a `Buffer` replacing every `Tensor`.

## Strong type checking

Let's look at the model life cycle again, but this time annotated with the
corresponding types.

1. Open the model file and read the shapes of the weights -> `zml.Shape`

2. Instantiate a model struct -> `Model` struct (with `zml.Tensor` inside)

3. Compile the model struct and its `forward` function into an executable.
   `foward` is a `Tensor -> Tensor` function, executable is a
   `zml.Executable`

4. Load the model weights from disk, onto accelerator memory ->
   `zml.Bufferized(Model)` struct (with `zml.Buffer` inside)

5. Load some user inputs (custom struct), encode them into arrays of numbers
   (`zml.Slice`), and copy them to the accelerator (`zml.Buffer`).

6. Call the executable on the user inputs. `exe.call` accepts `zml.Buffer`
   arguments and returns `zml.Buffer`

7. Return the model output (`zml.Buffer`) to the host (`zml.Slice`),
   decode it (custom struct) and finally return to the user.

## Manual computations and sharding

Ordinary graph results leave their sharding open for Shardy to infer. A manual
computation declares the layouts its local kernel requires, independently of
annotations on earlier tensors:

```zig
const result = zml.ops.manualComputation(Kernel.body, .{
    .q = projected_q,
    .cache = updated_cache,
    .lengths = lengths,
}, output_shape, .{
    .mesh = mesh,
    .in_specs = .{
        .q = .{ .hkv = .model },
        .cache = .{ .hkv = .model },
        .lengths = .replicated,
    },
    .out_specs = .{ .hkv = .model },
});
```

The body receives local input tensors and local output shapes. Each tensor needs
an explicit spec; omitted dimensions within a spec are replicated. Open
specs, unknown axes, and dimensions that cannot be divided into equal shards
are rejected. Specs follow nested tensor fields, including optional values;
`.replicated` can explicitly replicate a whole subtree.

`out_specs` describes how local results form the global result. Returned tensors
retain this explicit sharding, including when returned directly from the
executable. Subsequent ordinary graph operations leave their results open for
Shardy to infer; an unspecified executable output defaults to replication.

Attention accepts a `zml.attention.Sharding` with a mesh and separate `heads`
and `kv_heads` logical axes. MoE takes `Options.sharding` with a mesh and an
`experts` logical axis. An axis value of `null` explicitly requests replication.
These choices reach the backend without reading intermediate tensor sharding.
Mesh construction maps the logical axes onto the physical topology; the two
meshes can have different ranks.
