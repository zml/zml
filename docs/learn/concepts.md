
# ZML Concepts

## Model lifecycle

ZML is an inference stack that helps running Machine Learning (ML) models, and
particulary Neural Networks (NN).

The lifecycle of a model is implemented in the following steps:

1. Open the model file and read the shapes of the weights, but leave the
   weights on the disk.

2. Using the loaded shapes and optional metadata, instantiate a model struct
   with `Tensor`s, representing the shape and layout of each layer of the NN.

3. Compile the model struct and it's `forward` function into an accelerator
   specific executable. The `forward` function describes the mathematical
   operations corresponding to the model inference.

4. Load the model weights from disk, onto the accelerator memory.

5. Bind the model weights to the executable.

6. Load some user inputs, and copy them to the accelerator.

7. Call the executable on the user inputs.

8. Fetch the returned model output from accelerator into host memory, and
   finally present it to the user.

9. When all user inputs have been processed, free the executable resources and
   the associated weights.


**Some details:**

Note that the compilation and weight loading steps are both bottlenecks to your
model startup time, but they can be done in parallel. **ZML provides
asynchronous primitives** to make that easy.

The **compilation can be cached** across runs, and if you're always using the
same model architecture with the same shapes, it's possible to by-pass it
entirely.

The accelerator is typically a GPU, but can be another chip, or even the CPU
itself, churning vector instructions.


## Tensor Bros.

In ZML, we leverage Zig's static type system to differentiate between a few
concepts, hence we not only have a `Tensor` to work with, like other ML
frameworks, but also `Buffer`, `HostBuffer`, and `Shape`.

Let's explain all that.

* `Shape`: _describes_ a multi-dimension array.
    - `Shape.init(.{16}, .f32)` represents a vector of 16 floats of 32 bits
      precision.
    - `Shape.init(.{512, 1024}, .f16)` represents a matrix of `512*1024` floats
      of 16 bits precision, i.e. a `[512][1024]f16` array.

    A `Shape` is only **metadata**, it doesn't point to or own any memory. The
    `Shape` struct can also represent a regular number, aka a scalar:
    `Shape.init(.{}, .i32)` represents a 32-bit signed integer.

* `HostBuffer`: _is_ a multi-dimensional array, whose memory is allocated **on
  the CPU**.
  - points to the slice of memory containing the array 
  - typically owns the underlying memory - but has a flag to remember when it
    doesn't.

* `Buffer`: _is_ a multi-dimension array, whose memory is allocated **on an
  accelerator**.
    - contains a handle that the ZML runtime can use to convert it into a
      physical address, but there is no guarantee this address is visible from
      the CPU.
    - can be created by loading weights from disk directly to the device via
      `zml.aio.loadBuffers` 
    - can be created by calling `HostBuffer.toDevice(accelerator)`.

* `Tensor`: is a mathematical object representing an intermediary result of a
  computation.
  - is basically a `Shape` with an attached MLIR value representing the
    mathematical operation that produced this `Tensor`.


## The model struct

The model struct is the Zig code that describes your Neural Network (NN).
Let's look a the following model architecture:

![Multilayer perceptrons](https://zml.ai/docs-assets/perceptron.png)

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

1. Open the model file and read the shapes of the weights -> `zml.HostBuffer`
   (using memory mapping, no actual copies happen yet)

2. Instantiate a model struct -> `Model` struct (with `zml.Tensor` inside)

3. Compile the model struct and its `forward` function into an executable.
   `foward` is a `Tensor -> Tensor` function, executable is a
   `zml.Exe(Model.forward)`

4. Load the model weights from disk, onto accelerator memory ->
   `zml.Bufferized(Model)` struct (with `zml.Buffer` inside)

5. Bind the model weights to the executable `zml.ExeWithWeight(Model.forward)`

6. Load some user inputs (custom struct), encode them into arrays of numbers
   (`zml.HostBuffer`), and copy them to the accelerator (`zml.Buffer`).

7. Call the executable on the user inputs. `module.call` accepts `zml.Buffer`
   arguments and returns `zml.Buffer`

8. Return the model output (`zml.Buffer`) to the host (`zml.HostBuffer`),
   decode it (custom struct) and finally return to the user.
