
# Writing your first model

**In this short guide, we will do the following:**

- clone ZML to work directly within the prepared example folder
- add Zig code to implement our model
- add some Bazel to integrate our code with ZML
- no weights files or anything external is required for this example

The reason we're doing our exercise in the `examples` folder is because it's
especially prepared for new ZML projects. It contains everything needed for ZML
development. From `bazel` configs to `vscode` settings, and `neovim` LSP
support. The `examples` folder serves as a cookiecutter ZML project example,
with just a few example models added already.

**Note:** _The `examples` folder is self-contained. You **can** make a copy of
it to a location outside of the ZML repository. Simply remove all examples you
don't need and use it as a template for your own projects._

So, let's get started, shall we?



**If you haven't done so already, please [install bazel](../tutorials/getting_started.md)**.



Check out the ZML repository. In the `examples` directory, create a new folder
for your project. Let's call it `simple_layer`.

```
git clone https://github.com/zml/zml.git
cd zml/examples
mkdir -p simple_layer
```

... and add a file `main.zig` to it, along with a bazel build file:

```
touch simple_layer/main.zig
touch simple_layer/BUILD.bazel
```

By the way, you can access the complete source code of this walkthrough here:

- [main.zig](https://github.com/zml/zml/tree/master/examples/simple_layer/main.zig)
- [BUILD.bazel](https://github.com/zml/zml/tree/master/examples/simple_layer/BUILD.bazel)



## The high-level Overview

Before firing up our editor, let's quickly talk about a few basic ZML
fundamentals.

In ZML, we describe a _Module_, which represents our AI model, as a Zig
`struct`. That struct can contain Tensor fields that are used for computation,
e.g. weights and biases. In the _forward_ function of a Module, we describe the
computation by calling tensor operations like _mul_, _add_, _dotGeneral_,
_conv2D_, etc., or even nested Modules.

ZML creates an MLIR representation of the computation when we compile the
Module. For compilation, only the _Shapes_ of all tensors must be known. No
actual tensor data is needed at this step. This is important for large models:
we can compile them while the actual weight data is being fetched from disk.

To accomplish this, most ZML code uses a _BufferStore_.
The _BufferStore_ knows how to only load shapes and when to load actual tensor data.
In our example, we wont' use _BufferStore_ at all,
and manually fill out the shapes first then the weights.

In our example, we then copy the result from the computation device to CPU
memory and print it.

**So the steps for us are:**

- describe the computation as ZML _Module_, using tensor operations
- describe the shapes of our model
- compile the _Module_ **asynchronously**
- send the weights of the model to the computation device
- bind the model weights to the _Module_ producing an _executable_ module
- prepare input tensor and call the _executable_ module.
- get the result back to CPU memory and print it

If you like to read more about the underlying concepts of the above, please see
[ZML Concepts](../learn/concepts.md).


## The code

Let's start by writing some Zig code, importing ZML and often-used modules:

```zig
const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
```

You will use above lines probably in all ZML projects. Also, note that **ZML is
async** and comes with its own async runtime, thanks to
[zigcoro](https://github.com/rsepassi/zigcoro).


### Defining our Model

We will start with a very simple "Model".
One that resembles a "multiply and add" operation.

```zig
/// Model definition
const Layer = struct {
    bias: ?zml.Tensor = null,
    weight: zml.Tensor,

    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        var y = self.weight.mul(x);
        if (self.bias) |bias| {
            y = y.add(bias);
        }
        return y;
    }
};
```

You see, in ZML AI models are just structs with a forward function!

There are more things to observe:

- forward functions typically take Tensors as inputs, and return Tensors.
    - more advanced use-cases are passing in / returning structs or tuples, like
      `struct { Tensor, Tensor }` as an example for a tuple of two tensors.
      You can see such use-cases, for example in the
      [Llama Model](https://github.com/zml/zml/tree/master/examples/llama)
- in the model, tensors may be optional. As is the case with `bias`.



### Adding a main() function

ZML code is async. Hence, We need to provide an async main function. It works
like this:

```zig
pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try asynk.AsyncThread.main(gpa.allocator(), asyncMain);
}


pub fn asyncMain() !void {
    // ...
```

The above `main()` function only creates an allocator and an async main thread
that executes our `asyncMain()` function by calling it with no (`.{}`)
arguments.

So, let's start with the async main function:

```zig
pub fn asyncMain() !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    // Start ZML, detect available devices and chose a platform.
    var context = try zml.Context.init();
    defer context.deinit();
    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    ...
}
```

This is boilerplate code that provides a debug allocator,
initialize the ZML context `context`, discover available device
and chose a `platform` to use.
A `platform` is an homogenous set of devices that can work together,
eg 2 Nivdia 4090 GPUs.


### Compilation

We already described the model logic, but we didn't specify any sizes.
To compile an executable specialized to our usecase we are going to provide
the shapes of the individual tensors in the model as well as input shapes.

To do that we need to instantiate the `Layer` struct.
Note that `zml.Shape` always takes the data type associated with the tensor. 
In our example, we will use half-precision floats `f16`, expressed as the enum value `.f16`.

We also set give a `.buffer_id` to each `Tensor` to let the compiler know those are different objects despite the same shape.

```zig
    const model_shapes: Layer = .{
        .bias = zml.Tensor{
            ._shape = zml.Shape.init(.{4}, .f16).withSharding(.{-1}),
            ._id = .{ .buffer_id = 0 },
        },
        .weight = zml.Tensor{
            ._shape = zml.Shape.init(.{4}, .f16).withSharding(.{-1}),
            ._id = .{ .buffer_id = 1 },
        },
    };

    const input_shape = zml.Shape.init(.{4}, .f16);
```

Note: for real models the shape are generally loaded from a config file or directly from the
metadata of the weight file using `zml.aio.BufferStore`.

Then we can compile the model asynchronously so we can do other things in parallel,
typically loading the weights on the device.

```zig
var compilation = try asynk.asyncc(
    zml.compileModel,
    .{ gpa, Layer.forward, model_shapes, .{input_shape}, platform },
);
```

Compiling is happening in the background via the `asyncc` function. We call
`asyncc` with the `zml.compileModel` function and its arguments
separately.
The arguments themselves are:

- an allocator to allocate memory needed for the returned executable.
- `Layer.forward`: the function to be compiled
- the shapes of the model
- the shape of the input tensors (`Layer.forward` only got one argument, so it's a 1-tuple)
- the platform for which to compile

### Loading the weights

Now the model is compiling, we can load the model data to the GPU.
To represent the loaded weights, we are going to instantiate a `zml.Bufferized(Layer)` struct.
This is a struct that looks like our `Layer` struct,
but where all `Tensor`s have been replaced by the `zml.Buffer` type,
and where all non-`Tensor` fields have been straight up removed.

```
 // Now we need to create a model instance with actual weights.
const weights = [4]f16{ 2.0, -2.0, 1.0, -1.0 };
const bias = [4]f16{ 1.0, 2.0, 3.0, 4.0 };

var model_weights: zml.Bufferized(Layer) = .{
    .bias = try zml.Buffer.fromSlice(platform, .{4}, &bias),
    .weight = try zml.Buffer.fromSlice(platform, .{4}, &weights),
};
defer zml.aio.unloadBuffers(&model_weights);
```

Since `zml.Bufferized(Layer)` is a normal struct we have type checking enabled,
and the Zig compiler will tell us if we forget to load a struct.
Here we hard-coded the weights in the code,
but typically weights would be read from disk at this point using `BufferStore`.

### Putting it all together

Now we want to bind the weights we just loaded,
with the executable we started to compile earlier on.

```zig
// Wait for compilation to finish
const compiled = try compilation.awaitt();
defer compiled.deinit();

// pass the model weights to the compiled module to create an executable module.
const executable = compiled.prepare(model_weights);
```

`prepare` will check that the shapes of the buffers in `model_weights` match
with the shapes given to the compiler.
Also note we don't call `executable.deinit()` since it re-uses the memory of `compiled`.


### Calling / running the Model

The executable can now be invoked with an input of our choice.

```zig
// prepare the input buffer
    const input = [4]f16{ 5.0, 5.0, 5.0, 5.0 };
    var input_buffer = try zml.Buffer.fromSlice(platform, input_shape, &input);
    defer input_buffer.deinit();

    // call our executable module, the result is still on the device.
    var result: zml.Buffer = executable.call(.{input_buffer});
    defer result.deinit();

    // copy the result to CPU memory
    const cpu_result: zml.HostBuffer = try result.toHostAlloc(gpa);
    defer cpu_result.deinit(gpa);
    std.debug.print(
        "\nThe result of {d} * {d} + {d} = {d}\n",
        .{ &weights, &input, &bias, cpu_result.items(f16) },
    );
```

Note that the result of a computation is usually residing in the memory of the
computation device, so with `.toHostAlloc()` we bring it back to CPU memory in
the form of a `HostBuffer`. After that, we can print it.

In order to print it, we need to tell the host buffer how to interpret the
memory. We do that by calling `.items(f16)`, making it cast the memory to `f16`
items.

And that's it! Now, let's have a look at building and actually running this
example!


## Building it

As mentioned already, ZML uses Bazel; so to build our model, we just need to
create a simple `BUILD.bazel` file, next to the `main.zig` file, like this:

```python
load("@rules_zig//zig:defs.bzl", "zig_binary")

zig_binary(
    name = "simple_layer",
    main = "main.zig",
    deps = [
        "@zml//async",
        "@zml//zml",
    ],
)
```

To produce an executable, we import `zig_cc_binary` from the zig rules, and
pass it a name and the zig file we just wrote. The dependencies in `deps` are
what's needed for a basic ZML executable and correlate with our imports at the
top of the Zig file:

```zig
const zml = @import("zml");
const asynk = @import("async");
```


## Running it

With everything in place now, running the model is easy:

```
# run release (--config=release)
bazel run --config=release //examples/simple_layer

# compile and run debug version
bazel run //simple_layer
```

And voila! Here's the output:

```
bazel run --config=release //simple_layer
INFO: Analyzed target //simple_layer:simple_layer (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //simple_layer:simple_layer up-to-date:
  bazel-bin/simple_layer/simple_layer
INFO: Elapsed time: 0.120s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/simple_layer/simple_layer
info(pjrt): Loaded library: libpjrt_cpu.dylib
info(zml_module): Compiling main.Layer.forward with { Shape({3}, dtype=.f16) }

The result of { 2, 2, 2 } * { 5, 5, 5 } + { 1, 2, 3 } = { 11, 12, 13 }
```

---

You can access the complete source code of this walkthrough here:

- [main.zig](https://github.com/zml/zml/tree/master/examples/simple_layer/main.zig)
- [BUILD.bazel](https://github.com/zml/zml/tree/master/examples/simple_layer/BUILD.bazel)


## The complete example

```zig
const std = @import("std");

const asynk = @import("async");
const zml = @import("zml");

/// Model definition
const Layer = struct {
    bias: ?zml.Tensor = null,
    weight: zml.Tensor,

    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        var y = self.weight.mul(x);
        if (self.bias) |bias| {
            y = y.add(bias);
        }
        return y;
    }
};

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    // Start ZML, detect available devices and chose a platform.
    var context = try zml.Context.init();
    defer context.deinit();
    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    // Create a skeleton for our model with only the shapes.
    // The `buffer_id` are typically not set manually
    // but instead set by a zml.BufferStore
    // to map abstract Tensors to actual part of the model weight file.
    const model_shapes: Layer = .{
        .bias = zml.Tensor{
            ._shape = zml.Shape.init(.{4}, .f16).withSharding(.{-1}),
            ._id = .{ .buffer_id = 0 },
        },
        .weight = zml.Tensor{
            ._shape = zml.Shape.init(.{4}, .f16).withSharding(.{-1}),
            ._id = .{ .buffer_id = 1 },
        },
    };

    // Start compilation in a different thread.
    // We already specified the shape of the model weights,
    // but we still need to specifiy the shape of the input tensor.
    const input_shape = zml.Shape.init(.{4}, .f16);
    var compilation = try asynk.asyncc(
        zml.compileModel,
        .{ gpa, Layer.forward, model_shapes, .{input_shape}, platform },
    );

    // Now we need to create a model instance with actual weights.
    const weights = [4]f16{ 2.0, -2.0, 1.0, -1.0 };
    const bias = [4]f16{ 1.0, 2.0, 3.0, 4.0 };

    var model_weights: zml.Bufferized(Layer) = .{
        .bias = try zml.Buffer.fromSlice(platform, .{4}, &bias),
        .weight = try zml.Buffer.fromSlice(platform, .{4}, &weights),
    };
    defer zml.aio.unloadBuffers(&model_weights);

    // Wait for compilation to finish
    const compiled = try compilation.awaitt();
    defer compiled.deinit();

    // pass the model weights to the compiled module to create an executable module.
    // This is where the shapes of the buffers will be compared to the
    // shape expected by the executable.
    // Note: we don't call `executable.deinit()` since it uses the same memory than `compiled`.
    const executable = compiled.prepare(model_weights);

    // prepare the input buffer
    const input = [4]f16{ 5.0, 5.0, 5.0, 5.0 };
    var input_buffer = try zml.Buffer.fromSlice(platform, input_shape, &input);
    defer input_buffer.deinit();

    // call our executable module, the result is still on the device.
    var result: zml.Buffer = executable.call(.{input_buffer});
    defer result.deinit();

    // copy the result to CPU memory
    const cpu_result: zml.HostBuffer = try result.toHostAlloc(gpa);
    defer cpu_result.deinit(gpa);
    std.debug.print(
        "\nThe result of {d} * {d} + {d} = {d}\n",
        .{ &weights, &input, &bias, cpu_result.items(f16) },
    );
}
```

## Where to go from here

- [Add some weights files to your model](../howtos/add_weights.md)
- [Run the model on GPU](../tutorials/getting_started.md)
- [Deploy the model on a server](../howtos/deploy_on_server.md)
- [Dockerize this model](../howtos/dockerize_models.md)
- [Learn more about ZML concepts](../learn/concepts.md)
- [Find out how to best port PyTorch models](../howtos/howto_torch2zml.md)
