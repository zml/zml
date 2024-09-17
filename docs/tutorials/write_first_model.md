
# Writing your first model

**In this short guide, we will do the following:**

- clone ZML to work directly within the prepared example folder
- add Zig code to implement our model
- add some Bazel to integrate our code with ZML
- no weights files or anything external is required for this example

The reason we're doing our excercise in the `examples` folder is because it's
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

To accomplish this, ZML uses a _BufferStore_. The _BufferStore_ knows how to
only load shapes and when to load actual tensor data. In our example, we will
fake the _BufferStore_ a bit: we won't load from disk; we'll use float arrays
instead.

After compilation is done (and the _BufferStore_ has finished loading weights),
we can send the weights from the _BufferStore_ to our computation device. That
produces an _executable_ module which we can call with different _inputs_.

In our example, we then copy the result from the computation device to CPU
memory and print it.

**So the steps for us are:**

- describe the computation as ZML _Module_, using tensor operations
- create a _BufferStore_ that provides _Shapes_ and data of weights and bias
  (ca. 5 lines of code).
- compile the _Module_ **asynchronously**
- make the compiled _Module_ send the weights (and bias) to the computation
  device utilizing the _BufferStore_, producing an _executable_ module
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

// shortcut to the async_ function in the asynk module
const async_ = asynk.async_;
```

You will use above lines probably in all ZML projects. Also, note that **ZML is
async** and comes with its own async runtime, thanks to
[zigcoro](https://github.com/rsepassi/zigcoro).



### Defining our Model

We will start with a very simple "Model". One that resembles a "multiply and
add" operation.

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
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try asynk.AsyncThread.main(gpa.allocator(), asyncMain, .{});
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
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Arena allocator for BufferStore etc.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform();
    ...
}
```

This is boilerplate code that provides a general-purpose allocator and, for
convenience, an arena allocator that we will use later. The advantage of arena
allocators is that you don't need to deallocate individual allocations; you
simply call `.deinit()` to deinitialize the entire arena instead!

We also initialize the ZML context `context` and get our CPU `platform`
automatically.


### The BufferStore 

Next, we need to set up the concrete weight and bias tensors for our model.
Typically, we would load them from disk. But since our example works without
stored weights, we are going to create a BufferStore manually, containing
_HostBuffers_ (buffers on the CPU) for both the `weight` and the `bias` tensor.

A BufferStore basically contains a dictionary with string keys that match the
name of the struct fields of our `Layer` struct. So, let's create this
dictionary:

```zig
// Our weights and bias to use
var weights = [3]f16{ 2.0, 2.0, 2.0 };
var bias = [3]f16{ 1.0, 2.0, 3.0 };
const input_shape = zml.Shape.init(.{3}, .f16);

// We manually produce a BufferStore. You would not normally do that.
// A BufferStore is usually created by loading model data from a file.
var buffers: zml.aio.BufferStore.Buffers = .{};
try buffers.put(arena, "weight", zml.HostBuffer.fromArray(&weights));
try buffers.put(arena, "bias", zml.HostBuffer.fromArray(&bias));

// the actual BufferStore
var bs: zml.aio.BufferStore = .{
    .arena = arena_state,
    .buffers = buffers,
};
```

Our weights are `{2.0, 2.0, 2.0}`, and our bias is just `{1.0, 2.0, 3.0}`. The
shape of the weight and bias tensors is `{3}`, and because of that, the **shape
of the input tensor** is also going to be `{3}`!

Note that `zml.Shape` always takes the data type associated with the tensor. In
our example, that is `f16`, expressed as the enum value `.f16`.



### Compiling our Module for the accelerator

We're only going to use the CPU for our simple model, but we need to compile the
`forward()` function nonetheless. This compilation is usually done
asynchronously. That means, we can continue doing other things while the module
is compiling:

```zig
// A clone of our model, consisting of shapes. We only need shapes for compiling.
// We use the BufferStore to infer the shapes.
const model_shapes = try zml.aio.populateModel(Layer, allocator, bs);

// Start compiling. This uses the inferred shapes from the BufferStore.
// The shape of the input tensor, we have to pass in manually.
var compilation = try async_(
    zml.compileModel,
    .{ allocator, model_shapes, .forward, .{input_shape}, platform },
);

// Produce a bufferized weights struct from the fake BufferStore.
// This is like the inferred shapes, but with actual values.
// We will need to send those to the computation device later.
var model_weights = try zml.aio.loadBuffers(Layer, .{}, bs, arena, platform);
defer zml.aio.unloadBuffers(&model_weights);  // for good practice

// Wait for compilation to finish
const compiled = try compilation.await_();
```

Compiling is happening in the background via the `async_` function. We call
`async_` with the `zml.compileModel` function and its arguments
separately. The arguments themselves are basically the shapes of the weights in
the BufferStore, the `.forward` function name in order to compile
`Layer.forward`, the shape of the input tensor(s), and the platform for which to
compile (we used auto platform).



### Creating the Executable Model

Now that we have compiled the module utilizing the shapes, we turn it into an
executable. 

```zig
// pass the model weights to the compiled module to create an executable module
var executable = try compiled.prepare(arena, model_weights);
defer executable.deinit();
```


### Calling / running the Model

The executable can now be invoked with an input of our choice.

To create the `input`, we directly use `zml.Buffer` by calling
`zml.Buffer.fromArray()`. It's important to note that `Buffer`s reside in
_accelerator_ (or _device_) memory, which is precisely where the input needs to
be for the executable to process it on the device.

For clarity, let's recap the distinction: `HostBuffer`s are located in standard
_host_ memory, which is accessible by the CPU. When we initialized the weights,
we used `HostBuffers` to set up the `BufferStore`. This is because the
`BufferStore` typically loads weights from disk into `HostBuffer`s, and then
converts them into `Buffer`s when we call `loadBuffers()`.

However, for inputs, we bypass the `BufferStore` and create `Buffer`s directly
in device memory.


```zig
// prepare an input buffer
// Here, we use zml.HostBuffer.fromSlice to show how you would create a 
// HostBuffer with a specific shape from an array.
// For situations where e.g. you have an [4]f16 array but need a .{2, 2} input 
// shape.
var input = [3]f16{ 5.0, 5.0, 5.0 };
var input_buffer = try zml.Buffer.from(
    platform,
    zml.HostBuffer.fromSlice(input_shape, &input),
);
defer input_buffer.deinit();

// call our executable module
var result: zml.Buffer = executable.call(.{input_buffer});
defer result.deinit();

// fetch the result buffer to CPU memory
const cpu_result = try result.toHostAlloc(arena);
std.debug.print(
    "\n\nThe result of {d} * {d} + {d} = {d}\n",
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
load("@zml//bazel:zig.bzl", "zig_cc_binary")

zig_cc_binary(
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
# run release (-c opt)
cd examples
bazel run -c opt //simple_layer

# compile and run debug version
bazel run //simple_layer
```

And voila! Here's the output:

```
bazel run -c opt //simple_layer
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
const zml = @import("zml");
const asynk = @import("async");

const async_ = asynk.async_;

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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try asynk.AsyncThread.main(gpa.allocator(), asyncMain, .{});
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Arena allocator for BufferStore etc.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform();

    // Our weights and bias to use
    var weights = [3]f16{ 2.0, 2.0, 2.0 };
    var bias = [3]f16{ 1.0, 2.0, 3.0 };
    const input_shape = zml.Shape.init(.{3}, .f16);

    // We manually produce a BufferStore. You would not normally do that.
    // A BufferStore is usually created by loading model data from a file.
    var buffers: zml.aio.BufferStore.Buffers = .{};
    try buffers.put(arena, "weight", zml.HostBuffer.fromArray(&weights));
    try buffers.put(arena, "bias", zml.HostBuffer.fromArray(&bias));

    // the actual BufferStore
    const bs: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    // A clone of our model, consisting of shapes. We only need shapes for
    // compiling. We use the BufferStore to infer the shapes.
    const model_shapes = try zml.aio.populateModel(Layer, allocator, bs);

    // Start compiling. This uses the inferred shapes from the BufferStore.
    // The shape of the input tensor, we have to pass in manually.
    var compilation = try async_(
        zml.compileModel,
        .{ allocator, model_shapes, .forward, .{input_shape}, platform },
    );

    // Produce a bufferized weights struct from the fake BufferStore.
    // This is like the inferred shapes, but with actual values.
    // We will need to send those to the computation device later.
    var model_weights = try zml.aio.loadBuffers(Layer, .{}, bs, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights); // for good practice

    // Wait for compilation to finish
    const compiled = try compilation.await_();

    // pass the model weights to the compiled module to create an executable 
    // module
    var executable = try compiled.prepare(arena, model_weights);
    defer executable.deinit();

    // prepare an input buffer
    // Here, we use zml.HostBuffer.fromSlice to show how you would create a 
    // HostBuffer with a specific shape from an array.
    // For situations where e.g. you have an [4]f16 array but need a .{2, 2} 
    // input shape.
    var input = [3]f16{ 5.0, 5.0, 5.0 };
    var input_buffer = try zml.Buffer.from(
        platform,
        zml.HostBuffer.fromSlice(input_shape, &input),
    );
    defer input_buffer.deinit();

    // call our executable module
    var result: zml.Buffer = executable.call(.{input_buffer});
    defer result.deinit();

    // fetch the result to CPU memory
    const cpu_result = try result.toHostAlloc(arena);
    std.debug.print(
        "\n\nThe result of {d} * {d} + {d} = {d}\n",
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
