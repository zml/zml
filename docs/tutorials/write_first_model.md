
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

## The high-level Overview

Before firing up our editor, let's quickly talk about a few basic ZML
fundamentals.

In ZML, we describe a _Module_, which represents our AI model, as a Zig
`struct`. That struct can contain Tensor fields that are used for computation,
e.g. weights and biases. In the _forward_ function of a Module, we describe the
computation by calling tensor operations like _mul_, _add_, _dot_,
_conv2D_, etc., or even nested Modules.

ZML creates an MLIR representation of the computation when we compile the
_Module_. For compilation, only the _Tensors_ are required. No actual tensor data
is needed at this step. This is important for large models: we can compile them 
while the actual weight data is being fetched from disk.

To accomplish this, ZML uses a _TensorStore_. The _TensorStore_ loads everything
that is required to make _Tensors_ and later materialize them to _Buffers_. In our
example though, we won't even use a _TensorStore_ as it's optional. We will directly
create tensors using their shape and data type.

After compilation is done, we get what we call an _Executable_. From this _Executable_ 
we can create _Args_ and _Results_, two structs that will store respectively the inputs 
and outputs of a computation done with an _Executable_.

**So the steps for us are:**

- describe the computation as ZML _Module_, using tensor operations
- initialize the _Module_
- compile the _Module_ to produce an _Executable_
- make a "bufferized" version of the _Module_, containing the actual data on the 
  computation device
- prepare the _Args_ and _Results_ of a computation and call the _Executable_
- get the result back to CPU memory and print it

If you like to read more about the underlying concepts of the above, please see
[ZML Concepts](../learn/concepts.md).


## The code

Let's start by writing some Zig code, importing ZML and often-used modules:

```zig
const std = @import("std");
const zml = @import("zml");
```

You will use above lines probably in all ZML projects.


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

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    
    ...
}
```

The above `main()` function takes advantage of Zig's "fat main" to avoid having to write 
boilerplate code for allocations and io.

We also get our ZML CPU platform `platform` automatically.

### Initializing the Module

Next, we need to initialize the module so that we can compile it.

```zig
const layer: Layer = .{
    .weight = zml.Tensor.init(.{3}, .f16),
    .bias = zml.Tensor.init(.{3}, .f16),
};
```

As you can see, it's pretty simple. You can create _Tensors_ from their shape and data type.
Alternatively, you are free to add an arbitrary `.init()` function to your _Module_ if the
initialization code is complex.


### Compiling our Module for the accelerator

We're only going to use the CPU for our simple model, but we need to compile the
`forward()` function nonetheless:

```zig
// Our computation require an input tensor
const input: zml.Tensor = .init(.{3}, .f16);

const sharding = try zml.sharding.replicatedSharding(platform);

var executable = try platform.compile(allocator, io, layer, .forward, .{input}, .{ .shardings = &.{sharding}});
defer executable.deinit();
```

You might wonder what this `sharding` variable is for ?
TODO: explain things about sharding ?


### Creating the "bufferized" Model

Now that we have compiled the module, we can take care of loading the actual tensor data.
In our case, we'll create the _Buffers_ from the raw data directly (but keep in mind that ZML
provides ways to have extra-low loading times).

```zig

const weight_slice: zml.Slice = .init(layer.weight.shape(), std.mem.sliceAsBytes(&[3]f16{1.0, 2.0, 3.0}));
const bias_slice: zml.Slice = .init(layer.bias.?.shape(), std.mem.sliceAsBytes(&[3]f16{1.0, 1.0, 1.0}));
var layer_buffers: zml.Bufferized(Layer) = .{
    .weight = try zml.Buffer.fromSlice(io, platform, weight_slice, sharding),
    .bias = try zml.Buffer.fromSlice(io, platform, bias_slice, sharding),
};
defer layer_buffers.weight.deinit();
defer layer_buffers.bias.?.deinit();
```


### Calling / running the Model

Before being able to run our _Executable_ there is still two steps:
- create the input buffer
- create and fill the _Args_ and _Results_ structs

To create the `input` we already know how to do it, as we did it for the _Module_ 
buffers.

To create the _Args_ and _Results_, the _Executable_ exposes two conveniences.

```zig
// create the input buffer
const input_slice: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(&[3]f16{ 5.0, 5.0, 5.0 }));
var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice, sharding);
defer input_buffer.deinit();

// create the Args and Results structs
var args = try executable.args(allocator);
defer args.deinit(allocator);

var results = try executable.results(allocator);
defer results.deinit(allocator);

// fill the Args
args.set(.{ layer_buffers, input_buffer });

// call our executable 
executable.call(args, &results);

// Retrieve the resulting buffer
var result = results.get(zml.Buffer);
defer result.deinit();

// fetch the result buffer to CPU memory
const result_slice = try result.toSliceAlloc(allocator, io);
defer result_slice.free(allocator);

std.debug.print(
    "\n\nThe result of {d} * {d} + {d} = {d}\n",
    .{ weight_slice, input_slice, bias_slice, result_slice },
);
```

Note that the result of a computation is usually residing in the memory of the
computation device, so with `.toSliceAlloc()` we bring it back to CPU memory in
the form of a `Slice`. After that, we can print it.

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
        "@zml//zml",
    ],
)
```

To produce an executable, we import `zig_binary` from the zig rules, and
pass it a name and the zig file we just wrote. The dependency in `deps` is
what's needed for a basic ZML executable and correlate with our import at the
top of the Zig file:

```zig
const zml = @import("zml");
```


## Running it

With everything in place now, running the model is easy:

```
# run release (--config=release)
bazel run --config=release //examples/simple_layer

# compile and run debug version
bazel run //examples/simple_layer
```

And voila! Here's the output:

```
bazel run --config=release //examples/simple_layer
INFO: Analyzed target //examples/simple_layer:simple_layer (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples/simple_layer:simple_layer up-to-date:
  bazel-bin/examples/simple_layer/simple_layer
INFO: Elapsed time: 0.120s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/examples/simple_layer/simple_layer
info(pjrt): Loaded library: libpjrt_cpu.dylib
debug(zml/module):
******** ZML generated MLIR ********
module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @replicated = <["bus"=4]>
  func.func public @main(%arg0: tensor<3xf16> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>}, %arg1: tensor<3xf16> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>}, %arg2: tensor<3xf16> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>}) -> (tensor<3xf16> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>}) {
    %0 = stablehlo.multiply %arg1, %arg2 : tensor<3xf16>
    %1 = stablehlo.add %0, %arg0 : tensor<3xf16>
    return %1 : tensor<3xf16>
  }
}



The result of {1,2,3} * {5,5,5} + {1,1,1} = {6,11,16}
```


## The complete example

```zig
const std = @import("std");
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const layer: Layer = .{
        .weight = zml.Tensor.init(.{3}, .f16),
        .bias = zml.Tensor.init(.{3}, .f16),
    };

    // Our computation require an input tensor
    const input: zml.Tensor = .init(.{3}, .f16);

    const sharding = try zml.sharding.replicatedSharding(platform);

    var executable = try platform.compile(allocator, io, layer, .forward, .{input}, .{ .shardings = &.{sharding} });
    defer executable.deinit();

    const weight_slice: zml.Slice = .init(layer.weight.shape(), std.mem.sliceAsBytes(&[3]f16{ 1.0, 2.0, 3.0 }));
    const bias_slice: zml.Slice = .init(layer.bias.?.shape(), std.mem.sliceAsBytes(&[3]f16{ 1.0, 1.0, 1.0 }));
    var layer_buffers: zml.Bufferized(Layer) = .{
        .weight = try zml.Buffer.fromSlice(io, platform, weight_slice, sharding),
        .bias = try zml.Buffer.fromSlice(io, platform, bias_slice, sharding),
    };
    defer layer_buffers.weight.deinit();
    defer layer_buffers.bias.?.deinit();

    // create the input buffer
    const input_slice: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(&[3]f16{ 5.0, 5.0, 5.0 }));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice, sharding);
    defer input_buffer.deinit();

    // create the Args and Results structs
    var args = try executable.args(allocator);
    defer args.deinit(allocator);

    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    // fill the Args
    args.set(.{ layer_buffers, input_buffer });

    // call our executable
    executable.call(args, &results);

    // Retrieve the resulting buffer
    var result = results.get(zml.Buffer);
    defer result.deinit();

    // fetch the result buffer to CPU memory
    const result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);

    std.debug.print(
        "\n\nThe result of {d} * {d} + {d} = {d}\n",
        .{ weight_slice, input_slice, bias_slice, result_slice },
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
