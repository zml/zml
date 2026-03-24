
# How to port Pytorch models to ZML ?


## Requirements

We assume you already have a working ZML project,
and you can run it with a Bazel command like `bazel run //my_project:torch2zml`.
You can refer to [write your first model](../tutorials/write_first_model.md) to do so.
We also assume that you know enough Python to run the reference implementation.

## Overview

Porting Neural Network implementations can be tedious. Some small errors can
degrade the output of the model, in subtle or not so subtle ways. To track down
errors in a model with four thousand layers, we best be organized.

By the way if you are interested in a specific model, be careful that not all
implementations of a model you can find on Github are equivalent. Sometimes
people introduce subtle bugs when porting across Python libraries. Ideally use
the author's implementation, or at least one you have tested yourself.

**The recommended process is as follows:**

1. run the reference implementation on a known input, and sample layer activations
2. start a ZML project and load the sampled reference activations
3. start porting layers one by one, and test individual layers
4. end-to-end test the model

## Sampling reference activations

Pytorch exposes "forward hooks" that allow to inspect the input/output of each
`torch.nn.Module`. That way it is possible to create a dictionary with each
layer input/output, keyed by the name of the layer.

The main caveat is that if you have a functional implementation that doesn't
use `torch.nn.Module`, this technique won't work.

It is the easiest to start from a "huggingface" snippet, or a python script
that calls the model of your choice on an example input. eg:



```python
import torch
import transformers

model_path = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.float16},
    # device="cuda",
    token=token,
)

prompt = "Q: What is the largest animal?\nA:"
output = pipeline(prompt)
print(output)
```

Then edit the script to import [zml_utils](https://github.com/zml/zml/blob/master/tools/zml_utils.py).

`zml_utils.py` is standalone and currently it's not distributed as a python
package, so the simplest way to use it, is to copy it next to your python
script. Then wrap the model/pipeline in a `zml_utils.ActivationCollector`. The
collector wraps the given model, and returns the original results AND the
activations in a dict of `torch.Tensor` when it's being called. After that, you
can save those activations to a `.safetensors` file.

```python
import torch
import transformers
from safetensors.torch import save_file
import zml_utils

model_path = "meta-llama/Llama-3.2-1B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.float16},
    # device="cuda",
)
model, tokenizer = pipeline.model, pipeline.tokenizer

prompt = "Q: What is the largest animal?\nA:"
# Wrap the pipeline, and extract activations.
# Activations files can be huge for big models,
# so let's stop collecting after 1000 layers.
pipeline = zml_utils.ActivationCollector(pipeline, max_layers=1000, stop_after_first_step=True)
output, activations = pipeline(prompt)

# `output` can be `None` if activations collection
# has stopped before the end of the inference
if output:
    print(output)

# Save activations to a file.
filename = model_path.split("/")[-1] + ".activations.safetensors"
save_file(activations, filename)
print(f"Saved {len(activations)} activations to {filename}")
```

Run this script: `python activations.py`

If you're using HuggingFace, make note of the local path where the model is
saved, it should be something like `~/.cache/huggingface/hub/...`. (and should
appear on the console when running the script). We will need it in the next
steps.

## Loading model and activations in ZML

Let's create a basic ZML program that loads the activations and the Pytorch
model. Put the following in `my_project/torch2zml.zig`.

```zig
const std = @import("std");
const log = std.log;

const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    const model_path, const activations_path = args[1..3].*;

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, activations_path);
    defer activations_registry.deinit();
    log.info("Found {} activations in {s}", .{ activations_registry.tensors.count(), activations_path });

    var model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path);
    defer model_registry.deinit();
    log.info("Found {} activations in {s}", .{ model_registry.tensors.count(), model_path });
}
```

And add a `zig_binary` target in `my_project/BUILD.bazel`:

```python
load("@rules_zig//zig:defs.bzl", "zig_binary")

zig_binary(
    name = "torch2zml",
    main = "torch2zml.zig",
    deps = [
        "@zml//zml",
    ],
)
```

Now check that the weights can be loaded correctly using the bazel CLI.

```bash
bazel build //my_project:torch2zml
./bazel-bin/my_project/torch2zml /path/to/my/model.safetensors.index.json ./my_project/Llama-3.2-1B-Instruct.activations.safetensors

info: Found 564 activations in /Users/corendos/dev/zml/python/temp/Llama-3.2-1B-Instruct.activations.safetensors
debug(zml/safetensors): Added new metadata key=format with value=pt
info: Found 146 activations in /Users/corendos/models/Llama-3.2-1B-Instruct/model.safetensors
```

## Initializing an individual layer

In the above Zig code, the `model_registry` struct is a wrapper around a flat
dictionary, containing an entry for each tensor in the model (similar to a
"state dict"). Manipulating a dictionary is generally not very convenient, so
let's convert it to a Zig struct.

Declare the following layer at the bottom of your file:

```zig
const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,
};
```

The `zml.nn.Linear` is the equivalent of `torch.nn.Linear` and is defined by
its `weight` and optional `bias` tensors.

To create such a struct from our `model_registry` dictionary, we first need to
convert it to a `TensorStore`:

```zig
pub fn main(init: std.process.Init) !void {
    ...
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();
}
```

Then, we can initialize our MLP:

```zig
pub fn main(init: std.process.Init) !void {
    ...

    const mlp_view = model_store.view().withPrefix("model.layers.0.mlp");

    const mlp: Mlp = .{
        .up_proj = .init(
            mlp_view.createTensor("up_proj.weight", .{ .dout, .d }, null),
            mlp_view.maybeCreateTensor("up_proj.bias", .{.dout}, null),
            .d,
        ),
        .gate_proj = .init(
            mlp_view.createTensor("gate_proj.weight", .{ .dout, .d }, null),
            mlp_view.maybeCreateTensor("gate_proj.bias", .{.dout}, null),
            .d,
        ),
        .down_proj = .init(
            mlp_view.createTensor("down_proj.weight", .{ .d, .dout }, null),
            mlp_view.maybeCreateTensor("down_proj.bias", .{.d}, null),
            .dout,
        ),
    };
}
```

Build and run, using previous commands.

Typical errors are due to the naming of layers in Zig not matching the naming
in the file.
Double-check everything and don't hesitate to print more things, e.g. in the
Python script. Alternatively, Huggingface's web-interface allows to peek into
`.safetensor` files.


## Loading an individual layer

Currently, the layer is initialized but the weights are still on the disk.
ZML provides a convenience to automatically load weights from an initialized model:

```zig
pub fn main(init: std.process.Init) !void {
    ...

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    var mlp_weights = try zml.io.load(Mlp, &mlp, allocator, io, platform, &model_store, .auto);
}
```

TODO: Explain the load parameters ? Or find sane defaults to not have to worry about it ?


## Testing an individual layer

Finally, we are going to write the actual math code for our `MLP` layer.

```zig
const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj);
        return self.down_proj.forward(output);
    }
};
```

Our MLP layer is now almost ready to be tested with the `zml.testing.testLayer` utility.
As you may have noticed, ZML makes heavy-use of tagged tensors. In the MLP, we expect
to do a dot product on the `.d` dimension. But the activations we load from the
activations files are not tagged. One solution could be to tag the tensors before saving
them to file, but a much simpler solution is to intercept the inputs before providing
them to the layer we want to test, using a wrapper.

```zig
pub const Wrapper = struct {
    mlp: Mlp,

    pub fn forward(wrapper: Wrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{.b, .s, .d});
        return wrapper.mlp.forward(tagged_input);
    }
};

pub fn main(init: std.process.Init) !void {
    ...

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();

    try zml.testing.testLayer(allocator, io, platform, Wrapper{ .mlp = mlp }, .forward, activations_store.view(), "model.model.layers.0.mlp", .{ .mlp = mlp_weights }, &.{}, .{});
}
```

During this phase, you have three kinds of errors that can appear:

* Zig compilation errors: we've all been there, learning a new language
  can be tough. Normally, the compiler should help you figure out what's wrong.
  You can also check [ZML concepts](../learn/concepts.md) that explains types used
  by ZML.
* Buffer not found errors: be careful that you need to use
  the naming scheme of the inference pipeline when loading the activations.
  Depending on how you write your code, you may have a different naming
  convention in the model file and in the activation file. This is because in
  Python, and in particular the `transformers` library, it's not uncommon to
  wrap the model in a `Pipeline` object before using it. So a given layer may
  be named `layer.0.mlp` in the model file, but its activations may be saved
  under `model.layer.0.mlp`.
* MLIR compilation errors: typically this is caused by a mathematical
  error in the `forward` function. To help here, you can log the shapes of the
  input and intermediary values: `std.log.info("x: {}", .{x})`, and put similar
  print statements in the Python code. You can also consider splitting a big
  layer into smaller parts. Since our code only explicitly captures
  `torch.nn.Module` input/output, you may need to modify the Python script to
  add some extra tensors to the dictionary with example input/output of a
  specific function.

## General tips

* Porting models can be hard, especially if the original code is messy, has
  poor comments, behaves differently on different input shapes, or has unused
  code paths. Start by identifying parts of the Python code which are
  **unused**. It is common in research code that some code paths were written
  for one paper, but didn't get used in subsequent papers.

* ZML offers a few Pytorch specific helpers in `zml.torch`; those operators are
  offered to help you port models, but in general they may have weird APIs. If
  you're lucky and the code you are porting has comments indicating "tags", eg
  "C,W,H" of tensors, you can port this to actual tensor attributes using
  `x.withTags(.{.c, .w, .h})`, and use those tags (eg `.c`) to refer to axes
  instead of offsets. E.g. in Pytorch: `x.sum(0) # reduce over channel axis`
  becomes `x.sum(.c)`. More on this topic in
  ["Working with tensors"](../tutorials/working_with_tensors.md).
