//! Welcome to the ZML API documentation!
//! ZML provides tools to write high level code describing a neural network,
//! compiling it for various accelerators and targets, and executing it.
//!

// Namespaces
const std = @import("std");

const c = @import("c");
pub const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
pub const stdx = @import("stdx");
pub const tokenizer = @import("zml/tokenizer");

pub const attention = @import("attention/attention.zig");
pub const Buffer = @import("buffer.zig").Buffer;
pub const constants = @import("constants.zig");
pub const dtype = @import("dtype.zig");
pub const Data = dtype.Data;
pub const DataType = dtype.DataType;
pub const exe = @import("exe.zig");
pub const Exe = exe.Exe;
pub const floats = @import("floats.zig");
pub const io = @import("io.zig");
pub const mem = @import("mem.zig");
pub const Bufferized = mem.Bufferized;
pub const meta = @import("meta.zig");
pub const mlir = @import("mlirx.zig");
pub const module = @import("module.zig");
pub const nn = @import("nn.zig");
pub const ops = @import("ops.zig");
pub const pjrtx = @import("pjrtx.zig");
pub const platform = @import("platform.zig");
pub const Memory = platform.Memory;
pub const Platform = platform.Platform;
pub const Target = platform.Target;
pub const CompilationOptions = platform.CompilationOptions;
pub const safetensors = @import("safetensors.zig");
pub const shape = @import("shape.zig");
pub const Shape = shape.Shape;
pub const sharding = @import("sharding.zig");
pub const slice = @import("slice.zig");
pub const Slice = slice.Slice;
pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const testing = @import("testing.zig");

test "zml" {
    std.testing.refAllDecls(@This());
}

pub const KiB = 1024;
pub const MiB = 1024 * KiB;
pub const GiB = 1024 * MiB;
pub const TiB = 1024 * GiB;
