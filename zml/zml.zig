//! Welcome to the ZML API documentation!
//! ZML provides tools to write high level code describing a neural network,
//! compiling it for various accelerators and targets, and executing it.
//!

pub const Buffer = @import("buffer.zig").Buffer;
pub const Bufferized = @import("tensor.zig").Bufferized;
pub const CompilationOptions = @import("platform.zig").CompilationOptions;
pub const Context = @import("context.zig").Context;
pub const Data = @import("dtype.zig").Data;
pub const DataType = @import("dtype.zig").DataType;
pub const HostBuffer = @import("hostbuffer.zig").HostBuffer;
pub const Platform = @import("platform.zig").Platform;
pub const Shape = @import("shape.zig").Shape;
pub const ShapeOf = @import("tensor.zig").ShapeOf;
pub const Target = @import("platform.zig").Target;
pub const Tensor = @import("tensor.zig").Tensor;

// Namespaces
pub const context = @import("context.zig");
pub const exe = @import("exe.zig");
pub const floats = @import("floats.zig");
pub const helpers = @import("helpers.zig");
pub const nn = @import("nn.zig");
pub const module = @import("module.zig");
pub const meta = @import("meta.zig");
pub const platform = @import("platform.zig");
pub const mlir = @import("mlirx.zig");
pub const pjrt = @import("pjrtx.zig");
pub const testing = @import("testing.zig");
pub const torch = @import("torch.zig");

// pub const tokenizer = @import("tokenizer.zig");
pub const tokenizer = @import("zml/tokenizer");

pub const call = ops.call;
pub const compile = exe.compile;
pub const compileWithPrefix = exe.compileWithPrefix;
pub const compileFn = exe.compileFn;
pub const compileModel = exe.compileModel;
pub const FnExe = exe.FnExe;
pub const ModuleExe = exe.ModuleExe;
pub const ModuleSignature = exe.ModuleSignature;

pub const ops = @import("ops.zig");
pub const tools = struct {
    pub const Tracer = @import("tools/tracer.zig").Tracer;
};

pub const aio = @import("aio.zig");
pub const sentencepiece = @import("aio/sentencepiece.zig");

pub const log = std.log.scoped(.zml);

const std = @import("std");

test {
    // NOTE : testing entrypoint.
    // Don't forget to import your module if you want to declare tests declarations that will be run by //zml:test
    std.testing.refAllDecls(@This());
}
