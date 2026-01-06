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

pub const Buffer = @import("buffer.zig").Buffer;
pub const constants = @import("constants.zig");
pub const dtype = @import("dtype.zig");
pub const Data = dtype.Data;
pub const DataType = dtype.DataType;
pub const exe = @import("exe.zig");
pub const Exe = exe.Exe;
pub const floats = @import("floats.zig");
pub const io = @import("io.zig");
pub const meta = @import("meta.zig");
pub const mlir = @import("mlirx.zig");
pub const module = @import("module.zig");
pub const nn = @import("nn.zig");
pub const platform = @import("platform.zig");
pub const Platform = platform.Platform;
pub const Target = platform.Target;
pub const CompilationOptions = platform.CompilationOptions;
pub const safetensors = @import("safetensors.zig");
pub const shape = @import("shape.zig");
pub const Shape = shape.Shape;
pub const slice = @import("slice.zig");
pub const Slice = slice.Slice;
pub const ConstSlice = slice.ConstSlice;
pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const testing = @import("testing.zig");

var runfiles_once = std.once(struct {
    fn call_() !void {
        if (std.process.hasEnvVarConstant("RUNFILES_MANIFEST_FILE") or std.process.hasEnvVarConstant("RUNFILES_DIR")) {
            return;
        }

        var io_: std.Io.Threaded = .init_single_threaded;
        defer io_.deinit();

        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        const allocator = arena.allocator();
        defer arena.deinit();

        var envMap = std.process.EnvMap.init(allocator);
        var r = (try runfiles.Runfiles.create(.{ .allocator = allocator, .io = io_.io() })) orelse return;
        try r.environment(&envMap);

        var it = envMap.iterator();
        while (it.next()) |entry| {
            const keyZ = try allocator.dupeZ(u8, entry.key_ptr.*);
            const valueZ = try allocator.dupeZ(u8, entry.value_ptr.*);
            _ = c.setenv(keyZ.ptr, valueZ.ptr, 1);
        }
    }

    fn call() void {
        call_() catch @panic("Unable to init runfiles env");
    }
}.call);

pub fn init() void {
    runfiles_once.call();
    mlir.once.call();
}

pub fn deinit() void {}

/// Return a clone of a type with Tensors replaced by Buffer.
/// Non-Tensor metadata is stripped out of the resulting struct.
/// Recursively descends into the type.
pub fn Bufferized(comptime T: type) type {
    return meta.MapRestrict(Tensor, Buffer).map(T);
}

test "zml" {
    std.testing.refAllDecls(@This());
}
