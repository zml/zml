//! Welcome to the ZML API documentation!
//! ZML provides tools to write high level code describing a neural network,
//! compiling it for various accelerators and targets, and executing it.
//!

// Namespaces
const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const runfiles = @import("runfiles");

pub const platform = @import("platform.zig");
pub const Platform = platform.Platform;
pub const Target = platform.Target;
pub const CompilationOptions = platform.CompilationOptions;
pub const floats = @import("floats.zig");
const dtype = @import("dtype.zig");
pub const Data = dtype.Data;
pub const DataType = dtype.DataType;
pub const pjrt = @import("pjrtx.zig");

var mlir_once = std.once(struct {
    fn call() void {
        mlir.registerPasses("Transforms");
    }
}.call);

var runfiles_once = std.once(struct {
    fn call_() !void {
        if (std.process.hasEnvVarConstant("RUNFILES_MANIFEST_FILE") or std.process.hasEnvVarConstant("RUNFILES_DIR")) {
            return;
        }

        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        const allocator = arena.allocator();
        defer arena.deinit();

        var envMap = std.process.EnvMap.init(allocator);
        var r = (try runfiles.Runfiles.create(.{ .allocator = allocator })) orelse return;
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
    mlir_once.call();
}

pub fn deinit() void {}
