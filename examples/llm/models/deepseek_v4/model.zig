const std = @import("std");

const zml = @import("zml");
const common = @import("../common.zig");
const inference = @import("inference.zig");

pub const Config = struct {
};

pub const Buffers = struct {
};

pub const Model = struct {
};

pub const LoadedModel = struct {
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        _ = allocator; // autofix
        _ = io; // autofix
        _ = repo; // autofix
        _ = store; // autofix
        _ = generation; // autofix
        return .{};
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        _ = allocator; // autofix
        _ = self; // autofix
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = store; // autofix
        _ = progress; // autofix
        _ = shardings; // autofix
        return error.NotImplemented;
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self; // autofix
        _ = buffers; // autofix
        _ = allocator; // autofix
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = backend; // autofix
        _ = shardings; // autofix
        _ = seqlen; // autofix
        _ = progress; // autofix
        return error.NotImplemented;
    }
};
