const std = @import("std");

const stdx = @import("stdx");

pub const cute = @import("cute/cute.zig");
pub const metal = @import("metal/metal.zig");
pub const nki = @import("nki/nki.zig");
pub const tcl = @import("tcl/tcl.zig");
const zml = @import("../../zml.zig");
const Options = @import("../../attention/Options.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    return switch (platform.target) {
        .neuron => true,
        .metal => true,
        .furiosa => true,
        .cuda => if (cute.isAvailable(platform)) true else false,
        .cpu, .rocm, .tpu, .oneapi => false,
    };
}

pub const Backend = enum {
    cute,
    nki,
    metal,
    tcl,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .neuron => .nki,
            .metal => .metal,
            .cuda => .cute,
            .furiosa => .tcl,
            .cpu, .rocm, .tpu, .oneapi => stdx.debug.panic("Flash attention is not available on the {} platform", .{platform.target}),
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .cute => cute.isAvailable(platform),
            .nki => platform.target == .neuron,
            .metal => platform.target == .metal,
            .tcl => platform.target == .furiosa,
        };
    }
};

pub const Parameters = union(Backend) {
    cute: void,
    nki: nki.Parameters,
    metal: void,
    tcl: void,

    pub const InitOptions = union(Backend) {
        cute: void,
        nki: nki.Parameters,
        metal: void,
        tcl: void,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .cute => .{ .cute = {} },
                .nki => .{ .nki = .init() },
                .metal => .{ .metal = {} },
                .tcl => .{ .tcl = {} },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .cute => .{ .cute = {} },
            .nki => |v| .{ .nki = v },
            .metal => .{ .metal = {} },
            .tcl => .{ .tcl = {} },
        };
    }
};

pub const Metadata = union(Backend) {
    cute: void,
    nki: void,
    metal: metal.Metadata,
    tcl: void,

    pub const InitOptions = union(Backend) {
        cute: void,
        nki: void,
        metal: void,
        tcl: void,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .cute => .{ .cute = {} },
                .nki => .{ .nki = {} },
                .metal => .{ .metal = {} },
                .tcl => .{ .tcl = {} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .cute => .{ .cute = {} },
            .nki => .{ .nki = {} },
            .metal => .{ .metal = .init() },
            .tcl => .{ .tcl = {} },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .cute => .{ .cute = {} },
            .nki => .{ .nki = {} },
            .tcl => .{ .tcl = {} },
            inline else => |v, tag| @unionInit(zml.Bufferized(Metadata), @tagName(tag), try v.initBuffer(io, platform)),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .cute => {},
            .nki => {},
            .tcl => {},
            .metal => |*v| metal.Metadata.deinitBuffer(v),
        }
    }
};

/// Causal attention as used in transformers.
///
/// **Shapes**:
///   - q, result: .{ .q, .h, .hd }
///   - k, v:      .{ .k, .h, .hd }
///
/// Where:
///   - .h is the number of head
///   - .q is the number of queries
///   - .k is the number of keys
///   - .hd is the head dimension
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters, opts: Options) zml.Tensor {
    zml.stdx.debug.assert(zml.Compiler.current().manual_computation_depth > 0, "Attention kernels require an explicit manualComputation boundary", .{});
    return switch (parameters) {
        .cute => cute.attention(q, k, v, token_index, opts.causal),
        .tcl => tcl.attention(q, k, v, token_index, opts.causal),
        // The NKI and Metal kernels are causal only.
        .nki => if (opts.causal) nki.attention(q, k, v, token_index, parameters.nki) else zml.nn.sdpa(q, k, v, .{}),
        .metal => if (opts.causal) metal.attention(q, k, v, token_index, metadata.metal) else zml.nn.sdpa(q, k, v, .{}),
    };
}
