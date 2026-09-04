//! Shared load / compile / host-buffer helpers.
//! Model math is in `encoder.zig`, `pack.zig`, `dit.zig`, and `vae.zig`.

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

const loader_opts: zml.io.Loader.Opts = .{
    .dma_chunks = 8,
    .dma_chunk_size = 64 * zml.MiB,
    .parallelism = 8,
};

// =============================================================================
// Run context
// =============================================================================

/// Allocator, IO, platform, mesh, and progress for one generation.
pub const Run = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: config.Shardings,
    mesh_buf: [1]zml.Sharding,
    progress: *std.Progress.Node,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        shardings: config.Shardings,
        progress: *std.Progress.Node,
    ) Run {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = shardings,
            .mesh_buf = shardings.all(),
            .progress = progress,
        };
    }

    pub fn mesh(self: *const Run) []const zml.Sharding {
        return &self.mesh_buf;
    }
};

// =============================================================================
// Checkpoints
// =============================================================================

/// One safetensors index (`*.safetensors.index.json`) plus its tensor store.
///
/// Call `open` on a stable `*Checkpoint`: `store` holds a pointer to `reg`.
pub const Checkpoint = struct {
    reg: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,

    pub fn open(
        self: *Checkpoint,
        allocator: std.mem.Allocator,
        io: std.Io,
        model_dir: []const u8,
        index: []const u8,
    ) !void {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ model_dir, index });
        self.reg = try .fromPath(allocator, io, path);
        self.store = .fromRegistry(allocator, &self.reg);
    }

    pub fn deinit(self: *Checkpoint) void {
        self.store.deinit();
        self.reg.deinit();
    }

    pub fn view(self: *Checkpoint) zml.io.TensorStore.View {
        return self.store.view();
    }
};

// =============================================================================
// Weight constructors
// =============================================================================

/// `nn.Linear` from `{name}.weight` and optional `{name}` bias.
pub fn linear(
    store: zml.io.TensorStore.View,
    weight_name: []const u8,
    bias_name: ?[]const u8,
    partitions: anytype,
    bias_partitions: anytype,
) zml.nn.Linear {
    return .init(
        store.createTensor(weight_name, .{ .dout, .d }, partitions),
        if (bias_name) |name| store.maybeCreateTensor(name, .{.dout}, bias_partitions) else null,
        .d,
    );
}

pub fn rms(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) zml.nn.RmsNorm {
    return .{ .weight = store.createTensor("weight", tagz, .replicated), .eps = eps };
}

pub fn ln(store: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = store.createTensor("weight", .{.d}, .replicated),
        .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
        .eps = eps,
    };
}

pub fn drop(comptime T: type, m: *zml.Bufferized(T)) void {
    zml.Buffer.deinitAll(T, m);
}

// =============================================================================
// Load / compile / host
// =============================================================================

pub fn initLoader(run: *const Run) !zml.io.Loader {
    return .init(run.allocator, run.platform, loader_opts);
}

/// Bufferize `m` and DMA weights from `store`. Pass a shared `loader` when
/// loading many layers in a loop so DMA stays pipelined.
pub fn load(
    run: *const Run,
    store: *zml.io.TensorStore,
    comptime T: type,
    m: *const T,
    loader: ?*zml.io.Loader,
) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(run.allocator, T, m);
    if (loader) |shared| {
        shared.load(run.io, T, m, &buffers, store, run.mesh(), .{ .progress = run.progress });
        try shared.await(run.io);
        return buffers;
    }
    var owned = try initLoader(run);
    defer owned.deinit();
    owned.load(run.io, T, m, &buffers, store, run.mesh(), .{ .progress = run.progress });
    try owned.await(run.io);
    return buffers;
}

/// Replicated host→device copy of `items`.
pub fn host(run: *const Run, shape: zml.Shape, items: anytype) !zml.Buffer {
    return zml.Buffer.fromBytes(run.io, run.platform, shape, .replicated, std.mem.sliceAsBytes(items));
}

/// Rank-0 f32 buffer. Euler uses this for σ and σ'.
pub fn scalarF32(run: *const Run, value: f32) !zml.Buffer {
    var item = value;
    return zml.Buffer.fromBytes(run.io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&item));
}

/// `values` as f32 or bf16 to match `shape.dtype()`.
pub fn hostF32(run: *const Run, shape: zml.Shape, values: []const f32) !zml.Buffer {
    switch (shape.dtype()) {
        .f32 => return host(run, shape, values),
        .bf16 => {
            const converted = try run.allocator.alloc(zml.floats.BFloat16, values.len);
            defer run.allocator.free(converted);
            for (converted, values) |*dst, src| dst.* = .fromF32(src);
            return host(run, shape, converted);
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

/// Compile `function` with `args` as the example input (shapes + dtypes).
pub fn compileFn(
    comptime function: anytype,
    comptime name: []const u8,
    run: *const Run,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) !zml.FnExe(function) {
    run.progress.increaseEstimatedTotalItems(1);
    const now: std.Io.Timestamp = .now(run.io, .awake);
    const exe = try zml.FnExe(function).compile(
        run.allocator,
        run.io,
        run.platform,
        .{ .shardings = run.mesh(), .program_name = name },
        args,
    );
    log.info("compile {s}: ok [{f}]", .{ name, now.untilNow(run.io, .awake) });
    return exe;
}

/// Linear in the weight dtype, result back in `x`'s dtype.
pub fn applyLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    return lin.forward(x.convert(lin.weight.dtype())).convert(x.dtype());
}

/// 3-axis MM-RoPE (`MiniMaxH3RotaryPosEmbed`): concat t/h/w freqs, then duplicate.
pub fn ropeCat3(pos: zml.Tensor, inv: zml.Tensor) zml.Tensor {
    const parts = pos.convert(.f32).withPartialTags(.{ .s, .ax }).outer(inv).chunkExact(.ax, 3);
    const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
    return zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
}
