const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("common.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    model: []const u8,
    activations: []const u8,
    backend: ?attention.Backend = null,
    layer: ?u32 = null,
    only: ?[]const u8 = null,
    include_gpu_only: bool = false,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path> [options]
        \\
        \\ Validate Step 3.5 Flash layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>         Path to the model repository
        \\   --activations=<path>   Path to activation safetensors
        \\   --backend=<text>       Attention backend to use (default: vanilla, cuda_fa2, cuda_fa3)
        \\   --layer=<u32>          Optional transformer layer index to test
        \\   --only=<name>          Optional exact activation prefix to test
        \\   --include-gpu-only     Run fused MoE checks even on non-GPU platforms
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer model_registry.deinit();
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    var activation_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.activations);
    defer activation_registry.deinit();
    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &activation_registry);
    defer activation_store.deinit();

    var parsed_config = try common.parseConfig(model.Config, allocator, io, repo);
    defer parsed_config.deinit();

    const shardings: common.Shardings = try .init(platform);
    var single_backend: [1]attention.Backend = undefined;
    const backends: []const attention.Backend = if (args.backend) |backend| blk: {
        single_backend[0] = backend;
        break :blk single_backend[0..];
    } else &.{ .vanilla, .cuda_fa2, .cuda_fa3 };

    var ctx: TestContext = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .model_store = &model_store,
        .activation_store = &activation_store,
        .config = parsed_config.value,
        .shardings = shardings,
        .all_shardings = shardings.all(),
        .backend = .vanilla,
        .selected_layer = args.layer,
        .only = args.only,
        .include_gpu_only = args.include_gpu_only,
    };
    for (backends) |backend| {
        ctx.backend = backend;
        ctx.tested = 0;
        try ctx.run();
    }
}

const RmsNormLayer = struct {
    rms: model.RmsNorm,

    pub fn forward(self: RmsNormLayer, x: zml.Tensor) zml.Tensor {
        return self.rms.forward(x, .d);
    }
};

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    config: model.Config,
    shardings: common.Shardings,
    all_shardings: [2]zml.Sharding,
    backend: attention.Backend,
    selected_layer: ?u32,
    only: ?[]const u8,
    include_gpu_only: bool,
    tested: usize = 0,

    fn run(self: *TestContext) !void {
        std.log.info("Step 3.5 black-box layer tests: backend={s}", .{@tagName(self.backend)});
        for (0..@as(usize, @intCast(self.config.numMainLayers()))) |layer_idx| {
            if (self.selected_layer) |selected| {
                if (layer_idx != selected) continue;
            }
            try self.testLayer(layer_idx);
        }

        if (self.tested == 0) {
            std.log.warn("no Step 3.5 fixtures matched the selected filters", .{});
        }
    }

    fn testLayer(self: *TestContext, layer_idx: usize) !void {
        const layer_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}", .{layer_idx});
        defer self.allocator.free(layer_name);
        const layer = try model.TransformerLayer.init(self.model_store.view().withPrefix(layer_name), layer_idx, self.config);
        if (self.layerUsesGpuOnlyMoe(layer_idx)) {
            self.skipGpuOnly(layer_name);
        } else {
            try self.testKvForward(layer_name, layer, .{ .absolute_tolerance = 2e-2, .minimum_close_fraction = 0.99 });
        }

        const attn_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.self_attn", .{layer_idx});
        defer self.allocator.free(attn_name);
        const attn = try model.Attn.init(self.model_store.view().withPrefix(attn_name), layer_idx, self.config);
        try self.testKvForward(attn_name, attn, .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 });

        const input_norm_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
        defer self.allocator.free(input_norm_name);
        try self.testForward(input_norm_name, RmsNormLayer{ .rms = model.RmsNorm.init(self.model_store.view().withPrefix(input_norm_name), 1e-5) }, .{ .absolute_tolerance = 1e-2 });

        const post_norm_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
        defer self.allocator.free(post_norm_name);
        try self.testForward(post_norm_name, RmsNormLayer{ .rms = model.RmsNorm.init(self.model_store.view().withPrefix(post_norm_name), 1e-5) }, .{ .absolute_tolerance = 1e-2 });

        const ffn_limit: f32 = if (layer_idx < self.config.swiglu_limits_shared.len) self.config.swiglu_limits_shared[layer_idx] else 0.0;
        if (self.isMoeLayer(layer_idx)) {
            const moe_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.moe", .{layer_idx});
            defer self.allocator.free(moe_name);
            if (self.layerUsesGpuOnlyMoe(layer_idx)) {
                self.skipGpuOnly(moe_name);
            } else {
                try self.testForward(moe_name, try model.Moe.init(self.model_store.view().withPrefix(moe_name), layer_idx, self.config), .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 });
            }

            const shared_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.share_expert", .{layer_idx});
            defer self.allocator.free(shared_name);
            try self.testForward(shared_name, model.Mlp.init(self.model_store.view().withPrefix(shared_name), ffn_limit), .{ .absolute_tolerance = 1e-2 });
        } else {
            const mlp_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.mlp", .{layer_idx});
            defer self.allocator.free(mlp_name);
            try self.testForward(mlp_name, model.Mlp.init(self.model_store.view().withPrefix(mlp_name), ffn_limit), .{ .absolute_tolerance = 1e-2 });
        }
    }

    fn loadOpts(self: *TestContext) zml.io.LoadOpts {
        return .{
            .parallelism = 1,
            .shardings = &self.all_shardings,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
        };
    }

    fn testForward(self: *TestContext, name: []const u8, layer: anytype, opts: zml.testing.CompareOpts) !void {
        if (!self.shouldRun(name)) return;
        if (!self.hasFixture(name)) return self.skip(name);
        std.log.info("Testing layer: {s}", .{name});

        var layer_buffers = try zml.io.load(@TypeOf(layer), &layer, self.allocator, self.io, self.platform, self.model_store, self.loadOpts());
        defer deinitBuffers(&layer_buffers);

        const view = self.activation_store.view().withPrefix(name);
        var input = try loadBuffer(self.allocator, self.io, self.platform, view, "in.0", .replicated);
        defer input.deinit();
        var expected = try loadBuffer(self.allocator, self.io, self.platform, view, "out.0", .replicated);
        defer expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, zml.Tensor.fromShape(input.shape()) }, .{ .shardings = &self.all_shardings });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, input });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.callOpts(self.io, args, &results, .{ .wait = true });

        var actual = results.get(zml.Buffer);
        defer releaseBuffer(input, &actual);

        zml.testing.expectClose(self.io, actual, expected, opts) catch |err| switch (err) {
            error.TestUnexpectedResult => return,
            else => return err,
        };
        self.tested += 1;
    }

    fn testKvForward(self: *TestContext, name: []const u8, layer: anytype, opts: zml.testing.CompareOpts) !void {
        if (!self.shouldRun(name)) return;
        if (!self.hasFixture(name)) return self.skip(name);
        std.log.info("Testing layer: {s}", .{name});

        var layer_buffers = try zml.io.load(@TypeOf(layer), &layer, self.allocator, self.io, self.platform, self.model_store, self.loadOpts());
        defer deinitBuffers(&layer_buffers);

        const view = self.activation_store.view().withPrefix(name);
        var input = try loadBuffer(self.allocator, self.io, self.platform, view, "in.0", .replicated);
        defer input.deinit();
        var expected = try loadBuffer(self.allocator, self.io, self.platform, view, "out.0", .replicated);
        defer expected.deinit();

        const input_tensor = zml.Tensor.fromShape(input.shape()).withTags(.{ .b, .s, .d });
        const token_index: zml.Tensor = .init(.{ .s = 1 }, .u32);
        const kv_cache = self.kvCacheFor(input_tensor);
        const metadata = self.attentionMetadata(input_tensor.dim(.s));
        const parameters: attention.Parameters = .init(.fromBackend(self.backend));

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, input_tensor, token_index, kv_cache, metadata, parameters }, .{ .shardings = &self.all_shardings });
        defer exe.deinit();

        var token_index_buffer = try zml.Buffer.fromSlice(self.io, self.platform, .init(zml.Shape.init(.{ .s = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0})), .replicated);
        defer token_index_buffer.deinit();
        var kv_cache_buffers = try kv_cache.initBuffer(self.allocator, self.io, self.platform, self.shardings.model);
        defer model.KvCache.AttentionCache.deinitBuffer(&kv_cache_buffers);
        var metadata_buffers = try metadata.initBuffer(self.io, self.platform, self.shardings.model);
        defer attention.Metadata.deinitBuffer(&metadata_buffers);

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, input, token_index_buffer, kv_cache_buffers, metadata_buffers });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.callOpts(self.io, args, &results, .{ .wait = true });

        var actual, var updated_kv_cache = results.get(struct { zml.Buffer, zml.Bufferized(model.KvCache.AttentionCache) });
        defer releaseKvCacheBuffers(kv_cache_buffers, &updated_kv_cache);
        defer releaseBuffer(input, &actual);

        zml.testing.expectClose(self.io, actual, expected, opts) catch |err| switch (err) {
            error.TestUnexpectedResult => return,
            else => return err,
        };
        self.tested += 1;
    }

    fn kvCacheFor(self: *TestContext, input: zml.Tensor) model.KvCache.AttentionCache {
        const raw_shape = zml.Shape.init(.{
            .layer = 1,
            .b = input.dim(.b),
            .k = input.dim(.s),
            .h = @as(i64, @intCast(self.config.num_attention_groups)),
            .hd = @as(i64, @intCast(self.config.head_dim)),
        }, input.dtype());
        const partitions = self.shardings.model.numPartitionsForLogicalAxis(.model);
        return model.KvCache.AttentionCache.init(model.partitionKvCacheShape(raw_shape, @intCast(self.config.num_attention_groups), partitions));
    }

    fn attentionMetadata(self: *TestContext, seqlen: i64) attention.Metadata {
        const num_heads: i64 = @intCast(if (self.config.attention_other_setting) |other|
            @max(self.config.num_attention_heads, other.num_attention_heads)
        else
            self.config.num_attention_heads);
        return .init(.fromBackend(self.backend, seqlen, num_heads));
    }

    fn hasFixture(self: *TestContext, name: []const u8) bool {
        const view = self.activation_store.view().withPrefix(name);
        return view.hasKey("in.0") and view.hasKey("out.0");
    }

    fn shouldRun(self: *const TestContext, name: []const u8) bool {
        if (self.only) |only| return std.mem.eql(u8, only, name);
        return true;
    }

    fn skip(_: *TestContext, name: []const u8) void {
        std.log.warn("skipping {s}: no in.0/out.0 activations recorded", .{name});
    }

    fn skipGpuOnly(self: *const TestContext, name: []const u8) void {
        if (!self.shouldRun(name)) return;
        std.log.warn("skipping {s}: fused MoE uses Triton and current platform is {s}; pass --include-gpu-only to force it", .{ name, @tagName(self.platform.target) });
    }

    fn layerUsesGpuOnlyMoe(self: *const TestContext, layer_idx: usize) bool {
        if (!self.isMoeLayer(layer_idx)) return false;
        if (self.include_gpu_only) return false;
        if (layer_idx >= 43 and layer_idx <= 44) return false;
        return !self.canCompileTritonMoe();
    }

    fn canCompileTritonMoe(self: *const TestContext) bool {
        return switch (self.platform.target) {
            .cuda, .rocm, .oneapi => true,
            else => false,
        };
    }

    fn isMoeLayer(self: *const TestContext, layer_idx: usize) bool {
        for (self.config.moe_layers_enum.layers) |moe_layer_idx| {
            if (moe_layer_idx == layer_idx) return true;
        }
        return false;
    }
};

fn loadBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, view: zml.io.TensorStore.View, key: []const u8, sharding: zml.Sharding) !zml.Buffer {
    const shape = view.getShape(key) orelse return error.NotFound;
    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try view.getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

fn releaseKvCacheBuffers(expected: zml.Bufferized(model.KvCache.AttentionCache), actual: *zml.Bufferized(model.KvCache.AttentionCache)) void {
    releaseBuffer(expected.k, &actual.k);
    releaseBuffer(expected.v, &actual.v);
    releaseBuffer(expected.layer_index, &actual.layer_index);
}

fn releaseBuffer(expected: zml.Buffer, actual: *zml.Buffer) void {
    if (!sameBufferHandle(expected, actual.*)) actual.deinit();
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
