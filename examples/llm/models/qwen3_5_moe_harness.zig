const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("common.zig");
const qwen3_5_moe = @import("qwen3_5_moe.zig");
const model = qwen3_5_moe.model;

const log = std.log.scoped(.qwen3_5_moe_harness);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    layer: usize = 0,
    tokens: u32 = 8,

    pub const help =
        \\Use qwen3_5_moe_harness --model=<path> [options]
        \\
        \\ Compile and run a single Qwen3.5 MoE layer with model/expert sharding,
        \\ including router, shared-expert path, and full MoE path.
        \\
        \\ Options:
        \\   --model=<path>      Path to model repository
        \\   --layer=<index>     Layer index to test (default: 0)
        \\   --tokens=<n>        Sequence length for synthetic input (default: 8)
        \\
    ;
};

const SharedExpertModule = struct {
    shared_expert: model.Mlp,
    shared_expert_gate: zml.nn.Linear,

    pub fn forward(self: @This(), x: zml.Tensor) zml.Tensor {
        const shared_gate = self.shared_expert_gate.forward(x).sigmoid().broad(x.shape());
        return self.shared_expert.forward(x)
            .rename(.{ .dout = .d })
            .mul(shared_gate)
            .withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated });
    }
};

const FullMoeModule = struct {
    moe: model.Moe,

    pub fn forward(self: @This(), x: zml.Tensor, moe_metadata: zml.moe.Metadata, moe_parameters: zml.moe.Parameters) zml.Tensor {
        return self.moe.forward(x, moe_metadata, moe_parameters);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(io, build_working_directory, .{});
        defer working_dir.close(io);
        try std.process.setCurrentDir(io, working_dir);
    }

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const shardings = try common.Shardings.init(platform);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var loaded = try qwen3_5_moe.LoadedModel.init(allocator, io, repo, store.view(), .{});
    defer loaded.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = "qwen3_5_moe_harness" });
    defer progress.end();

    var buffers = try loaded.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer loaded.unloadBuffers(&buffers, allocator);

    try runLayerHarness(allocator, io, platform, loaded.inner, buffers, shardings, args.layer, args.tokens);
}

fn runLayerHarness(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: model.Model,
    model_buffers: model.Buffers,
    shardings: common.Shardings,
    layer_index: usize,
    tokens: u32,
) !void {
    if (layer_index >= mdl.text_model.layers.len) return error.LayerOutOfRange;

    const layer = mdl.text_model.layers[layer_index];
    const layer_buffers = model_buffers.text_model.layers[layer_index];

    const hidden_size = mdl.config.text_config.hidden_size;
    const hidden_dtype = mdl.text_model.embed_tokens.weight.dtype();
    const input_shape = zml.Shape.init(.{ .b = 1, .s = @as(i64, @intCast(tokens)), .d = hidden_size }, hidden_dtype).withPartitioning(.{
        .b = .replicated,
        .s = .replicated,
        .d = .replicated,
    });
    const x: zml.Tensor = .fromShape(input_shape);

    var input_slice = try zml.Slice.alloc(allocator, input_shape);
    defer input_slice.free(allocator);
    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.now(.real, io).toNanoseconds()));
    fillSliceRandom(&input_slice, hidden_dtype, prng.random());

    var input_buffer = try zml.Buffer.fromSlice(io, platform, input_slice, .replicated);
    defer input_buffer.deinit();

    const moe_dtype = layer.moe.gate_up_proj.dtype();
    const moe_backend = try zml.moe.Backend.auto(platform, moe_dtype);
    const moe_metadata = zml.moe.Metadata.init(.fromBackend(moe_backend));
    const moe_parameters = zml.moe.Parameters.init(.fromBackend(
        moe_backend,
        mdl.config.text_config.num_experts_per_tok,
        zml.moe.triton.Parameters.ActivationMode.silu,
    ));

    const all_shardings = shardings.all();

    const router_exe = try platform.compile(allocator, io, layer.moe.router, .forward, .{x}, .{ .shardings = &all_shardings });
    defer router_exe.deinit();

    const shared_module = SharedExpertModule{
        .shared_expert = layer.moe.shared_expert,
        .shared_expert_gate = layer.moe.shared_expert_gate,
    };
    const shared_exe = try platform.compile(allocator, io, shared_module, .forward, .{x}, .{ .shardings = &all_shardings });
    defer shared_exe.deinit();

    const full_module = FullMoeModule{ .moe = layer.moe };
    const full_exe = try platform.compile(allocator, io, full_module, .forward, .{ x, moe_metadata, moe_parameters }, .{ .shardings = &all_shardings });
    defer full_exe.deinit();

    var router_args = try router_exe.args(allocator);
    defer router_args.deinit(allocator);
    router_args.set(.{ layer_buffers.moe.router, input_buffer });

    var router_results = try router_exe.results(allocator);
    defer router_results.deinit(allocator);
    router_exe.call(router_args, &router_results);

    var router_scores = router_results.get(zml.Buffer);
    defer router_scores.deinit();
    var topk_ids = router_results.get(zml.Buffer);
    defer topk_ids.deinit();

    const shared_module_buffers: zml.Bufferized(SharedExpertModule) = .{
        .shared_expert = layer_buffers.moe.shared_expert,
        .shared_expert_gate = layer_buffers.moe.shared_expert_gate,
    };

    var shared_args = try shared_exe.args(allocator);
    defer shared_args.deinit(allocator);
    shared_args.set(.{ shared_module_buffers, input_buffer });

    var shared_results = try shared_exe.results(allocator);
    defer shared_results.deinit(allocator);
    shared_exe.call(shared_args, &shared_results);

    var shared_output = shared_results.get(zml.Buffer);
    defer shared_output.deinit();

    const full_module_buffers: zml.Bufferized(FullMoeModule) = .{
        .moe = layer_buffers.moe,
    };

    var full_args = try full_exe.args(allocator);
    defer full_args.deinit(allocator);
    full_args.set(.{ full_module_buffers, input_buffer });

    var full_results = try full_exe.results(allocator);
    defer full_results.deinit(allocator);
    full_exe.call(full_args, &full_results);

    var full_output = full_results.get(zml.Buffer);
    defer full_output.deinit();

    if (!shared_output.shape().eql(input_shape) or !full_output.shape().eql(input_shape)) {
        return error.UnexpectedOutputShape;
    }

    var router_scores_slice = try zml.Slice.alloc(allocator, router_scores.shape());
    defer router_scores_slice.free(allocator);
    try router_scores.toSlice(io, router_scores_slice);

    const scores = router_scores_slice.items(f32);
    var score_sum: f64 = 0;
    for (scores) |v| score_sum += v;
    const mean_score = if (scores.len > 0) score_sum / @as(f64, @floatFromInt(scores.len)) else 0;

    var topk_ids_slice = try zml.Slice.alloc(allocator, topk_ids.shape());
    defer topk_ids_slice.free(allocator);
    try topk_ids.toSlice(io, topk_ids_slice);

    log.info("Layer {d} MoE harness succeeded", .{layer_index});
    log.info("Input shape: {f}", .{input_shape});
    log.info("Router scores shape: {f}", .{router_scores.shape()});
    log.info("Top-k ids shape: {f}", .{topk_ids.shape()});
    log.info("Shared expert output shape: {f}", .{shared_output.shape()});
    log.info("Full MoE output shape: {f}", .{full_output.shape()});
    log.info("Router scores mean: {d:.6}", .{mean_score});

    const topk_values = topk_ids_slice.items(i32);
    const preview_len = @min(topk_values.len, 8);
    if (preview_len > 0) {
        log.info("Top-k preview: {any}", .{topk_values[0..preview_len]});
    }
}

fn fillSliceRandom(slice: *zml.Slice, dtype: zml.DataType, random: std.Random) void {
    switch (dtype.class()) {
        .float => switch (dtype) {
            inline else => |dt| {
                const T = dt.toZigType();
                const values = slice.items(T);
                for (values) |*v| {
                    const unit = random.float(f32);
                    const signed_unit = unit * 2.0 - 1.0;
                    v.* = @field(zml.DataType.Value.init(dt, signed_unit), @tagName(dt));
                }
            },
        },
        .integer => switch (dtype) {
            inline else => |dt| {
                const T = dt.toZigType();
                const values = slice.items(T);
                for (values) |*v| {
                    v.* = @field(zml.DataType.Value.init(dt, random.int(i32)), @tagName(dt));
                }
            },
        },
        .bool => {
            const values = slice.items(bool);
            for (values) |*v| v.* = random.boolean();
        },
        .complex => switch (dtype) {
            .c64 => {
                const T = zml.DataType.c64.toZigType();
                const values = slice.items(T);
                for (values) |*v| {
                    const re = random.float(f32) * 2.0 - 1.0;
                    const im = random.float(f32) * 2.0 - 1.0;
                    v.* = zml.DataType.Value.init(.c64, T.init(re, im)).c64;
                }
            },
            .c128 => {
                const T = zml.DataType.c128.toZigType();
                const values = slice.items(T);
                for (values) |*v| {
                    const re = random.float(f64) * 2.0 - 1.0;
                    const im = random.float(f64) * 2.0 - 1.0;
                    v.* = zml.DataType.Value.init(.c128, T.init(re, im)).c128;
                }
            },
            else => unreachable,
        },
    }
}
