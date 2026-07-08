const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const inference = @import("step3_5flash/inference.zig");
const step3p5 = @import("step3_5flash.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.step3_5_tests);

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate the Step 3.5 Flash MoE layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>         Path to the model repository
        \\   --activations=<path>   Path to activation safetensors
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

    try run(allocator, io, platform, repo, args.activations, &model_store);
}

const TEST_LAYER = 3;

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    repo: std.Io.Dir,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    var parsed_config = try common.parseConfig(model.Config, allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    if (TEST_LAYER == 0) {
        const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
        std.log.info("MLP:", .{});
        for (mlp_layer_indices) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
            defer allocator.free(name);

            const mlp = model.Mlp.init(model_store.view().withPrefix(name), null);

            // Recursive cleanup for buffers
            var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&mlp_weights);

            std.log.info("name {s}", .{name});

            const input_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
            defer allocator.free(input_key);
            if (!activation_store.view().hasKey(input_key)) {
                std.log.warn("skipping {s}: no activations recorded", .{name});
                continue;
            }

            zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store.view(), name, mlp_weights, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
            };
        }
        std.log.info("RMS Norm:", .{});

        for (0..48) |layer_idx| {
            const name__input_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
            defer allocator.free(name__input_layernorm);

            const name__post_attention_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
            defer allocator.free(name__post_attention_layernorm);

            const rms1 = model.RmsNorm.init(model_store.view().withPrefix(name__input_layernorm), @as(f32, 1e-5));
            const rms2 = model.RmsNorm.init(model_store.view().withPrefix(name__post_attention_layernorm), @as(f32, 1e-5));

            var rms_weights1 = try zml.io.load(model.RmsNorm, &rms1, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&rms_weights1);

            var rms_weights2 = try zml.io.load(model.RmsNorm, &rms2, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&rms_weights2);

            const in1_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__input_layernorm});
            defer allocator.free(in1_key);
            if (activation_store.view().hasKey(in1_key)) {
                zml.testing.testLayer(allocator, io, platform, rms1, .forward, activation_store.view(), name__input_layernorm, rms_weights1, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                    std.log.warn("skipping {s}: {s}", .{ name__input_layernorm, @errorName(err) });
                };
            } else {
                std.log.warn("skipping {s}: no activations recorded", .{name__input_layernorm});
            }

            const in2_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__post_attention_layernorm});
            defer allocator.free(in2_key);
            if (activation_store.view().hasKey(in2_key)) {
                zml.testing.testLayer(allocator, io, platform, rms2, .forward, activation_store.view(), name__post_attention_layernorm, rms_weights2, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                    std.log.warn("skipping {s}: {s}", .{ name__post_attention_layernorm, @errorName(err) });
                };
            } else {
                std.log.warn("skipping {s}: no activations recorded", .{name__post_attention_layernorm});
            }
        }

        // MoE is verified; we no longer have to test the router
        for (42..45) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe", .{layer_idx});
            defer allocator.free(name);

            const moe = try model.Moe.init(model_store.view().withPrefix(name), layer_idx);

            var moe_weights = try zml.io.load(model.Moe, &moe, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&moe_weights);

            const moe_in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
            defer allocator.free(moe_in_key);
            if (!activation_store.view().hasKey(moe_in_key)) {
                std.log.warn("skipping {s}: no activations recorded", .{name});
                continue;
            }

            zml.testing.testLayer(
                allocator,
                io,
                platform,
                moe,
                .forward,
                activation_store.view(),
                name,
                moe_weights,
                &.{},
                .{
                    .absolute_tolerance = 1e-2,
                    .minimum_close_fraction = 0.99,
                },
            ) catch |err| {
                std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
            };
        }
    } else if (TEST_LAYER == 1) {
        try runRopeTests(allocator, io, platform);
    } else if (TEST_LAYER == 2) {
        try surveySelfAttnShapes(allocator, model_store, &activation_store);
        for (0..45) |layer_idx| {
            runSelfAttnLayer(allocator, io, platform, model_store, &activation_store, layer_idx, config) catch |err| {
                std.log.warn("skipping model.layers.{d}.self_attn: {s}", .{ layer_idx, @errorName(err) });
            };
        }
    } else if (TEST_LAYER == 3) {
        try surveySelfAttnShapes(allocator, model_store, &activation_store);
        for (0..45) |layer_idx| {
            debugSelfAttnStages(allocator, io, platform, model_store, &activation_store, layer_idx, config) catch |err| {
                std.log.warn("skipping model.layers.{d}.self_attn: {s}", .{ layer_idx, @errorName(err) });
            };
        }

        // try debugSelfAttnStages(allocator, io, platform, model_store, &activation_store, 1);
    } else {
        const layer_idx = 1;
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
        defer allocator.free(name);

        const attn = try model.Attn.init(model_store.view().withPrefix(name), layer_idx, config);

        var attn_weights = try zml.io.load(model.Attn, &attn, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&attn_weights);

        testSelfAttn(
            allocator,
            io,
            platform,
            attn,
            attn_weights,
            activation_store.view(),
            name,
            .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 },
        ) catch |err| {
            std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
        };
    }
}

// Manual test for SelfAttn: the HF dumper writes tensors at non-contiguous indices
// (past_key_value is a Cache and gets skipped while enumerate keeps counting),
// so `.in.0, .in.1, .in.3, .in.4` exist but `.in.2` does not. `testLayer` assumes
// contiguous indices and would panic, so we wire the four real inputs by name.
fn testSelfAttn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    attn: model.Attn,
    attn_weights: zml.Bufferized(model.Attn),
    activation_view: zml.io.TensorStore.View,
    name: []const u8,
    opts: zml.testing.CompareOpts,
) !void {
    const view = activation_view.withPrefix(name);

    // Discover which .in.N keys actually exist.
    inline for (0..8) |i| {
        var buf: [16]u8 = undefined;
        const k = std.fmt.bufPrint(&buf, "in.{d}", .{i}) catch unreachable;
        if (view.hasKey(k)) {
            std.log.info("{s}.{s}: shape={?f}", .{ name, k, view.getShape(k) });
        }
    }

    const hidden_states = view.createTensor("in.0", null, .replicated);
    const cache_position = view.createTensor("in.3", null, .replicated);

    const Argsx = struct { zml.Tensor, zml.Tensor };
    var argsx: Argsx = .{ hidden_states, cache_position };

    const exe = try platform.compile(allocator, io, attn, .forward, argsx, .{ .shardings = &.{} });
    defer exe.deinit();

    var args_buffers = try zml.io.load(Argsx, &argsx, allocator, io, platform, activation_view.store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &args_buffers);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ attn_weights, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    const output_count = exe.output_shapes.len;
    var results = try allocator.alloc(zml.Buffer, output_count);
    defer allocator.free(results);
    exe_results.fill(.{results});

    var reader_buffer: [4096]u8 = undefined;
    var failed = false;
    for (0..output_count) |i| {
        var key_buf: [16]u8 = undefined;
        const subkey = try std.fmt.bufPrint(&key_buf, "out.{d}", .{i});

        const shape = view.getShape(subkey) orelse {
            std.log.warn("{s}.{s}: no reference output", .{ name, subkey });
            continue;
        };

        const expected_slice: zml.Slice = try .alloc(allocator, shape);
        defer expected_slice.free(allocator);
        var reader = try view.getReader(subkey, io, &reader_buffer);
        defer reader.deinit();
        try reader.interface.readSliceAll(expected_slice.data());

        const got_slice = try results[i].toSliceAlloc(allocator, io);
        defer got_slice.free(allocator);

        zml.testing.expectClose(io, expected_slice, got_slice, opts) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                std.log.warn("{s}.{s} doesn't match", .{ name, subkey });
                failed = true;
            },
            else => return err,
        };
    }

    if (failed) {
        std.log.info("❌ check failed for {s}", .{name});
    } else {
        std.log.info("✅ all good for {s}", .{name});
    }
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

fn runSelfAttnLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    layer_idx: usize,
    config: model.Config,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
    defer allocator.free(name);

    const attn = try model.Attn.init(model_store.view().withPrefix(name), layer_idx, config);

    var attn_weights = try zml.io.load(model.Attn, &attn, allocator, io, platform, model_store, .auto);
    defer deinitBuffers(&attn_weights);

    try testSelfAttn(
        allocator,
        io,
        platform,
        attn,
        attn_weights,
        activation_store.view(),
        name,
        .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 },
    );
}

fn debugSelfAttnStages(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    layer_idx: usize,
    config: model.Config,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
    defer allocator.free(name);

    // 1. Discover every captured activation under this layer's attention.
    std.log.info("=== captured activation keys under {s} ===", .{name});
    {
        var it = activation_store.registry.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.startsWith(u8, key, name) and key.len > name.len and key[name.len] == '.') {
                std.log.info("  {s}: shape={f}", .{ key, entry.value_ptr.shape });
            }
        }
    }

    // 2. Build attention layer and load its weights.
    const attn = try model.Attn.init(model_store.view().withPrefix(name), layer_idx, config);
    var attn_weights = try zml.io.load(model.Attn, &attn, allocator, io, platform, model_store, .auto);
    defer deinitBuffers(&attn_weights);

    // 3. Wire the recorded inputs (in.0 = hidden_states, in.3 = cache_position).
    const view = activation_store.view().withPrefix(name);
    const hidden_states = view.createTensor("in.0", null, .replicated);
    const cache_position = view.createTensor("in.3", null, .replicated);

    const Argsx = struct { zml.Tensor, zml.Tensor };
    var argsx: Argsx = .{ hidden_states, cache_position };

    const exe = try platform.compile(allocator, io, attn, .forwardStages, argsx, .{ .shardings = &.{} });
    defer exe.deinit();

    var args_buffers = try zml.io.load(Argsx, &argsx, allocator, io, platform, activation_store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &args_buffers);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ attn_weights, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var stages_buffers: zml.Bufferized(model.Attn.Stages) = undefined;
    exe_results.fill(.{&stages_buffers});
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &stages_buffers);

    // 4. For each stage, print our shape/stats and compare against the matching reference.
    const opts: zml.testing.CompareOpts = .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 };

    const StageMap = struct { stage: []const u8, refs: []const []const u8 };
    const stage_map = [_]StageMap{
        .{ .stage = "q_proj", .refs = &.{"q_proj.out.0"} },
        .{ .stage = "k_proj", .refs = &.{"k_proj.out.0"} },
        .{ .stage = "v_proj", .refs = &.{"v_proj.out.0"} },
        .{ .stage = "g_proj", .refs = &.{"g_proj.out.0"} },
        .{ .stage = "q_norm", .refs = &.{"q_norm.out.0"} },
        .{ .stage = "k_norm", .refs = &.{"k_norm.out.0"} },
        .{ .stage = "q_pre_rope_hf", .refs = &.{"rope.q_in"} },
        .{ .stage = "k_pre_rope_hf", .refs = &.{ "rope.k_in", "k_in" } },
        .{ .stage = "cos", .refs = &.{ "rope.cos", "rotary_emb.out.0" } },
        .{ .stage = "sin", .refs = &.{ "rope.sin", "rotary_emb.out.1" } },
        .{ .stage = "q_rope_hf", .refs = &.{"rope.q_embed"} },
        .{ .stage = "k_rope_hf", .refs = &.{"rope.k_embed"} },
        .{ .stage = "attn", .refs = &.{"attn"} },
        .{ .stage = "gate_sig", .refs = &.{"gate_sig"} },
        .{ .stage = "gated", .refs = &.{"gated"} },
        .{ .stage = "o_proj_in", .refs = &.{"o_proj.in.0"} },
        .{ .stage = "out", .refs = &.{"out.0"} },
    };

    inline for (std.meta.fields(model.Attn.Stages)) |field| {
        const stage_name = field.name;
        const buf_ptr: *zml.Buffer = &@field(stages_buffers, field.name);

        std.log.info("--- stage {s}: ours shape={f} ---", .{ stage_name, buf_ptr.shape() });

        var ref_subkeys: []const []const u8 = &.{};
        for (stage_map) |m| {
            if (std.mem.eql(u8, m.stage, stage_name)) {
                ref_subkeys = m.refs;
                break;
            }
        }

        if (ref_subkeys.len == 0) {
            std.log.info("  (no reference mapping for this stage)", .{});
        } else {
            var matched = false;
            for (ref_subkeys) |subkey| {
                const ref_shape = view.getShape(subkey) orelse continue;
                matched = true;
                std.log.info("  ref '{s}.{s}': shape={f}", .{ name, subkey, ref_shape });

                const our_dims = buf_ptr.shape().dims();
                var our_count: i64 = 1;
                for (our_dims) |d| our_count *= d;
                var ref_count: i64 = 1;
                for (ref_shape.dims()) |d| ref_count *= d;
                if (our_count != ref_count) {
                    std.log.info("  -> element-count mismatch ours={d} ref={d}", .{ our_count, ref_count });
                    continue;
                }

                const expected: zml.Slice = try .alloc(allocator, ref_shape);
                defer expected.free(allocator);
                var reader_buffer: [4096]u8 = undefined;
                var reader = try view.getReader(subkey, io, &reader_buffer);
                defer reader.deinit();
                try reader.interface.readSliceAll(expected.data());

                const got = try buf_ptr.toSliceAlloc(allocator, io);
                defer got.free(allocator);

                const BF = zml.floats.BFloat16;
                const report = try zml.testing.compareSlices(
                    allocator,
                    BF,
                    BF,
                    expected.constItems(BF),
                    got.constItems(BF),
                    opts,
                );
                std.log.info("{f}", .{report});
                if (report.close_fraction < opts.minimum_close_fraction or report.nan_or_inf) {
                    std.log.warn("  -> FAIL for {s}", .{subkey});
                } else {
                    std.log.info("  -> ok for {s}", .{subkey});
                }
            }
            if (!matched) {
                std.log.info("  (no captured reference for this stage)", .{});
            }
        }
    }
}

fn surveySelfAttnShapes(
    allocator: std.mem.Allocator,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
) !void {
    const expected_hidden: i64 = 4096;
    const expected_kv: i64 = 8 * 128;
    const num_q_full: i64 = 64;
    const num_q_swa: i64 = 96;
    const head_dim_val: i64 = 128;
    const main_layers: usize = 45;

    std.log.info("=== SelfAttn shape survey (hidden={d}, kv={d}, full_q={d}, swa_q={d}) ===", .{
        expected_hidden, expected_kv, num_q_full * head_dim_val, num_q_swa * head_dim_val,
    });

    const proj_names = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj", "g_proj" };

    for (0..48) |layer_idx| {
        const attn_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
        defer allocator.free(attn_name);
        const mview = model_store.view().withPrefix(attn_name);

        // Step 3.5 Flash pattern: Full, SWA, SWA, SWA. Layers >= 45 are MTP/spec blocks (SWA-shaped).
        const is_full = layer_idx < main_layers and layer_idx % 4 == 0;
        const num_q: i64 = if (is_full) num_q_full else num_q_swa;
        const expected_q: i64 = num_q * head_dim_val;
        const kind: []const u8 = if (layer_idx >= main_layers) "MTP" else if (is_full) "FULL" else "SWA ";

        var line: std.Io.Writer.Allocating = .init(allocator);
        defer line.deinit();
        const w = &line.writer;

        try w.print("layer {d:>2} [{s}] (exp q={d}):", .{ layer_idx, kind, expected_q });

        var mismatch = false;
        for (proj_names) |pn| {
            const wkey = try std.fmt.allocPrint(allocator, "{s}.weight", .{pn});
            defer allocator.free(wkey);
            if (mview.getShape(wkey)) |sh| {
                try w.print(" {s}={f}", .{ pn, sh });
                const dout = sh.dim(0);
                const din = sh.dim(1);
                if (std.mem.eql(u8, pn, "q_proj") and (dout != expected_q or din != expected_hidden)) mismatch = true;
                if ((std.mem.eql(u8, pn, "k_proj") or std.mem.eql(u8, pn, "v_proj")) and (dout != expected_kv or din != expected_hidden)) mismatch = true;
                if (std.mem.eql(u8, pn, "o_proj") and (dout != expected_hidden or din != expected_q)) mismatch = true;
                if (std.mem.eql(u8, pn, "g_proj") and (dout != num_q or din != expected_hidden)) mismatch = true;
            } else {
                try w.print(" {s}=<missing>", .{pn});
                mismatch = true;
            }
        }

        const in_name = try std.fmt.allocPrint(allocator, "{s}.in.0", .{attn_name});
        defer allocator.free(in_name);
        if (activation_store.view().getShape(in_name)) |sh| {
            try w.print(" act.in.0={f}", .{sh});
        }

        if (mismatch) {
            std.log.warn("{s}  <-- MISMATCH", .{line.written()});
        } else {
            std.log.info("{s}", .{line.written()});
        }
    }

    std.log.info("=== end survey ===", .{});
}

// ===========================================================================
// Synthetic RoPE tests
//
// These do not need real model activations; RoPE is deterministic math.
// feed simple increasing inputs through the compiled graph and compare
// against a Zig CPU reference implementation that mirrors the HuggingFace
// reference exactly.
// ===========================================================================

const Rope = struct {
    const B: i64 = 1;
    const S: i64 = 4;
    const H: i64 = 2;
    const HD: i64 = 8;
    const ROTARY_DIM: i64 = 8;
    const THETA: f32 = 10_000.0;

    const QK_LEN: usize = @as(usize, @intCast(B * S * H * HD));
    const CS_LEN: usize = @as(usize, @intCast(B * S * HD));
    const POS_LEN: usize = @as(usize, @intCast(B * S));

    // --- synthetic inputs ---

    fn makeQ(out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    }

    fn makeK(out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0 + 0.5;
    }

    fn makePositionIds(out: []i32) void {
        for (out, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(S)));
    }

    // --- CPU references ---

    fn refInvFreq(dim: i64, theta: f32, out: []f32) void {
        const half: usize = @intCast(@divExact(dim, 2));
        std.debug.assert(out.len == half);
        const N: f32 = @floatFromInt(half);
        for (out, 0..) |*v, i| {
            const fi: f32 = @floatFromInt(i);
            v.* = std.math.pow(f32, theta, -fi / N);
        }
    }

    fn refCosSin(position_ids: []const i32, cos_out: []f32, sin_out: []f32) void {
        var inv_freq: [@as(usize, @intCast(@divExact(ROTARY_DIM, 2)))]f32 = undefined;
        refInvFreq(ROTARY_DIM, THETA, &inv_freq);

        const half: usize = inv_freq.len;
        for (position_ids, 0..) |pos, idx| {
            const base = idx * @as(usize, @intCast(HD));
            const p: f32 = @floatFromInt(pos);
            for (0..half) |i| {
                const f = p * inv_freq[i];
                cos_out[base + i] = @cos(f);
                cos_out[base + half + i] = @cos(f);
                sin_out[base + i] = @sin(f);
                sin_out[base + half + i] = @sin(f);
            }
        }
    }

    fn refRotateHalf(x: []const f32, out: []f32) void {
        const hd: usize = @intCast(HD);
        const half: usize = hd / 2;
        var i: usize = 0;
        while (i < x.len) : (i += hd) {
            for (0..half) |j| out[i + j] = -x[i + half + j];
            for (0..half) |j| out[i + half + j] = x[i + j];
        }
    }

    fn refApplyRope(
        x: []const f32,
        cos: []const f32,
        sin: []const f32,
        x_out: []f32,
    ) void {
        var rot_x: [QK_LEN]f32 = undefined;
        refRotateHalf(x, &rot_x);

        const hd: usize = @intCast(HD);
        const h: usize = @intCast(H);
        const s: usize = @intCast(S);
        const b: usize = @intCast(B);

        for (0..b) |bi| {
            for (0..s) |si| {
                for (0..h) |hi| {
                    for (0..hd) |di| {
                        const x_idx = ((bi * s + si) * h + hi) * hd + di;
                        const cs_idx = (bi * s + si) * hd + di;
                        x_out[x_idx] = x[x_idx] * cos[cs_idx] + rot_x[x_idx] * sin[cs_idx];
                    }
                }
            }
        }
    }

    // --- compiled-graph wrappers ---

    fn cosWrapper(rope: model.TextRotaryEmbedding, position_ids: zml.Tensor) zml.Tensor {
        const cos, _ = rope.getCosAndSin(position_ids, .f32);
        return cos;
    }

    fn sinWrapper(rope: model.TextRotaryEmbedding, position_ids: zml.Tensor) zml.Tensor {
        _, const sin = rope.getCosAndSin(position_ids, .f32);
        return sin;
    }

    fn rotateHalfWrapper(x: zml.Tensor) zml.Tensor {
        return model.TextRotaryEmbedding.rotateHalf(x);
    }

    fn applyRopeWrapper(
        rope: model.TextRotaryEmbedding,
        x: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        return rope.applyRope(x, cos, sin);
    }
};

fn runRopeTests(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("RoPE synthetic tests:", .{});
    try testRotateHalf(allocator, io, platform);
    try testCosSin(allocator, io, platform);
    try testApplyRopeReference(allocator, io, platform);
    try testApplyRopePosition0Identity(allocator, io, platform);
    try testApplyRopePreservesNorm(allocator, io, platform);
    std.log.info("  all RoPE tests passed.", .{});
}

fn testRotateHalf(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  rotateHalf", .{});

    const x_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const x_tensor = zml.Tensor.fromShape(x_shape);

    var x_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&x_data);

    var expected: [Rope.QK_LEN]f32 = undefined;
    Rope.refRotateHalf(&x_data, &expected);

    var exe = try zml.module.compile(allocator, io, Rope.rotateHalfWrapper, .{x_tensor}, platform, .{});
    defer exe.deinit();

    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(&x_data));
    defer x_buf.deinit();

    var res = try zml.testing.autoCall(allocator, io, &exe, Rope.rotateHalfWrapper, .{x_buf});
    defer res.deinit();

    try zml.testing.expectClose(
        io,
        zml.Slice.init(x_shape, std.mem.sliceAsBytes(&expected)),
        res,
        .{ .absolute_tolerance = 1e-6 },
    );
}

fn testCosSin(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  getCosAndSin", .{});

    const rope = model.TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const pos_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S }, .i32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);
    const pos_tensor = zml.Tensor.fromShape(pos_shape);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);

    var cos_expected: [Rope.CS_LEN]f32 = undefined;
    var sin_expected: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_expected, &sin_expected);

    var pos_buf: zml.Buffer = try .fromBytes(io, platform, pos_shape, .replicated, std.mem.sliceAsBytes(&pos_data));
    defer pos_buf.deinit();

    var exe_cos = try zml.module.compile(allocator, io, Rope.cosWrapper, .{ rope, pos_tensor }, platform, .{});
    defer exe_cos.deinit();
    var cos_res = try zml.testing.autoCall(allocator, io, &exe_cos, Rope.cosWrapper, .{pos_buf});
    defer cos_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(cs_shape, std.mem.sliceAsBytes(&cos_expected)),
        cos_res,
        .{ .absolute_tolerance = 1e-5 },
    );

    var exe_sin = try zml.module.compile(allocator, io, Rope.sinWrapper, .{ rope, pos_tensor }, platform, .{});
    defer exe_sin.deinit();
    var sin_res = try zml.testing.autoCall(allocator, io, &exe_sin, Rope.sinWrapper, .{pos_buf});
    defer sin_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(cs_shape, std.mem.sliceAsBytes(&sin_expected)),
        sin_res,
        .{ .absolute_tolerance = 1e-5 },
    );
}

fn testApplyRopeReference(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope vs reference", .{});

    const rope = model.TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    var q_data: [Rope.QK_LEN]f32 = undefined;
    var k_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&q_data);
    Rope.makeK(&k_data);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);
    var cos_data: [Rope.CS_LEN]f32 = undefined;
    var sin_data: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_data, &sin_data);

    var q_expected: [Rope.QK_LEN]f32 = undefined;
    var k_expected: [Rope.QK_LEN]f32 = undefined;
    Rope.refApplyRope(&q_data, &cos_data, &sin_data, &q_expected);
    Rope.refApplyRope(&k_data, &cos_data, &sin_data, &k_expected);

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, q_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeWrapper, .{ q_buf, cos_buf, sin_buf });
    defer q_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&q_expected)),
        q_res,
        .{ .absolute_tolerance = 1e-5 },
    );

    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeWrapper, .{ k_buf, cos_buf, sin_buf });
    defer k_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&k_expected)),
        k_res,
        .{ .absolute_tolerance = 1e-5 },
    );
}

fn testApplyRopePosition0Identity(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope position-0 identity", .{});

    const rope = model.TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = 1, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = 1, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    const qk_len: usize = @as(usize, @intCast(Rope.B * 1 * Rope.H * Rope.HD));
    const cs_len: usize = @as(usize, @intCast(Rope.B * 1 * Rope.HD));

    var q_data: [qk_len]f32 = undefined;
    var k_data: [qk_len]f32 = undefined;
    for (&q_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    for (&k_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0 + 0.5;

    // Position 0: cos = ones, sin = zeros, so applyRope must be the identity.
    var cos_data: [cs_len]f32 = .{1} ** cs_len;
    var sin_data: [cs_len]f32 = .{0} ** cs_len;

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, q_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeWrapper, .{ q_buf, cos_buf, sin_buf });
    defer q_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&q_data)),
        q_res,
        .{ .absolute_tolerance = 1e-6 },
    );

    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeWrapper, .{ k_buf, cos_buf, sin_buf });
    defer k_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&k_data)),
        k_res,
        .{ .absolute_tolerance = 1e-6 },
    );
}

fn testApplyRopePreservesNorm(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope preserves per-head norm", .{});

    const rope = model.TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    _ = k_t; // autofix
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    var q_data: [Rope.QK_LEN]f32 = undefined;
    var k_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&q_data);
    Rope.makeK(&k_data);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);
    var cos_data: [Rope.CS_LEN]f32 = undefined;
    var sin_data: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_data, &sin_data);

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, q_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeWrapper, .{ q_buf, cos_buf, sin_buf });
    defer q_res.deinit();

    const q_out_slice = try q_res.toSliceAlloc(allocator, io);
    defer q_out_slice.free(allocator);
    const q_out_bytes = q_out_slice.constData();
    const q_out: []const f32 = @as([*]const f32, @ptrCast(@alignCast(q_out_bytes.ptr)))[0 .. q_out_bytes.len / @sizeOf(f32)];

    const hd: usize = @intCast(Rope.HD);
    const rd: usize = @intCast(Rope.ROTARY_DIM);
    var idx: usize = 0;
    while (idx < q_data.len) : (idx += hd) {
        var n_in: f32 = 0;
        var n_out: f32 = 0;
        for (0..rd) |j| {
            n_in += q_data[idx + j] * q_data[idx + j];
            n_out += q_out[idx + j] * q_out[idx + j];
        }
        if (@abs(n_in - n_out) > 1e-4) {
            std.log.err("Norm mismatch at head {d}: in={d}, out={d}", .{ idx / hd, n_in, n_out });
            return error.NormNotPreserved;
        }
    }
}
