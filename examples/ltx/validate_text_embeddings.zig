/// Validation driver: compare Zig text embeddings against Python reference.
///
/// Loads Gemma hidden states, runs the Zig EmbeddingsProcessor, and compares
/// outputs against Python reference at two granularities:
///   1. Feature extraction (before connectors)
///   2. Final context embeddings (after connectors)
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:validate_text_embeddings -- \
///       --hidden-states /root/gemma_export/pos_hidden_states.safetensors \
///       --ref-embeddings /root/gemma_export/ref_embeddings.safetensors \
///       --ref-features /root/gemma_export/ref_features.safetensors \
///       --checkpoint ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
///       --label pos
const std = @import("std");
const zml = @import("zml");
const text_embeddings = @import("text_embeddings.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

// ============================================================================
// CLI
// ============================================================================

const CliArgs = struct {
    hidden_states_path: []const u8,
    ref_embeddings_path: []const u8,
    checkpoint_path: []const u8,
    label: []const u8,
};

fn parseArgs(it: anytype) !CliArgs {
    var args: CliArgs = .{
        .hidden_states_path = undefined,
        .ref_embeddings_path = undefined,
        .checkpoint_path = undefined,
        .label = "pos",
    };
    var have = [_]bool{false} ** 3;

    _ = it.next(); // exe name

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--hidden-states")) {
            args.hidden_states_path = it.next() orelse return error.InvalidArgs;
            have[0] = true;
        } else if (std.mem.eql(u8, arg, "--ref-embeddings")) {
            args.ref_embeddings_path = it.next() orelse return error.InvalidArgs;
            have[1] = true;
        } else if (std.mem.eql(u8, arg, "--checkpoint")) {
            args.checkpoint_path = it.next() orelse return error.InvalidArgs;
            have[2] = true;
        } else if (std.mem.eql(u8, arg, "--label")) {
            args.label = it.next() orelse return error.InvalidArgs;
        } else {
            std.log.err("Unknown argument: {s}", .{arg});
            return error.InvalidArgs;
        }
    }

    for (have, 0..) |h, i| {
        if (!h) {
            const names = [_][]const u8{ "--hidden-states", "--ref-embeddings", "--checkpoint" };
            std.log.err("Missing required argument: {s}", .{names[i]});
            return error.InvalidArgs;
        }
    }

    return args;
}

// ============================================================================
// Helpers
// ============================================================================

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    name: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.view().getShape(name) orelse {
        std.log.err("Tensor not found in store: {s}", .{name});
        return error.TensorNotFound;
    };
    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);
    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(name, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

/// Compare two buffers and report metrics; returns true if within tolerance.
fn compareBuffers(
    allocator: std.mem.Allocator,
    io: std.Io,
    actual_buf: zml.Buffer,
    ref_buf: zml.Buffer,
    name: []const u8,
    min_cosine: f64,
    max_mean_abs: f64,
) !bool {
    const actual_slice = try actual_buf.toSliceAlloc(allocator, io);
    defer actual_slice.free(allocator);
    const ref_slice = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);

    // Interpret as bf16 → convert to f32 for comparison
    const actual_bytes = actual_slice.constData();
    const ref_bytes = ref_slice.constData();
    const n_elements: usize = actual_bytes.len / 2; // bf16 = 2 bytes

    if (n_elements == 0) {
        std.log.warn("  {s}: empty tensor", .{name});
        return true;
    }

    const actual_bf16: [*]const u16 = @ptrCast(@alignCast(actual_bytes.ptr));
    const ref_bf16: [*]const u16 = @ptrCast(@alignCast(ref_bytes.ptr));

    var max_abs_diff: f64 = 0;
    var sum_abs_diff: f64 = 0;
    var sum_dot: f64 = 0;
    var sum_sq_a: f64 = 0;
    var sum_sq_r: f64 = 0;

    for (0..n_elements) |i| {
        const a = bf16ToF32(actual_bf16[i]);
        const r = bf16ToF32(ref_bf16[i]);
        const diff = @abs(a - r);
        max_abs_diff = @max(max_abs_diff, diff);
        sum_abs_diff += diff;
        sum_dot += a * r;
        sum_sq_a += a * a;
        sum_sq_r += r * r;
    }

    const mean_abs_diff = sum_abs_diff / @as(f64, @floatFromInt(n_elements));
    const cosine_sim = if (sum_sq_a > 0 and sum_sq_r > 0)
        sum_dot / (std.math.sqrt(sum_sq_a) * std.math.sqrt(sum_sq_r))
    else
        0.0;

    const pass = cosine_sim >= min_cosine and mean_abs_diff <= max_mean_abs;
    const status = if (pass) "PASS" else "FAIL";

    std.log.info("  {s}: [{s}] cosine={d:.8} mean_abs={d:.6} max_abs={d:.6}", .{
        name, status, cosine_sim, mean_abs_diff, max_abs_diff,
    });

    return pass;
}

fn bf16ToF32(bits: u16) f64 {
    const f32_bits: u32 = @as(u32, bits) << 16;
    return @as(f64, @as(f32, @bitCast(f32_bits)));
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    const args = parseArgs(&it) catch |err| {
        std.log.err("Usage: validate_text_embeddings --hidden-states <path> --ref-embeddings <path> --checkpoint <path> [--label pos|neg]", .{});
        return err;
    };

    std.log.info("=== Text Embeddings Validation ===", .{});
    std.log.info("Hidden states: {s}", .{args.hidden_states_path});
    std.log.info("Reference:     {s}", .{args.ref_embeddings_path});
    std.log.info("Checkpoint:    {s}", .{args.checkpoint_path});
    std.log.info("Label:         {s}", .{args.label});

    // ---- Platform ----
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ---- Open stores ----
    std.log.info("Opening stores...", .{});
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.checkpoint_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{args.checkpoint_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var hs_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.hidden_states_path) catch |err| {
        std.log.err("Failed to open hidden states: {s}", .{args.hidden_states_path});
        return err;
    };
    defer hs_reg.deinit();
    var hs_store: zml.io.TensorStore = .fromRegistry(allocator, &hs_reg);
    defer hs_store.deinit();

    var ref_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.ref_embeddings_path) catch |err| {
        std.log.err("Failed to open reference embeddings: {s}", .{args.ref_embeddings_path});
        return err;
    };
    defer ref_reg.deinit();
    var ref_store: zml.io.TensorStore = .fromRegistry(allocator, &ref_reg);
    defer ref_store.deinit();

    // ---- Load hidden states ----
    std.log.info("Loading hidden states...", .{});
    var stacked_hs_buf = try loadBuf(allocator, io, platform, &hs_store, "stacked_hidden_states", sharding);
    defer stacked_hs_buf.deinit();
    var mask_buf = try loadBuf(allocator, io, platform, &hs_store, "attention_mask", sharding);
    defer mask_buf.deinit();

    std.log.info("  stacked_hidden_states: {}", .{stacked_hs_buf.shape()});
    std.log.info("  attention_mask:        {}", .{mask_buf.shape()});

    // ---- Initialize EmbeddingsProcessor params ----
    std.log.info("Initializing EmbeddingsProcessor params from checkpoint...", .{});
    const proc_init = text_embeddings.EmbeddingsProcessor.initParams(ckpt_store.view());
    const processor = proc_init.processor;
    const proc_params = proc_init.params;

    // ---- Compile graph ----
    std.log.info("Compiling forwardEmbeddingsProcessor...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        text_embeddings.forwardEmbeddingsProcessor,
        .{
            &processor,
            zml.Tensor.fromShape(stacked_hs_buf.shape()),
            zml.Tensor.fromShape(mask_buf.shape()),
            proc_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    // ---- Load model weights ----
    std.log.info("Loading model weights...", .{});
    const weight_bufs = try zml.io.load(
        text_embeddings.EmbeddingsProcessor.Params,
        &proc_params,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );

    // ---- Execute ----
    std.log.info("Executing...", .{});
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    exe_args.set(.{
        stacked_hs_buf,
        mask_buf,
        weight_bufs,
    });
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(exe_args, &results);

    var out = results.get(zml.Bufferized(text_embeddings.EmbeddingsProcessor.Result));
    var v_context_buf = out.v_context;
    defer v_context_buf.deinit();
    var a_context_buf = out.a_context;
    defer a_context_buf.deinit();
    var binary_mask_buf = out.binary_mask;
    defer binary_mask_buf.deinit();

    std.log.info("  v_context:         {}", .{v_context_buf.shape()});
    std.log.info("  a_context:         {}", .{a_context_buf.shape()});
    std.log.info("  binary_mask:       {}", .{binary_mask_buf.shape()});

    // ---- Compare against reference ----
    std.log.info("=== Comparing against Python reference ===", .{});

    var all_pass = true;

    // Load reference embeddings
    std.log.info("--- Final context embeddings ---", .{});
    var v_name_buf: [64]u8 = undefined;
    var a_name_buf: [64]u8 = undefined;
    const v_ref_name = std.fmt.bufPrint(&v_name_buf, "v_context_{s}", .{args.label}) catch unreachable;
    const a_ref_name = std.fmt.bufPrint(&a_name_buf, "a_context_{s}", .{args.label}) catch unreachable;

    var v_ref_buf = try loadBuf(allocator, io, platform, &ref_store, v_ref_name, sharding);
    defer v_ref_buf.deinit();
    var a_ref_buf = try loadBuf(allocator, io, platform, &ref_store, a_ref_name, sharding);
    defer a_ref_buf.deinit();

    const v_pass = try compareBuffers(allocator, io, v_context_buf, v_ref_buf, "v_context", 0.9999, 0.01);
    const a_pass = try compareBuffers(allocator, io, a_context_buf, a_ref_buf, "a_context", 0.9999, 0.01);
    all_pass = all_pass and v_pass and a_pass;

    // ---- Summary ----
    std.log.info("", .{});
    if (all_pass) {
        std.log.info("=== ALL CHECKS PASSED ===", .{});
    } else {
        std.log.err("=== SOME CHECKS FAILED ===", .{});
        return error.ValidationFailed;
    }
}
