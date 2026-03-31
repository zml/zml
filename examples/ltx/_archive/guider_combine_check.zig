/// Guider combine parity checker: Stage 1 guidance formula.
///
/// Verifies that forwardGuiderCombine produces outputs matching
/// the Python reference for the CFG+STG+modality combine formula.
///
/// Usage: guider_combine_check <fixture.safetensors>
///
/// The fixture must contain:
///   - cond_v, neg_v, ptb_v, iso_v (bf16 inputs, video)
///   - cond_a, neg_a, ptb_a, iso_a (bf16 inputs, audio)
///   - cfg_v, stg_v, mod_v, rescale_v (f32 scalar params)
///   - cfg_a, stg_a, mod_a, rescale_a (f32 scalar params)
///   - guided_v, guided_a (bf16 reference outputs)

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    return check_utils.loadBufferFromStore(allocator, io, platform, store, key, sharding) catch |err| {
        std.log.err("Fixture missing key: {s}", .{key});
        return err;
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Guider combine parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const fixture_path = it.next() orelse {
        std.log.err("Usage: guider_combine_check <fixture.safetensors>", .{});
        return error.InvalidArgs;
    };

    var fix_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fix_reg.deinit();
    var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_reg);
    defer fix_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ── Load inputs ───────────────────────────────────────────────────────
    var cond_v_buf = try loadBuf(allocator, io, platform, &fix_store, "cond_v", sharding);
    defer cond_v_buf.deinit();
    var neg_v_buf = try loadBuf(allocator, io, platform, &fix_store, "neg_v", sharding);
    defer neg_v_buf.deinit();
    var ptb_v_buf = try loadBuf(allocator, io, platform, &fix_store, "ptb_v", sharding);
    defer ptb_v_buf.deinit();
    var iso_v_buf = try loadBuf(allocator, io, platform, &fix_store, "iso_v", sharding);
    defer iso_v_buf.deinit();

    var cond_a_buf = try loadBuf(allocator, io, platform, &fix_store, "cond_a", sharding);
    defer cond_a_buf.deinit();
    var neg_a_buf = try loadBuf(allocator, io, platform, &fix_store, "neg_a", sharding);
    defer neg_a_buf.deinit();
    var ptb_a_buf = try loadBuf(allocator, io, platform, &fix_store, "ptb_a", sharding);
    defer ptb_a_buf.deinit();
    var iso_a_buf = try loadBuf(allocator, io, platform, &fix_store, "iso_a", sharding);
    defer iso_a_buf.deinit();

    // ── Load scalar guidance params ───────────────────────────────────────
    var cfg_v_buf = try loadBuf(allocator, io, platform, &fix_store, "cfg_v", sharding);
    defer cfg_v_buf.deinit();
    var stg_v_buf = try loadBuf(allocator, io, platform, &fix_store, "stg_v", sharding);
    defer stg_v_buf.deinit();
    var mod_v_buf = try loadBuf(allocator, io, platform, &fix_store, "mod_v", sharding);
    defer mod_v_buf.deinit();
    var rescale_v_buf = try loadBuf(allocator, io, platform, &fix_store, "rescale_v", sharding);
    defer rescale_v_buf.deinit();

    var cfg_a_buf = try loadBuf(allocator, io, platform, &fix_store, "cfg_a", sharding);
    defer cfg_a_buf.deinit();
    var stg_a_buf = try loadBuf(allocator, io, platform, &fix_store, "stg_a", sharding);
    defer stg_a_buf.deinit();
    var mod_a_buf = try loadBuf(allocator, io, platform, &fix_store, "mod_a", sharding);
    defer mod_a_buf.deinit();
    var rescale_a_buf = try loadBuf(allocator, io, platform, &fix_store, "rescale_a", sharding);
    defer rescale_a_buf.deinit();

    // ── Load reference outputs ────────────────────────────────────────────
    var guided_v_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "guided_v", sharding);
    defer guided_v_ref_buf.deinit();
    var guided_a_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "guided_a", sharding);
    defer guided_a_ref_buf.deinit();

    // ── Compile forwardGuiderCombine ──────────────────────────────────────
    std.log.info("Compiling guider combine graph...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        model.forwardGuiderCombine,
        .{
            zml.Tensor.fromShape(cond_v_buf.shape()),
            zml.Tensor.fromShape(neg_v_buf.shape()),
            zml.Tensor.fromShape(ptb_v_buf.shape()),
            zml.Tensor.fromShape(iso_v_buf.shape()),
            zml.Tensor.fromShape(cond_a_buf.shape()),
            zml.Tensor.fromShape(neg_a_buf.shape()),
            zml.Tensor.fromShape(ptb_a_buf.shape()),
            zml.Tensor.fromShape(iso_a_buf.shape()),
            zml.Tensor.fromShape(cfg_v_buf.shape()),
            zml.Tensor.fromShape(stg_v_buf.shape()),
            zml.Tensor.fromShape(mod_v_buf.shape()),
            zml.Tensor.fromShape(rescale_v_buf.shape()),
            zml.Tensor.fromShape(cfg_a_buf.shape()),
            zml.Tensor.fromShape(stg_a_buf.shape()),
            zml.Tensor.fromShape(mod_a_buf.shape()),
            zml.Tensor.fromShape(rescale_a_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    args.set(.{
        cond_v_buf, neg_v_buf, ptb_v_buf, iso_v_buf,
        cond_a_buf, neg_a_buf, ptb_a_buf, iso_a_buf,
        cfg_v_buf, stg_v_buf, mod_v_buf, rescale_v_buf,
        cfg_a_buf, stg_a_buf, mod_a_buf, rescale_a_buf,
    });

    std.log.info("Running guider combine...", .{});
    exe.call(args, &res);

    const out = res.get(zml.Bufferized(model.GuiderCombineResult));

    // ── Compare against reference ─────────────────────────────────────────
    std.log.info("Comparing Zig vs Python reference...", .{});
    const v_stats = try check_utils.compareBuffersExtended(io, out.guided_v, guided_v_ref_buf, 1e-3, 1e-2);
    std.log.info("  Video: cos_sim={d:.6} close={d:.6} max_abs={d:.4} mean_abs={d:.6}", .{
        v_stats.cosine_similarity, v_stats.close_fraction, v_stats.max_abs_error, v_stats.mean_abs_error,
    });

    const a_stats = try check_utils.compareBuffersExtended(io, out.guided_a, guided_a_ref_buf, 1e-3, 1e-2);
    std.log.info("  Audio: cos_sim={d:.6} close={d:.6} max_abs={d:.4} mean_abs={d:.6}", .{
        a_stats.cosine_similarity, a_stats.close_fraction, a_stats.max_abs_error, a_stats.mean_abs_error,
    });

    // ── Verdict ───────────────────────────────────────────────────────────
    const pass_cos = 0.99999;
    const pass_close = 0.999;
    const v_pass = v_stats.cosine_similarity >= pass_cos and v_stats.close_fraction >= pass_close;
    const a_pass = a_stats.cosine_similarity >= pass_cos and a_stats.close_fraction >= pass_close;

    if (v_pass and a_pass) {
        std.log.info("PASSED: guider combine matches Python reference.", .{});
    } else {
        if (!v_pass) std.log.err("FAILED: Video cos_sim={d:.6} close={d:.6}", .{ v_stats.cosine_similarity, v_stats.close_fraction });
        if (!a_pass) std.log.err("FAILED: Audio cos_sim={d:.6} close={d:.6}", .{ a_stats.cosine_similarity, a_stats.close_fraction });
        return error.ParityCheckFailed;
    }
}
