/// Sigma schedule parity checker: LTX2Scheduler.execute(steps=30).
///
/// Computes the Stage 1 sigma schedule in Zig and compares against
/// the Python reference values. No GPU needed — pure host-side math.
///
/// Usage: sigma_schedule_check

const std = @import("std");
const model = @import("model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    _ = init;

    const defaults = model.stage1_default_schedule;
    const sigmas = model.computeSigmaSchedule(
        64, // max_steps (comptime capacity)
        defaults.num_steps,
        defaults.default_num_tokens,
        defaults.max_shift,
        defaults.base_shift,
        defaults.terminal,
    );

    // Python reference from LTX2Scheduler().execute(steps=30)
    const python_ref = [31]f32{
        0.9999999404, 0.9949570298, 0.9896030426, 0.9839084148,
        0.9778395891, 0.9713582397, 0.9644212127, 0.9569781423,
        0.9489720464, 0.9403364062, 0.9309937954, 0.9208537936,
        0.9098098278, 0.8977352381, 0.8844786882, 0.8698580265,
        0.8536507487, 0.8355841041, 0.8153185844, 0.7924267054,
        0.7663628459, 0.7364180088, 0.7016562223, 0.6608133316,
        0.6121406555, 0.5531478524, 0.4801636934, 0.3875408173,
        0.2661204338, 0.1000000238, 0.0000000000,
    };

    std.log.info("Sigma schedule comparison (Zig vs Python):", .{});
    std.log.info("{s:>4}  {s:>14}  {s:>14}  {s:>14}", .{ "idx", "zig", "python", "abs_diff" });

    var max_diff: f32 = 0.0;
    var total_diff: f64 = 0.0;
    for (0..defaults.num_steps + 1) |i| {
        const diff = @abs(sigmas[i] - python_ref[i]);
        if (diff > max_diff) max_diff = diff;
        total_diff += diff;
        std.log.info("[{d:2}]  {d:14.10}  {d:14.10}  {d:14.10}", .{
            i, sigmas[i], python_ref[i], diff,
        });
    }

    const mean_diff = total_diff / @as(f64, @floatFromInt(defaults.num_steps + 1));
    std.log.info("", .{});
    std.log.info("max_abs_diff:  {d:.10}", .{max_diff});
    std.log.info("mean_abs_diff: {d:.10}", .{mean_diff});

    // f32 epsilon is ~1.19e-7; we allow some tolerance for exp() precision
    const tolerance: f32 = 1e-6;
    if (max_diff < tolerance) {
        std.log.info("PASSED: Sigma schedule matches Python reference (max_diff < {d})", .{tolerance});
    } else {
        std.log.err("FAILED: max_diff={d:.10} exceeds tolerance {d}", .{ max_diff, tolerance });
        return error.ParityCheckFailed;
    }
}
