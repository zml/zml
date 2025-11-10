const std = @import("std");
const async = @import("async");
const zml = @import("zml");

const Reference = struct {
    input: []const []const f32,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
    torch: struct {
        layernorm: []const []const f32,
        rmsnorm: []const []const f32,
        rmsnorm_scaled: []const []const f32,
    },
    flax: struct {
        layernorm: []const []const f32,
        rmsnorm: []const []const f32,
    },
};

var g_eps_layernorm: f32 = 0;
var g_eps_rms: f32 = 0;

const Compare = struct {
    pub fn layerNorm(input: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor) zml.Tensor {
        const ln = zml.nn.LayerNorm{
            .weight = weight.withTags(.{.d}),
            .bias = bias.withTags(.{.d}),
            .eps = g_eps_layernorm,
        };
        return ln.forward(input);
    }

    pub fn rmsNorm(input: zml.Tensor) zml.Tensor {
        return zml.nn.rmsNorm(input, .d, g_eps_rms);
    }

    pub fn rmsNormScaled(input: zml.Tensor, weight: zml.Tensor) zml.Tensor {
        return zml.nn.rmsNorm(input, .d, g_eps_rms).mul(weight.withTags(.{.d}).broad(input.shape()));
    }
};

fn flatten2D(allocator: std.mem.Allocator, matrix: []const []const f32) ![]f32 {
    if (matrix.len == 0) return &[_]f32{};
    const cols = matrix[0].len;
    var buf = try allocator.alloc(f32, matrix.len * cols);
    for (matrix, 0..) |row, i| {
        std.debug.assert(row.len == cols);
        std.mem.copyForwards(f32, buf[i * cols .. i * cols + cols], row);
    }
    return buf;
}

fn maxDiff(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: f32 = 0.0;
    for (a, 0..) |v, i| {
        const diff = @abs(v - b[i]);
        if (diff > acc) acc = diff;
    }
    return acc;
}

fn readRefFile(allocator: std.mem.Allocator, rel_path: []const u8) ![]u8 {
    const max_size = std.math.maxInt(usize);
    const cwd = std.fs.cwd();
    return cwd.readFileAlloc(allocator, rel_path, max_size) catch |err| switch (err) {
        error.FileNotFound => blk: {
            const joined = try std.fs.path.join(allocator, &.{ "__main__", rel_path });
            defer allocator.free(joined);
            const local = cwd.readFileAlloc(allocator, joined, max_size) catch |err2| switch (err2) {
                error.FileNotFound => blk2: {
                    const env_paths = [_][]const u8{ "RUNFILES_DIR", "JAVA_RUNFILES" };
                    for (env_paths) |env_name| {
                        const runfiles_dir_opt = std.process.getEnvVarOwned(allocator, env_name) catch null;
                        if (runfiles_dir_opt) |runfiles_dir| {
                            defer allocator.free(runfiles_dir);
                            const full = try std.fs.path.join(allocator, &.{ runfiles_dir, "__main__", rel_path });
                            defer allocator.free(full);
                            var file = try std.fs.openFileAbsolute(full, .{ .mode = .read_only });
                            defer file.close();
                            break :blk2 try file.readToEndAlloc(allocator, max_size);
                        }
                    }
                    break :blk2 err2;
                },
                else => break :blk err2,
            };
            break :blk local;
        },
        else => return err,
    };
}

pub fn main() !void {
    try async.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const refs_path = "examples/qwen_3_vl/tools/layernorm_refs.json";
    const contents = try readRefFile(alloc, refs_path);
    defer alloc.free(contents);

    var parsed = try std.json.parseFromSlice(Reference, alloc, contents, .{});
    defer parsed.deinit();
    const data = parsed.value;

    std.debug.assert(data.input.len > 0);
    const bs = data.input.len;
    const features = data.input[0].len;

    const input_flat = try flatten2D(alloc, data.input);
    defer alloc.free(input_flat);

    var context = try zml.Context.init();
    defer context.deinit();
    const platform = context.autoPlatform(.{});

    const bs_i64 = @as(i64, @intCast(bs));
    const features_i64 = @as(i64, @intCast(features));

    const input_buf = try zml.Buffer.fromSlice(platform, .{ .bs = bs_i64, .d = features_i64 }, input_flat);
    defer input_buf.deinit();
    const weight_buf = try zml.Buffer.fromSlice(platform, .{ .d = features_i64 }, data.weight);
    defer weight_buf.deinit();
    const bias_buf = try zml.Buffer.fromSlice(platform, .{ .d = features_i64 }, data.bias);
    defer bias_buf.deinit();

    const input_shape = zml.Shape.init(.{ .bs = bs_i64, .d = features_i64 }, .f32);
    const vec_shape = zml.Shape.init(.{ .d = features_i64 }, .f32);

    g_eps_layernorm = data.eps;
    g_eps_rms = data.eps;

    const ln_exe = try zml.compileFn(alloc, Compare.layerNorm, .{ input_shape, vec_shape, vec_shape }, platform);
    defer ln_exe.deinit();
    const ln_buf = ln_exe.call(.{ input_buf, weight_buf, bias_buf });
    defer ln_buf.deinit();

    const rms_exe = try zml.compileFn(alloc, Compare.rmsNorm, .{input_shape}, platform);
    defer rms_exe.deinit();
    const rms_buf = rms_exe.call(.{input_buf});
    defer rms_buf.deinit();

    const rms_scaled_exe = try zml.compileFn(alloc, Compare.rmsNormScaled, .{ input_shape, vec_shape }, platform);
    defer rms_scaled_exe.deinit();
    const rms_scaled_buf = rms_scaled_exe.call(.{ input_buf, weight_buf });
    defer rms_scaled_buf.deinit();

    const ln_host = try ln_buf.toHostAlloc(alloc);
    defer ln_host.deinit(alloc);
    const rms_host = try rms_buf.toHostAlloc(alloc);
    defer rms_host.deinit(alloc);
    const rms_scaled_host = try rms_scaled_buf.toHostAlloc(alloc);
    defer rms_scaled_host.deinit(alloc);

    const ln_slice = ln_host.items(f32);
    const rms_slice = rms_host.items(f32);
    const rms_scaled_slice = rms_scaled_host.items(f32);

    const torch_ln_flat = try flatten2D(alloc, data.torch.layernorm);
    defer alloc.free(torch_ln_flat);
    const torch_rms_flat = try flatten2D(alloc, data.torch.rmsnorm);
    defer alloc.free(torch_rms_flat);
    const torch_rms_scaled_flat = try flatten2D(alloc, data.torch.rmsnorm_scaled);
    defer alloc.free(torch_rms_scaled_flat);

    const flax_ln_flat = try flatten2D(alloc, data.flax.layernorm);
    defer alloc.free(flax_ln_flat);
    const flax_rms_flat = try flatten2D(alloc, data.flax.rmsnorm);
    defer alloc.free(flax_rms_flat);

    std.debug.print("LayerNorm max diff (Torch): {d:.6}\n", .{maxDiff(ln_slice, torch_ln_flat)});
    std.debug.print("LayerNorm max diff (Flax):  {d:.6}\n", .{maxDiff(ln_slice, flax_ln_flat)});
    std.debug.print("RMSNorm max diff (Torch):   {d:.6}\n", .{maxDiff(rms_slice, torch_rms_flat)});
    std.debug.print("RMSNorm max diff (Flax):    {d:.6}\n", .{maxDiff(rms_slice, flax_rms_flat)});
    std.debug.print("RMSNorm·scale max diff:     {d:.6}\n", .{maxDiff(rms_scaled_slice, torch_rms_scaled_flat)});
}
