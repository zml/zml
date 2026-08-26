const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const buffers = @import("core/buffers.zig");
const sharding = @import("core/sharding.zig");
const vision = @import("model/vision.zig");

const log = std.log.scoped(.minimax_h3_vision_probe);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    dump: []const u8,

    pub const help =
        \\ Use vision_probe --dump=<dir>
        \\
        \\ Compiles visionAttn only (no DiT / encoder / VAE). Compares
        \\ vision_q, vision_k, vision_v against vision_attn from
        \\ repro_vision.py --dump-attn.
        \\
    ;
};

const Dump = struct {
    values: []f32,
    seq: u32,
    heads: u32,
    head_dim: u32,

    fn deinit(self: Dump, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }
};

fn openDumpDir(io: std.Io, path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path))
        return std.Io.Dir.openDirAbsolute(io, path, .{});
    return std.Io.Dir.cwd().openDir(io, path, .{});
}

fn parseDims(text: []const u8) ![4]u32 {
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, text, " \n\r\t"), ' ');
    var dims: [4]u32 = undefined;
    var n: usize = 0;
    while (it.next()) |part| {
        if (n == dims.len) return error.BadShape;
        dims[n] = std.fmt.parseInt(u32, part, 10) catch return error.BadShape;
        n += 1;
    }
    if (n != 4) return error.BadShape;
    return dims;
}

fn loadNamed(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !Dump {
    var shape_name_buf: [64]u8 = undefined;
    const shape_name = try std.fmt.bufPrint(&shape_name_buf, "{s}.shape", .{name});
    const shape_text = try dir.readFileAlloc(io, shape_name, allocator, .limited(128));
    defer allocator.free(shape_text);
    const dims = try parseDims(shape_text);
    if (dims[0] != 1) return error.BadShape;

    var file_name_buf: [64]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&file_name_buf, "{s}.f32", .{name});
    const bytes = try dir.readFileAlloc(io, file_name, allocator, .unlimited);
    defer allocator.free(bytes);
    if (bytes.len % 4 != 0) return error.BadDump;
    const want = @as(usize, dims[0]) * dims[1] * dims[2] * dims[3];
    if (bytes.len / 4 != want) return error.BadDump;
    const values = try allocator.alloc(f32, want);
    errdefer allocator.free(values);
    @memcpy(std.mem.sliceAsBytes(values), bytes);
    return .{
        .values = values,
        .seq = dims[1],
        .heads = dims[2],
        .head_dim = dims[3],
    };
}

fn report(name: []const u8, got: []const f32, ref: []const f32) void {
    var nan: u32 = 0;
    var inf: u32 = 0;
    var max_abs: f32 = 0;
    var dot: f64 = 0;
    var n_a: f64 = 0;
    var n_b: f64 = 0;
    const n = @min(got.len, ref.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const a = got[i];
        const b = ref[i];
        if (!std.math.isFinite(a)) {
            if (std.math.isNan(a)) nan += 1 else inf += 1;
            continue;
        }
        const d = @abs(a - b);
        if (d > max_abs) max_abs = d;
        const af: f64 = a;
        const bf: f64 = b;
        dot += af * bf;
        n_a += af * af;
        n_b += bf * bf;
    }
    const cosine = if (n_a > 0 and n_b > 0) dot / (@sqrt(n_a) * @sqrt(n_b)) else 0;
    log.info(
        "{s}: cosine={d:.6} max_abs={d:.4} nan={d} inf={d} n={d}",
        .{ name, cosine, max_abs, nan, inf, n },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();
    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());
    const io = vfs.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{
        .cpu = .{ .device_count = 1 },
        .physical_mesh = .{ .custom = sharding.physicalMesh },
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    try vision.register(platform);
    const shardings: sharding.Shardings = try .init(platform);
    const all = shardings.all();
    log.info("\n{f}", .{platform.fmtVerbose()});

    var dir = try openDumpDir(io, args.dump);
    defer dir.close(io);
    var q_dump = try loadNamed(allocator, io, dir, "vision_q");
    defer q_dump.deinit(allocator);
    var k_dump = try loadNamed(allocator, io, dir, "vision_k");
    defer k_dump.deinit(allocator);
    var v_dump = try loadNamed(allocator, io, dir, "vision_v");
    defer v_dump.deinit(allocator);
    var o_dump = try loadNamed(allocator, io, dir, "vision_attn");
    defer o_dump.deinit(allocator);
    if (k_dump.seq != q_dump.seq or v_dump.seq != q_dump.seq or o_dump.seq != q_dump.seq)
        return error.SeqMismatch;
    if (k_dump.heads != q_dump.heads or v_dump.heads != q_dump.heads or o_dump.heads != q_dump.heads)
        return error.HeadMismatch;
    if (k_dump.head_dim != q_dump.head_dim or v_dump.head_dim != q_dump.head_dim or o_dump.head_dim != q_dump.head_dim)
        return error.HeadDimMismatch;

    const seq = q_dump.seq;
    const heads = q_dump.heads;
    const head_dim = q_dump.head_dim;
    const dt = zml.DataType.bf16;
    log.info("attn probe seq={d} heads={d} hd={d} target={s}", .{
        seq,
        heads,
        head_dim,
        @tagName(platform.target),
    });

    var progress = std.Progress.start(io, .{ .root_name = "vision_probe" });
    defer progress.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    const exe = try zml.FnExe(vision.probeAttn).compile(allocator, io, platform, .{
        .shardings = &all,
        .program_name = "h3_vision_attn_probe",
    }, .{.{
        .q = .init(.{ .b = 1, .q = seq, .h = heads, .hd = head_dim }, dt),
        .k = .init(.{ .b = 1, .k = seq, .h = heads, .hd = head_dim }, dt),
        .v = .init(.{ .b = 1, .k = seq, .h = heads, .hd = head_dim }, dt),
    }});
    defer exe.deinit();
    log.info("compile h3_vision_attn_probe: ok [{f}]", .{now.untilNow(io, .awake)});

    const q_shape = zml.Shape.init(.{ .b = 1, .q = seq, .h = heads, .hd = head_dim }, dt);
    const k_shape = zml.Shape.init(.{ .b = 1, .k = seq, .h = heads, .hd = head_dim }, dt);
    var q_buf = try buffers.fromF32(allocator, io, platform, q_shape, q_dump.values);
    defer q_buf.deinit();
    var k_buf = try buffers.fromF32(allocator, io, platform, k_shape, k_dump.values);
    defer k_buf.deinit();
    var v_buf = try buffers.fromF32(allocator, io, platform, k_shape, v_dump.values);
    defer v_buf.deinit();

    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var out_buf: zml.Buffer = undefined;
    runner.run(io, .{
        .inputs = .{ .q = q_buf, .k = k_buf, .v = v_buf },
        .outputs = .{ .o = &out_buf },
        .opts = .{ .wait = true },
    });
    defer out_buf.deinit();

    const got = try buffers.toF32(allocator, io, out_buf);
    defer allocator.free(got);
    report("vision_attn", got, o_dump.values);
}
