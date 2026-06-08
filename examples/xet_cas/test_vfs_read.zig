// Step 4 acceptance driver: read an XET-backed file through the HF VFS
// (`hf://...`) and SHA-256 the bytes; compare against an LFS-oracle SHA-256
// of the same byte range fetched directly via the resolve URL. Exercises
// `Handle.XetState`, `xetFetch`, the chunk ring, and the plan cache.
//
// Usage:
//   HF_TOKEN=... bazel run //examples/xet_cas:test_vfs_read -- \
//     --model hf://meta-llama/Meta-Llama-3-70B \
//     --file model-00030-of-00030.safetensors \
//     [--offset 0] [--bytes 16777216]

const std = @import("std");
const zml = @import("zml");
const xet = @import("io").xet;
const util = @import("util.zig");

const log = std.log.scoped(.test_vfs_read);

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var file_arg: []const u8 = "";
    var offset: u64 = 0;
    var bytes: ?u64 = null;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--file")) {
            file_arg = args_it.next() orelse return error.MissingFileArg;
        } else if (std.mem.eql(u8, arg, "--offset")) {
            offset = try std.fmt.parseInt(u64, args_it.next() orelse return error.MissingOffsetArg, 10);
        } else if (std.mem.eql(u8, arg, "--bytes")) {
            bytes = try std.fmt.parseInt(u64, args_it.next() orelse return error.MissingBytesArg, 10);
        }
    }
    if (model_arg.len == 0 or file_arg.len == 0) {
        std.debug.print("Usage: test_vfs_read --model <uri> --file <repo-relative-path> [--offset N] [--bytes N]\n", .{});
        std.process.exit(1);
    }

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();
    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();
    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();
    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    const io = vfs.io();

    // Build the file URI directly from --model + --file (skip the registry
    // walk that would open every shard in the repo and trigger an XET probe
    // per file). `--model` may be `hf://repo/model` or `repo/model`.
    const model_path = if (std.mem.startsWith(u8, model_arg, "hf://")) model_arg["hf://".len..] else model_arg;
    var uri_buf: [4096]u8 = undefined;
    const file_uri = try std.fmt.bufPrint(&uri_buf, "hf://{s}/{s}", .{ model_path, file_arg });
    log.info("File: {s}", .{file_uri});

    // ── Open through the VFS ──────────────────────────────────────────────
    const vfile = try std.Io.Dir.openFile(.cwd(), io, file_uri, .{ .mode = .read_only });
    defer vfile.close(io);
    const stat = try vfile.stat(io);
    const file_size = stat.size;
    log.info("file_size={d}", .{file_size});

    if (offset >= file_size) {
        log.err("offset {d} >= file_size {d}", .{ offset, file_size });
        return error.OffsetOutOfRange;
    }
    const want: u64 = blk: {
        const tail = file_size - offset;
        break :blk if (bytes) |b| @min(b, tail) else tail;
    };
    log.info("range: offset={d} bytes={d}", .{ offset, want });

    // ── VFS read → SHA-256 ────────────────────────────────────────────────
    const buf = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(buf);
    var hasher_vfs = std.crypto.hash.sha2.Sha256.init(.{});
    var vfs_reader = vfile.reader(io, &.{});
    try vfs_reader.seekTo(offset);
    var remaining = want;
    const bytes_fetched_before = hf_vfs.xet_state.bytes_fetched.load(.monotonic);
    const ts_vfs: std.Io.Timestamp = .now(init.io, .awake);
    while (remaining > 0) {
        const take: usize = @intCast(@min(@as(u64, buf.len), remaining));
        const n = try vfs_reader.interface.readSliceShort(buf[0..take]);
        if (n == 0) break;
        hasher_vfs.update(buf[0..n]);
        remaining -= n;
    }
    const vfs_ns: u64 = @intCast(ts_vfs.untilNow(init.io, .awake).toNanoseconds());
    const bytes_fetched_delta = hf_vfs.xet_state.bytes_fetched.load(.monotonic) - bytes_fetched_before;
    var digest_vfs: [32]u8 = undefined;
    hasher_vfs.final(&digest_vfs);

    // ── LFS oracle → SHA-256 ──────────────────────────────────────────────
    const hf_path = if (std.mem.startsWith(u8, file_uri, "hf://")) file_uri["hf://".len..] else file_uri;
    const hf_repo = try zml.io.VFS.HF.Repo.parse(hf_path);
    const repo: xet.State.Repo = .{ .repo = hf_repo.repo, .model = hf_repo.model, .rev = hf_repo.rev, .path = hf_repo.path };
    const hf_token = try loadHfToken(allocator, init.io, init.environ_map);
    defer allocator.free(hf_token);
    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;
    var oracle_net_ns: u64 = 0;
    const digest_oracle = try util.sha256LfsRange(&http_client, repo, auth, offset, offset + want - 1, init.io, &oracle_net_ns);

    log.info("vfs    sha256={x} time_ms={d}", .{ digest_vfs, vfs_ns / std.time.ns_per_ms });
    log.info("vfs    bytes_fetched={d} ratio={d:.3}", .{ bytes_fetched_delta, @as(f64, @floatFromInt(bytes_fetched_delta)) / @as(f64, @floatFromInt(want)) });
    log.info("oracle sha256={x} net_ms={d}", .{ digest_oracle, oracle_net_ns / std.time.ns_per_ms });

    if (!std.mem.eql(u8, &digest_vfs, &digest_oracle)) {
        log.err("SHA-256 MISMATCH", .{});
        return error.Sha256Mismatch;
    }
    log.info("OK — SHA-256 match over {d} bytes at offset {d}", .{ want, offset });
}

fn loadHfToken(allocator: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map) ![]u8 {
    if (env.get("HF_TOKEN")) |t| return allocator.dupe(u8, t);
    const home = env.get("HOME") orelse return error.MissingToken;
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/.cache/huggingface/token", .{home}) catch return error.MissingToken;
    var file = std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only }) catch return error.MissingToken;
    defer file.close(io);
    const size = file.stat(io) catch return error.MissingToken;
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(allocator, size.size) catch error.MissingToken;
}
