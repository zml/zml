// Scan every tensor of an HF model repo for intra-tensor chunk dedup.
//
// For each tensor in the safetensors registry, asks the HF CAS for the
// reconstruction of [offset, offset+byteSize), then counts how many plans
// reuse the same (xorb_hash, chunk_start, chunk_end) at a different device
// offset. Prints a line per tensor and a summary at the end.
//
// Usage:
//   scan_dedup --model hf://meta-llama/Llama-3.3-70B-Instruct-FP8
//   scan_dedup --model hf://... --min-size 1048576    # skip tiny tensors

const std = @import("std");
const zml = @import("zml");
const xet = @import("io").xet;

const log = std.log.scoped(.scan_dedup);

pub const std_options: std.Options = .{ .log_level = .info };

const MAX_XORBS: usize = 256;
const MAX_PLANS: usize = 8192;

const XorbKey = struct {
    hash: []const u8,
};

const TermPlan = struct {
    xorb_idx: u16,
    dst_off: u64,
    chunk_start: u32,
    chunk_end: u32,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var min_size: u64 = 0;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--min-size")) {
            const v = args_it.next() orelse return error.MissingMinSizeArg;
            min_size = try std.fmt.parseInt(u64, v, 10);
        }
    }
    if (model_arg.len == 0) {
        std.debug.print("Usage: scan_dedup --model <uri> [--min-size <bytes>]\n", .{});
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

    log.info("Resolving model repo: {s}", .{model_arg});
    const repo_dir = try zml.safetensors.resolveModelRepo(io, model_arg);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo_dir);
    defer registry.deinit();

    const hf_token = init.environ_map.get("HF_TOKEN") orelse {
        log.err("HF_TOKEN env var must be set", .{});
        return error.MissingToken;
    };

    var xet_client: xet.Client = .init(allocator, &http_client, hf_token);
    defer xet_client.deinit();

    var xorbs: [MAX_XORBS]XorbKey = undefined;
    var plans: [MAX_PLANS]TermPlan = undefined;

    var n_total: u32 = 0;
    var n_scanned: u32 = 0;
    var n_skipped: u32 = 0;
    var n_with_dup: u32 = 0;
    var total_dup_plans: u64 = 0;

    var tensor_it = registry.tensors.iterator();
    while (tensor_it.next()) |entry| {
        n_total += 1;
        const tensor = entry.value_ptr.*;
        if (tensor.byteSize() < min_size) {
            n_skipped += 1;
            continue;
        }

        const hf_path = if (std.mem.startsWith(u8, tensor.file_uri, "hf://")) tensor.file_uri["hf://".len..] else tensor.file_uri;
        const hf_repo = zml.io.VFS.HF.Repo.parse(hf_path) catch {
            log.warn("skip {s}: bad file_uri {s}", .{ tensor.name, tensor.file_uri });
            n_skipped += 1;
            continue;
        };
        const repo: xet.Client.Repo = .{ .repo = hf_repo.repo, .model = hf_repo.model, .rev = hf_repo.rev, .path = hf_repo.path };

        const parsed = xet_client.reconstruct(repo, tensor.offset, tensor.offset + tensor.byteSize()) catch |e| {
            log.warn("skip {s}: reconstruction failed: {s}", .{ tensor.name, @errorName(e) });
            n_skipped += 1;
            continue;
        };
        defer parsed.deinit();
        const resp = parsed.value;

        // Pass 1: build plans + unique xorb list.
        const win_start: u64 = resp.offset_into_first_range;
        var n_xorbs: u16 = 0;
        var n_plans: u32 = 0;
        var stream_pos: u64 = 0;
        var overflowed = false;
        for (resp.terms) |t| {
            const t_off = stream_pos;
            stream_pos = t_off + t.unpacked_length;
            const dst_off: u64 = if (t_off < win_start) 0 else t_off - win_start;

            var xi: u16 = 0;
            while (xi < n_xorbs) : (xi += 1) {
                if (std.mem.eql(u8, xorbs[xi].hash, t.hash)) break;
            }
            if (xi == n_xorbs) {
                if (n_xorbs >= MAX_XORBS) {
                    overflowed = true;
                    break;
                }
                xorbs[n_xorbs] = .{ .hash = t.hash };
                n_xorbs += 1;
            }
            if (n_plans >= MAX_PLANS) {
                overflowed = true;
                break;
            }
            plans[n_plans] = .{
                .xorb_idx = xi,
                .dst_off = dst_off,
                .chunk_start = @intCast(t.range.start),
                .chunk_end = @intCast(t.range.end),
            };
            n_plans += 1;
        }
        if (overflowed) {
            log.warn("skip {s}: plans/xorbs overflow ({d} terms)", .{ tensor.name, resp.terms.len });
            n_skipped += 1;
            continue;
        }

        var dup: u32 = 0;
        for (plans[0..n_plans], 0..) |p, i| {
            for (plans[0..i]) |q| {
                if (p.xorb_idx == q.xorb_idx and p.chunk_start == q.chunk_start and p.chunk_end == q.chunk_end and p.dst_off != q.dst_off) {
                    dup += 1;
                    break;
                }
            }
        }
        n_scanned += 1;
        if (dup > 0) {
            n_with_dup += 1;
            total_dup_plans += dup;
            log.info("HIT  {s}: size={d} terms={d} xorbs={d} dup_plans={d}", .{
                tensor.name, tensor.byteSize(), n_plans, n_xorbs, dup,
            });
        } else {
            log.info("  -  {s}: size={d} terms={d} xorbs={d}", .{
                tensor.name, tensor.byteSize(), n_plans, n_xorbs,
            });
        }
    }

    log.info("done: total={d} scanned={d} skipped={d} with_dup={d} total_dup_plans={d}", .{
        n_total, n_scanned, n_skipped, n_with_dup, total_dup_plans,
    });
}
