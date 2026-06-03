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

const RepoInfo = struct {
    namespace: []const u8,
    model: []const u8,
    rev: []const u8,
    filepath: []const u8,
};

const CasAuth = struct {
    url: []const u8,
    token: []const u8,
};

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
    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;

    // Per-file caches: file_id + cas auth are per (namespace, model, rev, filepath).
    var file_id_cache: std.StringHashMap([]const u8) = .init(allocator);
    defer {
        var it = file_id_cache.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.*);
        }
        file_id_cache.deinit();
    }
    var cas_cache: std.StringHashMap(CasAuth) = .init(allocator);
    defer {
        var it = cas_cache.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.*.url);
            allocator.free(e.value_ptr.*.token);
        }
        cas_cache.deinit();
    }

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

        const repo = parseFileUri(tensor.file_uri) catch {
            log.warn("skip {s}: bad file_uri {s}", .{ tensor.name, tensor.file_uri });
            n_skipped += 1;
            continue;
        };

        // Per-file repo key.
        const repo_key = try std.fmt.allocPrint(allocator, "{s}/{s}@{s}/{s}", .{
            repo.namespace, repo.model, repo.rev, repo.filepath,
        });
        defer allocator.free(repo_key);

        const file_id = blk: {
            if (file_id_cache.get(repo_key)) |fid| break :blk fid;
            const fid = getXetFileId(allocator, &http_client, repo, auth) catch |e| {
                log.warn("skip {s}: getXetFileId failed: {s}", .{ tensor.name, @errorName(e) });
                n_skipped += 1;
                continue;
            };
            try file_id_cache.put(try allocator.dupe(u8, repo_key), fid);
            break :blk fid;
        };

        const cas = blk: {
            if (cas_cache.get(repo_key)) |c| break :blk c;
            const c = getCasToken(allocator, &http_client, repo, auth) catch |e| {
                log.warn("skip {s}: getCasToken failed: {s}", .{ tensor.name, @errorName(e) });
                n_skipped += 1;
                continue;
            };
            try cas_cache.put(try allocator.dupe(u8, repo_key), c);
            break :blk c;
        };

        var cas_auth_buf: [65536]u8 = undefined;
        const cas_auth = std.fmt.bufPrint(&cas_auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

        const recon_body = callReconstruction(
            allocator,
            &http_client,
            cas.url,
            cas_auth,
            file_id,
            tensor.offset,
            tensor.offset + tensor.byteSize(),
        ) catch |e| {
            log.warn("skip {s}: reconstruction failed: {s}", .{ tensor.name, @errorName(e) });
            n_skipped += 1;
            continue;
        };
        defer allocator.free(recon_body);

        const parsed = std.json.parseFromSlice(
            xet.ReconstructionResponse,
            allocator,
            recon_body,
            .{ .ignore_unknown_fields = true },
        ) catch |e| {
            log.warn("skip {s}: json parse failed: {s}", .{ tensor.name, @errorName(e) });
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

fn parseFileUri(uri: []const u8) !RepoInfo {
    var path = uri;
    if (std.mem.startsWith(u8, path, "hf://")) path = path["hf://".len..];
    var parts = std.mem.splitScalar(u8, path, '/');
    const namespace = parts.next() orelse return error.InvalidFileUri;
    var model = parts.next() orelse return error.InvalidFileUri;
    var rev: []const u8 = "main";
    if (std.mem.indexOfScalar(u8, model, '@')) |at| {
        rev = model[at + 1 ..];
        model = model[0..at];
    }
    const filepath = parts.rest();
    if (filepath.len == 0) return error.InvalidFileUri;
    return .{ .namespace = namespace, .model = model, .rev = rev, .filepath = filepath };
}

fn getXetFileId(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: RepoInfo,
    auth: []const u8,
) ![]const u8 {
    var url_buf: [4096]u8 = undefined;
    const resolve_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
        .{ repo.namespace, repo.model, repo.rev, repo.filepath },
    );
    const uri: std.Uri = try .parse(resolve_url);
    var req = try client.request(.GET, uri, .{
        .redirect_behavior = .unhandled,
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    var header_it = res.head.iterateHeaders();
    while (header_it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "X-Xet-Hash")) {
            return try allocator.dupe(u8, header.value);
        }
    }
    return error.XetHashNotFound;
}

fn getCasToken(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: RepoInfo,
    auth: []const u8,
) !CasAuth {
    var url_buf: [4096]u8 = undefined;
    const token_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/api/models/{s}/{s}/xet-read-token/{s}",
        .{ repo.namespace, repo.model, repo.rev },
    );
    const uri: std.Uri = try .parse(token_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [4 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok) return error.CasTokenFailed;
    const body = try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 128 * 1024);
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(struct {
        accessToken: []const u8,
        casUrl: []const u8,
    }, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return .{
        .url = try allocator.dupe(u8, parsed.value.casUrl),
        .token = try allocator.dupe(u8, parsed.value.accessToken),
    };
}

fn callReconstruction(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    cas_url: []const u8,
    cas_auth: []const u8,
    file_id: []const u8,
    range_start: u64,
    range_end_exclusive: u64,
) ![]const u8 {
    var url_buf: [4096]u8 = undefined;
    const recon_url = try std.fmt.bufPrint(&url_buf, "{s}/v1/reconstructions/{s}", .{ cas_url, file_id });
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ range_start, range_end_exclusive - 1 }) catch unreachable;
    const uri: std.Uri = try .parse(recon_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = cas_auth },
        },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok and res.head.status != .partial_content) return error.ReconstructionFailed;
    return try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 64 * 1024 * 1024);
}
