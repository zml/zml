const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.xet_cas);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const TEN_GB: u64 = 10 * 1024 * 1024 * 1024;

const Args = struct {
    output: []const u8 = ".",

    positional: struct {
        url: []const u8,
    },

    pub const help =
        \\ Fetch CAS reconstruction data for a HuggingFace safetensors file.
        \\
        \\ Usage: xet_cas [options] <url>
        \\
        \\ Arguments:
        \\   <url>             HuggingFace safetensors blob URL
        \\
        \\ Options:
        \\   --output=<path>   Output directory for manifest and group files (default: .)
        \\
        \\ Environment:
        \\   HF_TOKEN          HuggingFace API token (required)
        \\
        \\ Example:
        \\   xet_cas --output=./out https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors
        \\
    ;
};

// ── Types ───────────────────────────────────────────────────────────────────

const RepoInfo = struct {
    namespace: []const u8,
    model: []const u8,
    rev: []const u8,
    filepath: []const u8,
};

const MAX_RANK = 8;

const TensorInfo = struct {
    name: []const u8, // points into parsed JSON (arena-owned)
    dtype: []const u8, // points into parsed JSON (arena-owned)
    shape_dims: [MAX_RANK]i64 = .{0} ** MAX_RANK,
    shape_len: u4 = 0,
    data_start: u64, // absolute byte offset in file
    data_end: u64, // absolute byte offset in file (exclusive)

    fn shape(self: TensorInfo) []const i64 {
        return self.shape_dims[0..self.shape_len];
    }

    fn byteSize(self: TensorInfo) u64 {
        return self.data_end - self.data_start;
    }

    fn lessThan(_: void, a: TensorInfo, b: TensorInfo) bool {
        return a.data_start < b.data_start;
    }
};

const TensorGroup = struct {
    tensors: []const TensorInfo,
    range_start: u64, // absolute file offset (first tensor start)
    range_end: u64, // absolute file offset (last tensor end, exclusive)
};

const CasAuth = struct {
    url: []const u8,
    token: []const u8,
};

// ── URL parsing ─────────────────────────────────────────────────────────────

fn parseHFUrl(url: []const u8) !RepoInfo {
    // https://huggingface.co/{namespace}/{model}/blob/{rev}/{filepath}
    var path = url;
    if (std.mem.startsWith(u8, path, "https://huggingface.co/")) {
        path = path["https://huggingface.co/".len..];
    } else if (std.mem.startsWith(u8, path, "http://huggingface.co/")) {
        path = path["http://huggingface.co/".len..];
    }

    var parts = std.mem.splitScalar(u8, path, '/');
    const namespace = parts.next() orelse return error.InvalidURL;
    const model = parts.next() orelse return error.InvalidURL;
    _ = parts.next() orelse return error.InvalidURL; // "blob" or "resolve"
    const rev = parts.next() orelse return error.InvalidURL;
    const filepath = parts.rest();
    if (filepath.len == 0) return error.InvalidURL;

    return .{
        .namespace = namespace,
        .model = model,
        .rev = rev,
        .filepath = filepath,
    };
}

// ── HTTP helpers ────────────────────────────────────────────────────────────

fn httpRangeGet(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    auth: []const u8,
    range_start: u64,
    range_end_inclusive: u64,
) ![]u8 {
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ range_start, range_end_inclusive }) catch unreachable;

    const uri: std.Uri = try .parse(url);

    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
        .extra_headers = &.{
            .{ .name = "Range", .value = range_header },
        },
    });
    defer req.deinit();

    try req.sendBodiless();

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);

    if (res.head.status != .partial_content and res.head.status != .ok) {
        log.err("HTTP range GET failed: status={} url={s}", .{ res.head.status, url });
        return error.HttpRequestFailed;
    }

    const expected_len = range_end_inclusive - range_start + 1;
    return try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse expected_len);
}

// ── Safetensor header parsing ───────────────────────────────────────────────

const HeaderResult = struct {
    tensors: []TensorInfo,
    data_start_offset: u64,
};

fn fetchSafetensorHeader(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: RepoInfo,
    auth: []const u8,
) !HeaderResult {
    var url_buf: [4096]u8 = undefined;
    const resolve_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
        .{ repo.namespace, repo.model, repo.rev, repo.filepath },
    );

    // Fetch first 8 bytes: little-endian u64 = JSON header length
    const len_bytes = try httpRangeGet(allocator, client, resolve_url, auth, 0, 7);

    if (len_bytes.len < 8) return error.InvalidSafetensor;
    const json_header_length = std.mem.readInt(u64, len_bytes[0..8], .little);
    const data_start_offset: u64 = 8 + json_header_length;

    log.info("JSON header: {} bytes, data offset: {}", .{ json_header_length, data_start_offset });

    // Fetch the JSON header — kept alive so TensorInfo.name/dtype can point into it
    const json_data = try httpRangeGet(allocator, client, resolve_url, auth, 8, 8 + json_header_length - 1);

    // Parse — kept alive (arena-owned), tensor names/dtypes reference its strings
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});

    const obj = &parsed.value.object;
    const num_tensors = obj.count() - @as(usize, if (obj.contains("__metadata__")) 1 else 0);

    var tensor_list: std.ArrayListUnmanaged(TensorInfo) = .empty;
    try tensor_list.ensureTotalCapacity(allocator, num_tensors);

    var it = obj.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) continue;

        const dtype_val = value.object.get("dtype") orelse continue;
        const shape_field = value.object.get("shape") orelse continue;
        const offsets_field = value.object.get("data_offsets") orelse continue;

        if (shape_field.array.items.len > MAX_RANK) {
            log.warn("Skipping tensor {s}: rank {} exceeds max {}", .{ key, shape_field.array.items.len, MAX_RANK });
            continue;
        }

        const rel_start: u64 = @intCast(offsets_field.array.items[0].integer);
        const rel_end: u64 = @intCast(offsets_field.array.items[1].integer);

        var info: TensorInfo = .{
            .name = key, // borrows from parsed JSON
            .dtype = dtype_val.string, // borrows from parsed JSON
            .data_start = data_start_offset + rel_start,
            .data_end = data_start_offset + rel_end,
        };
        for (shape_field.array.items, 0..) |dim, di| {
            info.shape_dims[di] = dim.integer;
        }
        info.shape_len = @intCast(shape_field.array.items.len);

        tensor_list.appendAssumeCapacity(info);
    }

    const tensors = try tensor_list.toOwnedSlice(allocator);
    std.sort.pdq(TensorInfo, tensors, {}, TensorInfo.lessThan);

    return .{ .tensors = tensors, .data_start_offset = data_start_offset };
}

// ── Tensor grouping ─────────────────────────────────────────────────────────

fn groupTensors(
    allocator: std.mem.Allocator,
    tensors: []const TensorInfo,
) ![]TensorGroup {
    if (tensors.len == 0) return try allocator.alloc(TensorGroup, 0);

    var groups: std.ArrayListUnmanaged(TensorGroup) = .empty;
    var group_start_idx: usize = 0;

    for (0..tensors.len) |i| {
        const is_last = (i + 1 == tensors.len);
        const current_range_size = tensors[i].data_end - tensors[group_start_idx].data_start;

        // Would adding the next tensor exceed 10GB?
        const should_cut = if (!is_last) blk: {
            const next_range_size = tensors[i + 1].data_end - tensors[group_start_idx].data_start;
            break :blk next_range_size > TEN_GB;
        } else true;

        if (should_cut) {
            try groups.append(allocator, .{
                .tensors = tensors[group_start_idx .. i + 1],
                .range_start = tensors[group_start_idx].data_start,
                .range_end = tensors[i].data_end,
            });
            group_start_idx = i + 1;
            _ = current_range_size;
        }
    }

    return try groups.toOwnedSlice(allocator);
}

// ── Xet file ID (X-Xet-Hash) ───────────────────────────────────────────────

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

    // Do NOT follow redirects — the X-Xet-Hash is on the 302 response
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

    log.err("X-Xet-Hash not found (status={}). File may not use Xet storage.", .{res.head.status});
    return error.XetHashNotFound;
}

// ── CAS token ───────────────────────────────────────────────────────────────

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

    if (res.head.status != .ok) {
        log.err("CAS token request failed: status={}", .{res.head.status});
        return error.CasTokenFailed;
    }

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

// ── CAS reconstruction ─────────────────────────────────────────────────────

fn callReconstruction(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    cas_url: []const u8,
    cas_auth: []const u8,
    file_id: []const u8,
    group: TensorGroup,
) ![]const u8 {
    var url_buf: [4096]u8 = undefined;
    const recon_url = try std.fmt.bufPrint(
        &url_buf,
        "{s}/v1/reconstructions/{s}",
        .{ cas_url, file_id },
    );

    // HTTP Range end is inclusive
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(
        &range_buf,
        "bytes={}-{}",
        .{ group.range_start, group.range_end - 1 },
    ) catch unreachable;

    const uri: std.Uri = try .parse(recon_url);

    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = cas_auth },
        },
        .extra_headers = &.{
            .{ .name = "Range", .value = range_header },
        },
    });
    defer req.deinit();

    try req.sendBodiless();

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);

    if (res.head.status != .ok and res.head.status != .partial_content) {
        log.err("Reconstruction failed: status={}", .{res.head.status});
        if (res.reader(&.{}).readAlloc(allocator, 4096)) |err_body| {
            log.err("Body: {s}", .{err_body});
            allocator.free(err_body);
        } else |_| {}
        return error.ReconstructionFailed;
    }

    return try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 64 * 1024 * 1024);
}

// ── Manifest builder ────────────────────────────────────────────────────────

fn buildManifestValue(
    allocator: std.mem.Allocator,
    file_id: []const u8,
    cas_url: []const u8,
    header_result: HeaderResult,
    groups: []const TensorGroup,
    total_bytes: u64,
) !std.json.Value {
    var root: std.json.ObjectMap = .empty;
    try root.put(allocator, "file_id", .{ .string = file_id });
    try root.put(allocator, "cas_url", .{ .string = cas_url });
    try root.put(allocator, "data_start_offset", .{ .integer = @intCast(header_result.data_start_offset) });
    try root.put(allocator, "num_tensors", .{ .integer = @intCast(header_result.tensors.len) });
    try root.put(allocator, "total_bytes", .{ .integer = @intCast(total_bytes) });
    try root.put(allocator, "num_groups", .{ .integer = @intCast(groups.len) });

    var groups_arr = std.json.Array.init(allocator);
    for (groups, 0..) |group, gi| {
        var g: std.json.ObjectMap = .empty;
        try g.put(allocator, "group_index", .{ .integer = @intCast(gi) });
        try g.put(allocator, "range_start", .{ .integer = @intCast(group.range_start) });
        try g.put(allocator, "range_end", .{ .integer = @intCast(group.range_end) });
        try g.put(allocator, "range_size_bytes", .{ .integer = @intCast(group.range_end - group.range_start) });
        try g.put(allocator, "num_tensors", .{ .integer = @intCast(group.tensors.len) });

        var tensors_arr = std.json.Array.init(allocator);
        for (group.tensors) |t| {
            var tensor: std.json.ObjectMap = .empty;
            try tensor.put(allocator, "name", .{ .string = t.name });
            try tensor.put(allocator, "dtype", .{ .string = t.dtype });
            try tensor.put(allocator, "data_start", .{ .integer = @intCast(t.data_start) });
            try tensor.put(allocator, "data_end", .{ .integer = @intCast(t.data_end) });
            try tensor.put(allocator, "size_bytes", .{ .integer = @intCast(t.byteSize()) });

            var shape_arr = std.json.Array.init(allocator);
            for (t.shape()) |dim| {
                try shape_arr.append(.{ .integer = dim });
            }
            try tensor.put(allocator, "shape", .{ .array = shape_arr });

            try tensors_arr.append(.{ .object = tensor });
        }
        try g.put(allocator, "tensors", .{ .array = tensors_arr });

        try groups_arr.append(.{ .object = g });
    }
    try root.put(allocator, "groups", .{ .array = groups_arr });

    return .{ .object = root };
}

// ── Main ────────────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const args = stdx.flags.parse(init.minimal.args, Args);
    const url = args.positional.url;
    const output_dir = args.output;

    const repo = try parseHFUrl(url);
    const hf_token = init.environ_map.get("HF_TOKEN") orelse {
        log.err("HF_TOKEN environment variable must be set", .{});
        return error.MissingToken;
    };

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;

    // ── Step 1: Fetch & parse safetensor header ─────────────────────────────

    log.info("Fetching safetensor header for {s}/{s} @ {s} / {s}...", .{
        repo.namespace, repo.model, repo.rev, repo.filepath,
    });
    const header_result = try fetchSafetensorHeader(arena_alloc, &http_client, repo, auth);
    log.info("Found {} tensors, data starts at byte {}", .{ header_result.tensors.len, header_result.data_start_offset });

    // Print total size
    var total_bytes: u64 = 0;
    for (header_result.tensors) |t| total_bytes += t.byteSize();
    log.info("Total tensor data: {d:.2} GB", .{@as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0 * 1024.0)});

    // ── Step 2: Group tensors into ≤10GB batches ────────────────────────────

    const groups = try groupTensors(arena_alloc, header_result.tensors);
    log.info("Created {} groups (max 10 GB each)", .{groups.len});

    for (groups, 0..) |g, i| {
        log.info("  Group {}: {} tensors, {d:.2} GB ({}-{})", .{
            i,                                                                                 g.tensors.len,
            @as(f64, @floatFromInt(g.range_end - g.range_start)) / (1024.0 * 1024.0 * 1024.0), g.range_start,
            g.range_end,
        });
    }

    // ── Step 3: Get Xet file ID ─────────────────────────────────────────────

    log.info("Getting Xet file ID...", .{});
    const file_id = try getXetFileId(arena_alloc, &http_client, repo, auth);
    log.info("File ID: {s}", .{file_id});

    // ── Step 4: Get CAS token ───────────────────────────────────────────────

    log.info("Getting CAS read token...", .{});
    const cas = try getCasToken(arena_alloc, &http_client, repo, auth);
    log.info("CAS endpoint: {s}", .{cas.url});

    var cas_auth_buf: [65536]u8 = undefined;
    const cas_auth = std.fmt.bufPrint(&cas_auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

    // ── Step 5: Fetch reconstructions & write files ──────────────────────────

    const io = init.io;

    // Create output directory if needed
    if (!std.mem.eql(u8, output_dir, ".")) {
        const cwd: std.Io.Dir = .cwd();
        cwd.createDirPath(io, output_dir) catch |err| {
            log.err("Failed to create output directory '{s}': {}", .{ output_dir, err });
            return err;
        };
    }

    const write_buf = try arena_alloc.alloc(u8, 256 * 1024);

    for (groups, 0..) |group, gi| {
        log.info("Fetching reconstruction for group {}/{}...", .{ gi + 1, groups.len });

        const recon = try callReconstruction(arena_alloc, &http_client, cas.url, cas_auth, file_id, group);

        // Parse and pretty-print CAS response to group_NNN.json
        const parsed_recon = try std.json.parseFromSlice(std.json.Value, arena_alloc, recon, .{});

        var name_buf: [256]u8 = undefined;
        const group_path = try std.fmt.bufPrint(&name_buf, "{s}/group_{d:0>3}.json", .{ output_dir, gi });
        const group_file = try std.Io.Dir.createFile(.cwd(), io, group_path, .{});
        defer group_file.close(io);
        var gw = group_file.writer(io, write_buf);
        var jw: std.json.Stringify = .{ .writer = &gw.interface, .options = .{ .whitespace = .indent_2 } };
        try parsed_recon.value.jsonStringify(&jw);
        _ = try gw.interface.write("\n");
        try gw.flush();

        log.info("  Wrote {s}", .{group_path});
    }

    // ── Write manifest.json ─────────────────────────────────────────────────

    var manifest_path_buf: [256]u8 = undefined;
    const manifest_path = try std.fmt.bufPrint(&manifest_path_buf, "{s}/manifest.json", .{output_dir});
    const manifest_file = try std.Io.Dir.createFile(.cwd(), io, manifest_path, .{});
    defer manifest_file.close(io);

    const manifest_value = try buildManifestValue(arena_alloc, file_id, cas.url, header_result, groups, total_bytes);

    const buffer = try arena_alloc.alloc(u8, 256 * 1024);
    var mw = manifest_file.writer(io, buffer);
    var jw: std.json.Stringify = .{ .writer = &mw.interface, .options = .{ .whitespace = .indent_2 } };
    try manifest_value.jsonStringify(&jw);
    _ = try mw.interface.write("\n");
    try mw.flush();

    log.info("Wrote {s}/manifest.json", .{output_dir});
    log.info("Done.", .{});
}
