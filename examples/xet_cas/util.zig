// Shared HTTP + hashing helpers for the xet_cas example binaries.
//
// These are example-only testing utilities (HTTP range GET into a caller-
// owned slot, LFS-oracle SHA-256 over a byte range, and SHA-256 of an
// arbitrary std.Io.Reader). They are intentionally not promoted into the
// library: production callers should drive xet.State directly.

const std = @import("std");
const xet = @import("io").xet;

const log = std.log.scoped(.xet_cas_util);

/// Issue a single HTTP range GET for [range_start, range_end_inclusive] and
/// read the full body into `slot`. `slot.len` must equal the requested range
/// length; short reads are an error.
pub fn httpRangeGetIntoSlot(
    client: *std.http.Client,
    url: []const u8,
    range_start: u64,
    range_end_inclusive: u64,
    slot: []u8,
) !void {
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ range_start, range_end_inclusive }) catch unreachable;
    const uri: std.Uri = try .parse(url);
    var req = try client.request(.GET, uri, .{
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .partial_content and res.head.status != .ok) {
        log.err("HTTP range GET failed: status={} url={s}", .{ res.head.status, url });
        return error.HttpRequestFailed;
    }
    try res.reader(&.{}).readSliceAll(slot);
}

/// LFS oracle: range-GET [range_start, range_end_inclusive] from the HF
/// resolve URL and return a streaming SHA-256 of the bytes. The cumulative
/// time spent waiting on the socket is reported via `out_net_ns`.
pub fn sha256LfsRange(
    client: *std.http.Client,
    repo: xet.State.Repo,
    auth: []const u8,
    range_start: u64,
    range_end_inclusive: u64,
    io: std.Io,
    out_net_ns: *u64,
) ![32]u8 {
    var url_buf: [4096]u8 = undefined;
    const lfs_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
        .{ repo.repo, repo.model, repo.rev, repo.path },
    );
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ range_start, range_end_inclusive }) catch unreachable;
    const uri: std.Uri = try .parse(lfs_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .partial_content and res.head.status != .ok) {
        log.err("LFS range GET failed: status={} url={s}", .{ res.head.status, lfs_url });
        return error.HttpRequestFailed;
    }
    var transfer_buf: [16 * 1024]u8 = undefined;
    const r = res.reader(&transfer_buf);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var stage: [64 * 1024]u8 = undefined;
    var net_ns_acc: u64 = 0;
    while (true) {
        const ts_net: std.Io.Timestamp = .now(io, .awake);
        const n = try r.readSliceShort(&stage);
        net_ns_acc += @intCast(ts_net.untilNow(io, .awake).toNanoseconds());
        if (n == 0) break;
        hasher.update(stage[0..n]);
        if (n < stage.len) break;
    }
    out_net_ns.* = net_ns_acc;
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

/// Stream `r` to EOF through SHA-256 and return the digest.
pub fn sha256ReadAll(r: *std.Io.Reader) ![32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var stage: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try r.readSliceShort(&stage);
        if (n == 0) break;
        hasher.update(stage[0..n]);
        if (n < stage.len) break;
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return digest;
}
