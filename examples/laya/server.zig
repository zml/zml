//! Minimal HTTP front-end for the Laya engine, serving the interactive demo.
//!
//!   GET  /             demo page (demo.html)
//!   GET  /v1/health    {"status": "ok", ...}
//!   POST /v1/decide    {"state": ..., "questions": {...}} -> Laya response + token view

const std = @import("std");

const zml = @import("zml");

const Engine = @import("engine.zig").Engine;
const prompt = @import("prompt.zig");

const log = std.log.scoped(.laya_server);

const demo_html = @embedFile("demo.html");

/// Requests larger than this are rejected (the model only reads `seqlen` tokens anyway).
const max_body_size = 1 << 20;

pub fn serve(allocator: std.mem.Allocator, io: std.Io, engine: *Engine, host: []const u8, port: u16) !void {
    const address: std.Io.net.IpAddress = try .parse(host, port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    log.info("🚀 Laya demo ready on http://{s}:{d}  (Ctrl-C to stop)", .{ host, port });

    // One connection at a time: the engine owns a single compiled executable.
    while (true) {
        const stream = server.accept(io) catch |err| {
            log.warn("accept failed: {t}", .{err});
            continue;
        };
        defer stream.close(io);
        handleConnection(allocator, io, engine, stream) catch |err| {
            log.warn("connection error: {t}", .{err});
        };
    }
}

fn handleConnection(allocator: std.mem.Allocator, io: std.Io, engine: *Engine, stream: std.Io.net.Stream) !void {
    var read_buffer: [16 * 1024]u8 = undefined;
    var write_buffer: [16 * 1024]u8 = undefined;
    var reader = stream.reader(io, &read_buffer);
    var writer = stream.writer(io, &write_buffer);
    var http: std.http.Server = .init(&reader.interface, &writer.interface);

    // One request per connection: connections are served one at a time, so an idle
    // keep-alive client (e.g. a browser tab) would otherwise block everyone else.
    var request = http.receiveHead() catch |err| switch (err) {
        error.HttpConnectionClosing => return,
        else => return err,
    };
    try handleRequest(allocator, engine, &request);
}

fn handleRequest(allocator: std.mem.Allocator, engine: *Engine, request: *std.http.Server.Request) !void {
    const target = request.head.target;
    const path = if (std.mem.indexOfScalar(u8, target, '?')) |i| target[0..i] else target;

    if (request.head.method == .GET and (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html"))) {
        return request.respond(demo_html, .{ .keep_alive = false, .extra_headers = &.{
            .{ .name = "content-type", .value = "text/html; charset=utf-8" },
            .{ .name = "cache-control", .value = "no-store" },
        } });
    }

    if (request.head.method == .GET and std.mem.eql(u8, path, "/v1/health")) {
        var buf: [256]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf, "{{\"status\": \"ok\", \"platform\": \"{t}\", \"seqlen\": {d}}}", .{ engine.platform.target, engine.seqlen });
        return respondJson(request, .ok, body);
    }

    if (request.head.method == .POST and std.mem.eql(u8, path, "/v1/decide")) {
        return handleDecide(allocator, engine, request);
    }

    return respondJson(request, .not_found, "{\"error\": \"not found\"}");
}

fn handleDecide(allocator: std.mem.Allocator, engine: *Engine, request: *std.http.Server.Request) !void {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const content_length = request.head.content_length orelse
        return respondJson(request, .length_required, "{\"error\": \"content-length required\"}");
    if (content_length > max_body_size)
        return respondJson(request, .payload_too_large, "{\"error\": \"request too large\"}");

    var body_buffer: [4096]u8 = undefined;
    const body_reader = try request.readerExpectContinue(&body_buffer);
    const body = try body_reader.readAlloc(arena.allocator(), @intCast(content_length));

    const input = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{}) catch
        return respondJson(request, .bad_request, "{\"error\": \"body must be JSON\"}");
    if (input != .object)
        return respondJson(request, .bad_request, "{\"error\": \"body must be a JSON object\"}");
    const state = input.object.get("state") orelse
        return respondJson(request, .bad_request, "{\"error\": \"missing state\"}");
    const questions = input.object.get("questions") orelse
        return respondJson(request, .bad_request, "{\"error\": \"missing questions\"}");

    const result = engine.decide(arena.allocator(), state, questions) catch |err| {
        const message = try std.fmt.allocPrint(arena.allocator(), "{{\"error\": \"{t}\"}}", .{err});
        return respondJson(request, .unprocessable_entity, message);
    };
    log.info("decided {d} question(s), {d} tokens, {d:.1} ms", .{ result.answers.len, result.input_tokens, result.latency_ms });

    var out: std.Io.Writer.Allocating = .init(arena.allocator());
    try prompt.writeResponse(&out.writer, result.answers, result.input_tokens, result.latency_ms);
    // Splice the token view in front of the closing brace so the demo can animate the prompt.
    out.writer.undo(1);
    try out.writer.writeAll(", \"prompts\": {");
    for (result.prompts, result.answers, 0..) |p, a, i| {
        if (i > 0) try out.writer.writeAll(", ");
        try out.writer.print("{f}: {{\"markers\": {f}, \"tokens\": [", .{ std.json.fmt(a.question.id, .{}), std.json.fmt(p.markers, .{}) });
        for (p.ids, 0..) |id, j| {
            if (j > 0) try out.writer.writeAll(", ");
            try out.writer.print("{f}", .{std.json.fmt(engine.tokenText(arena.allocator(), id), .{})});
        }
        try out.writer.writeAll("]}");
    }
    try out.writer.writeAll("}}");

    return respondJson(request, .ok, out.written());
}

fn respondJson(request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
    return request.respond(body, .{
        .status = status,
        .keep_alive = false,
        .extra_headers = &.{
            .{ .name = "content-type", .value = "application/json" },
            .{ .name = "access-control-allow-origin", .value = "*" },
        },
    });
}
