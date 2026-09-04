const std = @import("std");

const zml = @import("zml");

const pipeline = @import("../draft/pipeline.zig");
const refine = @import("../refine/run.zig");
const repo = @import("repo.zig");
const session = @import("../draft/session.zig");
const sku = @import("../recipe/sku.zig");
const taeh3 = @import("../draft/taeh3.zig");

const log = std.log.scoped(.minimax_h3);

// ============================================================================
// serve/http.zig — queue, /generate, /video, status JSON
// ============================================================================

/// Encoder / pack width for the browser server. Shorter prompts are padded.
pub const text_len: u32 = sku.prompt_tokens;

const html = @import("page.zig").html;

const token_len: usize = 16;
const token_hex_len: usize = token_len * 2;
const max_clients: u32 = 64;
const max_videos: u8 = 12;
const max_queue: u32 = 8;

const Token = [token_len]u8;

const Guest = struct {
    token: Token,
    set_cookie: ?[]const u8,
};

const Client = struct {
    used: bool = false,
    token: Token = @splat(0),
    videos: [max_videos]Token = @splat(@splat(0)),
    video_n: u8 = 0,
};

pub const Phase = enum(u8) { compiling, ready, failed };

pub const CompileSku = enum(u8) { pending, run, done, skip };

pub const Lane = struct {
    id: []const u8,
    duration_s: f32,
    target_w: u32,
    target_h: u32,
    hd: bool,
    geo: pipeline.Geometry,
    packed_run: *pipeline.Packed,
    h3: *pipeline.Compiled,
    taeh3: *taeh3.Compiled,
    ltx: *refine.Compiled,
    bake: *session.Bake,
    resident_blocks: u32 = 0,

    pub fn deinit(self: *Lane, allocator: std.mem.Allocator) void {
        self.packed_run.deinit(allocator);
        allocator.destroy(self.packed_run);
        self.h3.deinit();
        allocator.destroy(self.h3);
        self.taeh3.deinit();
        allocator.destroy(self.taeh3);
        self.ltx.deinit();
        allocator.destroy(self.ltx);
        self.bake.deinit(allocator);
        allocator.destroy(self.bake);
        self.* = undefined;
    }
};

pub const Runtime = struct {
    platform: *const zml.Platform,
    models: *repo.Bundle,
    warm: *session.Warm,
    shardings: []const zml.Sharding,
    tokenizer: *zml.tokenizer.Tokenizer,
    progress: *std.Progress.Node,
    resident_blocks: u32,
    lanes: []Lane,
    last_prompt: []u8 = &.{},
};

fn findLane(rt: *Runtime, id: []const u8) ?*Lane {
    for (rt.lanes) |*lane| {
        if (std.mem.eql(u8, lane.id, id)) return lane;
    }
    return null;
}

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    port: u16,
    devices: u32 = 0,
    runtime: ?*Runtime = null,
    phase: std.atomic.Value(u8) = .init(@intFromEnum(Phase.compiling)),
    compile_label: [80]u8 = @splat(0),
    compile_label_n: u32 = 0,
    compile_pct: u32 = 0,
    compile_sku: [sku.skus.len]CompileSku = @splat(.pending),
    mutex: std.Io.Mutex = .init,
    meta: std.Io.Mutex = .init,
    busy: std.atomic.Value(bool) = .init(false),
    job: session.JobProgress = .{},
    clients: [max_clients]Client = @splat(.{}),
    waiters: [max_queue]Token = @splat(@splat(0)),
    waiter_n: u32 = 0,
    running: Token = @splat(0),
    has_running: bool = false,
    evict_i: u32 = 0,

    pub fn setCompile(app: *App, label: []const u8, pct: u32) void {
        app.meta.lockUncancelable(app.io);
        defer app.meta.unlock(app.io);
        const n: u32 = @intCast(@min(label.len, app.compile_label.len));
        @memcpy(app.compile_label[0..n], label[0..n]);
        app.compile_label_n = n;
        app.compile_pct = @min(pct, 100);
        if (app.phase.load(.acquire) == @intFromEnum(Phase.failed)) return;
        app.phase.store(@intFromEnum(Phase.compiling), .release);
    }

    pub fn setCompileSku(app: *App, id: []const u8, state: CompileSku) void {
        app.meta.lockUncancelable(app.io);
        defer app.meta.unlock(app.io);
        for (sku.skus, 0..) |row, i| {
            if (std.mem.eql(u8, row.id, id)) {
                app.compile_sku[i] = state;
                return;
            }
        }
    }

    pub fn setReady(app: *App, runtime: *Runtime, devices: u32) void {
        app.runtime = runtime;
        app.devices = devices;
        app.phase.store(@intFromEnum(Phase.ready), .release);
        log.info("compiled, generate ready  http://0.0.0.0:{d}  {d} GPU", .{ app.port, devices });
    }

    pub fn setFailed(app: *App, label: []const u8) void {
        app.meta.lockUncancelable(app.io);
        defer app.meta.unlock(app.io);
        const n: u32 = @intCast(@min(label.len, app.compile_label.len));
        @memcpy(app.compile_label[0..n], label[0..n]);
        app.compile_label_n = n;
        app.phase.store(@intFromEnum(Phase.failed), .release);
    }
};

pub fn run(app: *App) !void {
    try std.Io.Dir.cwd().createDirPath(app.io, "output/serve");
    const address: std.Io.net.IpAddress = .{ .ip4 = .{
        .bytes = .{ 0, 0, 0, 0 },
        .port = app.port,
    } };
    var tcp = try address.listen(app.io, .{ .reuse_address = true });
    defer tcp.deinit(app.io);
    log.info("page open  http://0.0.0.0:{d}  compiling", .{app.port});

    var group: std.Io.Group = .init;
    while (true) {
        const stream = tcp.accept(app.io) catch break;
        group.concurrent(app.io, onConnection, .{ app, stream }) catch {
            stream.close(app.io);
            break;
        };
    }
    group.await(app.io) catch {};
}

fn onConnection(app: *App, stream: std.Io.net.Stream) std.Io.Cancelable!void {
    defer stream.close(app.io);
    handleConn(app, stream) catch |err| {
        log.warn("http {s}", .{@errorName(err)});
    };
}

fn handleConn(app: *App, stream: std.Io.net.Stream) !void {
    var read_buf: [16 * 1024]u8 = undefined;
    var reader = stream.reader(app.io, &read_buf);
    var write_buf: [16 * 1024]u8 = undefined;
    var writer = stream.writer(app.io, &write_buf);
    var http: std.http.Server = .init(&reader.interface, &writer.interface);

    while (true) {
        var request = http.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };
        var path_buf: [256]u8 = undefined;
        const path = clipPath(request.head.target, &path_buf) orelse {
            try request.respond("bad path\n", .{ .status = .bad_request });
            continue;
        };
        const method = request.head.method;
        var cookie_buf: [96]u8 = undefined;
        const guest = ensureGuest(app, &request, &cookie_buf);
        if (method == .GET and std.mem.eql(u8, path, "/")) {
            try replyPage(&request, guest.set_cookie);
        } else if (method == .GET and (std.mem.eql(u8, path, "/ready") or std.mem.eql(u8, path, "/api/status"))) {
            try replyStatus(app, &request, guest);
        } else if (method == .POST and std.mem.eql(u8, path, "/generate")) {
            try generate(app, &request, guest);
        } else if (method == .GET and std.mem.startsWith(u8, path, "/video/")) {
            try sendVideo(app, &request, guest, path["/video/".len..]);
        } else {
            try request.respond("not found\n", .{ .status = .not_found });
        }
    }
}

fn generate(app: *App, request: *std.http.Server.Request, guest: Guest) !void {
    if (app.phase.load(.acquire) != @intFromEnum(Phase.ready)) {
        try replyJson(request, .service_unavailable, "{\"ok\":false,\"error\":\"compiling\"}\n", guest.set_cookie);
        return;
    }
    const body = readBody(app.allocator, request) catch {
        try replyJson(request, .bad_request, "{\"ok\":false,\"error\":\"bad json\"}\n", guest.set_cookie);
        return;
    };
    defer app.allocator.free(body);

    const parsed = std.json.parseFromSlice(GenReq, app.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        try replyJson(request, .bad_request, "{\"ok\":false,\"error\":\"bad json\"}\n", guest.set_cookie);
        return;
    };
    defer parsed.deinit();
    const prompt = std.mem.trim(u8, parsed.value.prompt, " \t\r\n");
    if (prompt.len == 0) {
        try replyJson(request, .bad_request, "{\"ok\":false,\"error\":\"empty prompt\"}\n", guest.set_cookie);
        return;
    }
    const sku_id = if (parsed.value.sku.len != 0) parsed.value.sku else sku.default_sku_id;
    if (findLane(app.runtime.?, sku_id) == null) {
        try replyJson(request, .bad_request, "{\"ok\":false,\"error\":\"unknown sku\"}\n", guest.set_cookie);
        return;
    }

    const queued = blk: {
        app.meta.lockUncancelable(app.io);
        defer app.meta.unlock(app.io);
        if (app.waiter_n >= max_queue) break :blk false;
        app.waiters[app.waiter_n] = guest.token;
        app.waiter_n += 1;
        break :blk true;
    };
    if (!queued) {
        try replyJson(request, .too_many_requests, "{\"ok\":false,\"error\":\"queue full\"}\n", guest.set_cookie);
        return;
    }

    app.mutex.lockUncancelable(app.io);
    app.busy.store(true, .seq_cst);
    {
        app.meta.lockUncancelable(app.io);
        defer app.meta.unlock(app.io);
        removeWaiter(app, guest.token);
        app.running = guest.token;
        app.has_running = true;
    }
    defer {
        app.meta.lockUncancelable(app.io);
        app.has_running = false;
        app.running = @splat(0);
        app.meta.unlock(app.io);
        app.job.clear();
        app.busy.store(false, .seq_cst);
        app.mutex.unlock(app.io);
    }

    const result = generateLocked(app, prompt, parsed.value.seed, sku_id, guest.token) catch |err| {
        var err_buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "{{\"ok\":false,\"error\":\"{s}\"}}\n", .{@errorName(err)}) catch "{\"ok\":false}\n";
        try replyJson(request, .internal_server_error, msg, guest.set_cookie);
        return;
    };
    var hex_buf: [token_hex_len]u8 = undefined;
    const hex = fmtToken(result.video, &hex_buf);
    var out_buf: [256]u8 = undefined;
    const json = std.fmt.bufPrint(&out_buf, "{{\"ok\":true,\"video\":\"/video/{s}.mp4\",\"infer_ms\":{d},\"draft_ms\":{d},\"refine_ms\":{d},\"seed\":{d}}}\n", .{
        hex,
        result.infer_ms,
        result.draft_ms,
        result.refine_ms,
        result.seed,
    }) catch "{\"ok\":true}\n";
    try replyJson(request, .ok, json, guest.set_cookie);
}

const GenReq = struct {
    prompt: []const u8,
    seed: u64 = 42,
    sku: []const u8 = "",
};

const GenOut = struct {
    video: Token,
    infer_ms: u64,
    draft_ms: u64,
    refine_ms: u64,
    seed: u64,
};

fn generateLocked(app: *App, prompt: []const u8, seed: u64, sku_id: []const u8, owner: Token) !GenOut {
    const rt = app.runtime orelse return error.NotReady;
    const lane = findLane(rt, sku_id) orelse return error.UnknownSku;
    const ids = blk: {
        var enc = try rt.tokenizer.encoder();
        defer enc.deinit();
        break :blk try enc.encodeAlloc(app.allocator, prompt);
    };
    defer app.allocator.free(ids);
    const tokens = try padTokens(app.allocator, ids, text_len);
    defer app.allocator.free(tokens);
    if (ids.len > text_len) log.warn("prompt truncated to {d} H3 tokens (got {d})", .{ text_len, ids.len });

    const gemma_lane = &rt.lanes[0];
    if (gemma_lane.ltx.ctx == null or !std.mem.eql(u8, rt.last_prompt, prompt)) {
        app.job.set(.text, 1, 1);
        try refine.refreshContext(app.allocator, app.io, rt.platform, gemma_lane.ltx, prompt);
        const copy = try app.allocator.dupe(u8, prompt);
        if (rt.last_prompt.len != 0) app.allocator.free(rt.last_prompt);
        rt.last_prompt = copy;
    }
    const borrow = lane != gemma_lane;
    if (borrow) lane.ltx.ctx = gemma_lane.ltx.ctx;
    defer if (borrow) {
        lane.ltx.ctx = null;
    };

    const infer_start: std.Io.Timestamp = .now(app.io, .awake);
    var draft = try session.draft(
        app.allocator,
        app.io,
        rt.platform,
        rt.models,
        lane.h3,
        rt.shardings,
        rt.progress,
        .{
            .geo = lane.geo,
            .tokens = tokens,
            .layout = lane.packed_run.layout,
            .schedules = lane.packed_run.schedules,
            .seed = seed,
            .warm = rt.warm,
            .bake = lane.bake,
            .taeh3 = lane.taeh3,
            .handoff = lane.ltx.handoff,
            .resident_blocks = rt.resident_blocks,
            .job = &app.job,
        },
    );
    defer draft.deinit(app.allocator);
    const draft_done: std.Io.Timestamp = .now(app.io, .awake);
    const video = try refine.infer(
        app.allocator,
        app.io,
        rt.platform,
        lane.ltx,
        draft.pixels,
        .{
            .prompt = prompt,
            .seed = seed,
            .target_w = lane.target_w,
            .target_h = lane.target_h,
            .out = "output/serve",
            .audio = draft.wav,
            .job = &app.job,
        },
    );
    defer video.deinit(app.allocator);
    const refine_d = draft_done.untilNow(app.io, .awake);
    const infer_d = infer_start.untilNow(app.io, .awake);

    var video_tok: Token = undefined;
    app.io.random(&video_tok);
    var hex_buf: [token_hex_len]u8 = undefined;
    const hex = fmtToken(video_tok, &hex_buf);
    var name_buf: [64]u8 = undefined;
    const rel = std.fmt.bufPrint(&name_buf, "output/serve/{s}.mp4", .{hex}) catch return error.OutOfMemory;
    app.job.set(.remux, 1, 1);
    try refine.remux(app.allocator, app.io, rel, video.nchw, video.frames, video.height, video.width, draft.wav);
    app.job.set(.done, 1, 1);
    log.info("serve {s} infer [{f}] draft [{f}] refine [{f}]", .{ hex, infer_d, infer_start.durationTo(draft_done), refine_d });
    rememberVideo(app, owner, video_tok);
    return .{
        .video = video_tok,
        .infer_ms = nsToMs(infer_d.nanoseconds),
        .draft_ms = nsToMs(infer_start.durationTo(draft_done).nanoseconds),
        .refine_ms = nsToMs(refine_d.nanoseconds),
        .seed = seed,
    };
}

fn sendVideo(app: *App, request: *std.http.Server.Request, guest: Guest, name: []const u8) !void {
    if (parseVideoName(name) == null) {
        try request.respond("bad name\n", .{ .status = .bad_request });
        return;
    }
    var path_buf: [128]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "output/serve/{s}", .{name}) catch {
        try request.respond("bad name\n", .{ .status = .bad_request });
        return;
    };
    const file = std.Io.Dir.cwd().openFile(app.io, path, .{ .mode = .read_only }) catch {
        try request.respond("missing\n", .{ .status = .not_found });
        return;
    };
    defer file.close(app.io);
    const total = try file.length(app.io);
    var file_reader = file.reader(app.io, &.{});
    const bytes = file_reader.interface.readAlloc(app.allocator, total) catch {
        try request.respond("read failed\n", .{ .status = .internal_server_error });
        return;
    };
    defer app.allocator.free(bytes);

    var start: u64 = 0;
    var end: u64 = total;
    var status: std.http.Status = .ok;
    var range_buf: [80]u8 = undefined;
    var content_range: ?[]const u8 = null;
    if (headerValue(request, "range")) |rh| {
        if (parseBytesRange(rh, total)) |r| {
            start = r.start;
            end = r.end;
            status = .partial_content;
            content_range = std.fmt.bufPrint(&range_buf, "bytes {d}-{d}/{d}", .{ start, end - 1, total }) catch null;
        } else {
            const cr = std.fmt.bufPrint(&range_buf, "bytes */{d}", .{total}) catch "bytes */0";
            try request.respond("", .{
                .status = .range_not_satisfiable,
                .extra_headers = &.{
                    .{ .name = "accept-ranges", .value = "bytes" },
                    .{ .name = "content-range", .value = cr },
                },
            });
            return;
        }
    }

    const body = bytes[start..end];
    try respondVideo(request, status, body, content_range, guest.set_cookie);
}

fn readBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) ![]u8 {
    const len = request.head.content_length orelse return error.BadRequest;
    if (len == 0 or len > 64 * 1024) return error.BadRequest;
    var tmp: [512]u8 = undefined;
    const r = if (request.head.expect != null)
        try request.readerExpectContinue(&tmp)
    else
        request.readerExpectNone(&tmp);
    const buf = try allocator.alloc(u8, @intCast(len));
    errdefer allocator.free(buf);
    try r.readSliceAll(buf);
    return buf;
}

const StageView = struct { id: []const u8, title: []const u8 };

fn stageView(stage: session.JobProgress.Stage) StageView {
    return switch (stage) {
        .idle => .{ .id = "idle", .title = "Ready" },
        .text => .{ .id = "text", .title = "Text" },
        .draft => .{ .id = "draft", .title = "Draft" },
        .taeh3 => .{ .id = "decode", .title = "Decode" },
        .vae => .{ .id = "vae", .title = "VAE" },
        .refine => .{ .id = "refine", .title = "Refine" },
        .taehv => .{ .id = "output", .title = "Output" },
        .remux => .{ .id = "encode", .title = "Encode" },
        .done => .{ .id = "encode", .title = "Done" },
    };
}

fn jobPct(stage: session.JobProgress.Stage, step: u32, total: u32) u32 {
    const span: [2]u32 = switch (stage) {
        .idle => .{ 0, 0 },
        .text => .{ 0, 8 },
        .draft => .{ 8, 55 },
        .taeh3 => .{ 55, 64 },
        .vae => .{ 64, 72 },
        .refine => .{ 72, 92 },
        .taehv => .{ 92, 96 },
        .remux => .{ 96, 100 },
        .done => .{ 100, 100 },
    };
    if (total == 0 or step == 0) return span[0];
    const done = @min(step, total);
    return span[0] + (span[1] - span[0]) * done / total;
}

fn replyStatus(app: *App, request: *std.http.Server.Request, guest: Guest) !void {
    var videos: [max_videos]Token = undefined;
    var video_n: u8 = 0;
    var you: []const u8 = "idle";
    var ahead: u32 = 0;
    var queue: u32 = 0;
    var compile_label_buf: [80]u8 = undefined;
    var compile_label: []const u8 = "Compiling";
    var compile_pct: u32 = 0;
    var compile_sku: [sku.skus.len]CompileSku = @splat(.pending);
    const phase: Phase = @enumFromInt(app.phase.load(.acquire));
    const machine_busy = app.busy.load(.seq_cst) or phase != .ready;
    {
        app.meta.lockUncancelable(app.io);
        defer app.meta.unlock(app.io);
        queue = app.waiter_n + @as(u32, if (app.has_running) 1 else 0);
        if (app.has_running and tokenEql(app.running, guest.token)) {
            you = "running";
        } else if (waiterIndex(app, guest.token)) |idx| {
            you = "queued";
            ahead = idx + @as(u32, if (app.has_running) 1 else 0);
        }
        if (findClient(app, guest.token)) |client| {
            video_n = client.video_n;
            @memcpy(videos[0..video_n], client.videos[0..video_n]);
        }
        compile_pct = app.compile_pct;
        compile_sku = app.compile_sku;
        const n = app.compile_label_n;
        @memcpy(compile_label_buf[0..n], app.compile_label[0..n]);
        compile_label = compile_label_buf[0..n];
    }

    const mine = std.mem.eql(u8, you, "running");
    const stage: session.JobProgress.Stage = if (mine)
        @enumFromInt(app.job.stage.load(.acquire))
    else
        .idle;
    const step: u32 = if (mine) app.job.step.load(.acquire) else 0;
    const total: u32 = if (mine) app.job.total.load(.acquire) else 0;
    const view = stageView(stage);
    const pct = if (phase != .ready) compile_pct else if (mine) jobPct(stage, step, total) else 0;
    var label_buf: [32]u8 = undefined;
    const label = if (phase != .ready)
        compile_label
    else if (mine and total > 1)
        std.fmt.bufPrint(&label_buf, "{s} {d}/{d}", .{ view.title, step, total }) catch view.title
    else if (mine)
        view.title
    else
        "Ready";
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{{\"ok\":true,\"ready\":{s},\"phase\":\"{s}\",\"busy\":{s},\"you\":\"{s}\",\"ahead\":{d},\"queue\":{d},\"devices\":{d},\"stage\":\"{s}\",\"step\":{d},\"total\":{d},\"pct\":{d},\"label\":\"{s}\",\"videos\":[", .{
        if (phase == .ready) "true" else "false",
        @tagName(phase),
        if (machine_busy) "true" else "false",
        you,
        ahead,
        queue,
        app.devices,
        if (phase != .ready) "compile" else view.id,
        step,
        total,
        pct,
        label,
    });
    var i: u8 = 0;
    while (i < video_n) : (i += 1) {
        if (i != 0) try w.writeByte(',');
        var hex_buf: [token_hex_len]u8 = undefined;
        try w.print("\"{s}\"", .{fmtToken(videos[i], &hex_buf)});
    }
    try w.writeAll("],\"skus\":[");
    if (app.runtime) |rt| {
        for (rt.lanes, 0..) |lane, li| {
            if (li != 0) try w.writeByte(',');
            try writeSkuJson(&w, lane.id, @intFromFloat(lane.duration_s), lane.target_w, lane.target_h, lane.hd);
        }
    } else {
        var wrote = false;
        for (sku.skus) |row| {
            if (!sku.enabled(row.id)) continue;
            if (wrote) try w.writeByte(',');
            wrote = true;
            try writeSkuJson(&w, row.id, sku.seconds(row), row.target_w, row.target_h, sku.isHd(row));
        }
    }
    try w.writeAll("],\"compile_skus\":[");
    {
        var wrote = false;
        for (sku.skus, 0..) |row, si| {
            if (!sku.enabled(row.id) and compile_sku[si] == .pending) continue;
            if (wrote) try w.writeByte(',');
            wrote = true;
            try w.print("{{\"id\":\"{s}\",\"state\":\"{s}\"}}", .{ row.id, @tagName(compile_sku[si]) });
        }
    }
    try w.writeAll("]}\n");
    try replyJson(request, .ok, w.buffered(), guest.set_cookie);
}

fn replyPage(request: *std.http.Server.Request, set_cookie: ?[]const u8) !void {
    if (set_cookie) |cookie| {
        try request.respond(html, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "text/html; charset=utf-8" },
                .{ .name = "cache-control", .value = "no-store" },
                .{ .name = "set-cookie", .value = cookie },
            },
        });
    } else {
        try request.respond(html, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "text/html; charset=utf-8" },
                .{ .name = "cache-control", .value = "no-store" },
            },
        });
    }
}

fn replyJson(request: *std.http.Server.Request, status: std.http.Status, body: []const u8, set_cookie: ?[]const u8) !void {
    if (set_cookie) |cookie| {
        try request.respond(body, .{
            .status = status,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
                .{ .name = "cache-control", .value = "no-store" },
                .{ .name = "set-cookie", .value = cookie },
            },
        });
    } else {
        try request.respond(body, .{
            .status = status,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
                .{ .name = "cache-control", .value = "no-store" },
            },
        });
    }
}

fn ensureGuest(app: *App, request: *const std.http.Server.Request, cookie_buf: *[96]u8) Guest {
    if (cookieToken(request)) |token| {
        return .{ .token = token, .set_cookie = null };
    }
    var token: Token = undefined;
    app.io.random(&token);
    return .{
        .token = token,
        .set_cookie = writeCookie(token, cookie_buf),
    };
}

fn cookieToken(request: *const std.http.Server.Request) ?Token {
    var it = request.iterateHeaders();
    while (it.next()) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "cookie")) continue;
        var parts = std.mem.splitScalar(u8, h.value, ';');
        while (parts.next()) |part| {
            const kv = std.mem.trim(u8, part, " \t");
            if (std.mem.startsWith(u8, kv, "h3=")) {
                return parseToken(kv["h3=".len..]);
            }
        }
    }
    return null;
}

fn writeCookie(token: Token, buf: *[96]u8) []const u8 {
    var hex_buf: [token_hex_len]u8 = undefined;
    return std.fmt.bufPrint(buf, "h3={s}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000", .{fmtToken(token, &hex_buf)}) catch buf[0..0];
}

fn getOrCreateClient(app: *App, token: Token) *Client {
    if (findClient(app, token)) |client| return client;
    if (freeClient(app)) |slot| {
        slot.* = .{
            .used = true,
            .token = token,
        };
        return slot;
    }
    var n: u32 = 0;
    while (n < max_clients) : (n += 1) {
        const i = (app.evict_i + n) % max_clients;
        const slot = &app.clients[i];
        if (clientBusy(app, slot.token)) continue;
        app.evict_i = (i + 1) % max_clients;
        slot.* = .{
            .used = true,
            .token = token,
        };
        return slot;
    }
    const fallback = &app.clients[app.evict_i % max_clients];
    fallback.* = .{
        .used = true,
        .token = token,
    };
    return fallback;
}

fn findClient(app: *App, token: Token) ?*Client {
    for (&app.clients) |*client| {
        if (client.used and tokenEql(client.token, token)) return client;
    }
    return null;
}

fn freeClient(app: *App) ?*Client {
    for (&app.clients) |*client| {
        if (!client.used) return client;
    }
    return null;
}

fn clientBusy(app: *App, token: Token) bool {
    if (app.has_running and tokenEql(app.running, token)) return true;
    return waiterIndex(app, token) != null;
}

fn waiterIndex(app: *const App, token: Token) ?u32 {
    var i: u32 = 0;
    while (i < app.waiter_n) : (i += 1) {
        if (tokenEql(app.waiters[i], token)) return i;
    }
    return null;
}

fn removeWaiter(app: *App, token: Token) void {
    const idx = waiterIndex(app, token) orelse return;
    var i = idx;
    while (i + 1 < app.waiter_n) : (i += 1) app.waiters[i] = app.waiters[i + 1];
    app.waiter_n -= 1;
}

fn rememberVideo(app: *App, owner: Token, video: Token) void {
    app.meta.lockUncancelable(app.io);
    defer app.meta.unlock(app.io);
    const client = getOrCreateClient(app, owner);
    const n = @min(client.video_n, max_videos - 1);
    var i = n;
    while (i > 0) : (i -= 1) client.videos[i] = client.videos[i - 1];
    client.videos[0] = video;
    if (client.video_n < max_videos) client.video_n += 1;
}

fn tokenEql(a: Token, b: Token) bool {
    return std.mem.eql(u8, &a, &b);
}

fn fmtToken(token: Token, buf: *[token_hex_len]u8) []const u8 {
    const hex = "0123456789abcdef";
    for (token, 0..) |b, i| {
        buf[i * 2] = hex[b >> 4];
        buf[i * 2 + 1] = hex[b & 15];
    }
    return buf;
}

fn parseToken(s: []const u8) ?Token {
    if (s.len != token_hex_len) return null;
    var token: Token = undefined;
    var i: usize = 0;
    while (i < token_len) : (i += 1) {
        const hi = hexVal(s[i * 2]) orelse return null;
        const lo = hexVal(s[i * 2 + 1]) orelse return null;
        token[i] = (hi << 4) | lo;
    }
    return token;
}

fn parseVideoName(name: []const u8) ?Token {
    if (!std.mem.endsWith(u8, name, ".mp4")) return null;
    return parseToken(name[0 .. name.len - 4]);
}

const ByteRange = struct { start: u64, end: u64 };

fn headerValue(request: *const std.http.Server.Request, name: []const u8) ?[]const u8 {
    var it = request.iterateHeaders();
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, name)) return h.value;
    }
    return null;
}

fn parseBytesRange(header: []const u8, total: u64) ?ByteRange {
    if (total == 0 or !std.mem.startsWith(u8, header, "bytes=")) return null;
    const spec = header["bytes=".len..];
    const one = if (std.mem.indexOfScalar(u8, spec, ',')) |i| spec[0..i] else spec;
    const dash = std.mem.indexOfScalar(u8, one, '-') orelse return null;
    const left = std.mem.trim(u8, one[0..dash], " \t");
    const right = std.mem.trim(u8, one[dash + 1 ..], " \t");
    if (left.len == 0) {
        const n = std.fmt.parseInt(u64, right, 10) catch return null;
        if (n == 0) return null;
        const start = if (n >= total) 0 else total - n;
        return .{ .start = start, .end = total };
    }
    const start = std.fmt.parseInt(u64, left, 10) catch return null;
    if (start >= total) return null;
    const end: u64 = if (right.len == 0) total else blk: {
        const last = std.fmt.parseInt(u64, right, 10) catch return null;
        break :blk @min(last + 1, total);
    };
    if (end <= start) return null;
    return .{ .start = start, .end = end };
}

fn respondVideo(
    request: *std.http.Server.Request,
    status: std.http.Status,
    body: []const u8,
    content_range: ?[]const u8,
    set_cookie: ?[]const u8,
) !void {
    if (content_range) |cr| {
        if (set_cookie) |cookie| {
            try request.respond(body, .{
                .status = status,
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "video/mp4" },
                    .{ .name = "accept-ranges", .value = "bytes" },
                    .{ .name = "cache-control", .value = "private, max-age=86400, immutable" },
                    .{ .name = "content-range", .value = cr },
                    .{ .name = "set-cookie", .value = cookie },
                },
            });
        } else {
            try request.respond(body, .{
                .status = status,
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "video/mp4" },
                    .{ .name = "accept-ranges", .value = "bytes" },
                    .{ .name = "cache-control", .value = "private, max-age=86400, immutable" },
                    .{ .name = "content-range", .value = cr },
                },
            });
        }
    } else if (set_cookie) |cookie| {
        try request.respond(body, .{
            .status = status,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "video/mp4" },
                .{ .name = "accept-ranges", .value = "bytes" },
                .{ .name = "cache-control", .value = "private, max-age=86400, immutable" },
                .{ .name = "set-cookie", .value = cookie },
            },
        });
    } else {
        try request.respond(body, .{
            .status = status,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "video/mp4" },
                .{ .name = "accept-ranges", .value = "bytes" },
                .{ .name = "cache-control", .value = "private, max-age=86400, immutable" },
            },
        });
    }
}

fn hexVal(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

fn clipPath(target: []const u8, buf: []u8) ?[]const u8 {
    const raw = if (std.mem.indexOfScalar(u8, target, '?')) |i| target[0..i] else target;
    if (raw.len == 0 or raw.len >= buf.len) return null;
    @memcpy(buf[0..raw.len], raw);
    return buf[0..raw.len];
}

fn padTokens(allocator: std.mem.Allocator, ids: []const u32, n: u32) ![]u32 {
    const out = try allocator.alloc(u32, n);
    @memset(out, 0);
    const keep = @min(ids.len, n);
    if (keep != 0) @memcpy(out[0..keep], ids[0..keep]);
    return out;
}

fn nsToMs(ns: i128) u64 {
    if (ns <= 0) return 0;
    return @intCast(@divTrunc(ns, 1_000_000));
}

fn writeSkuJson(w: *std.Io.Writer, id: []const u8, seconds: u32, width: u32, height: u32, hd: bool) !void {
    try w.print(
        "{{\"id\":\"{s}\",\"seconds\":{d},\"width\":{d},\"height\":{d},\"hd\":{s},\"preferred\":{s}}}",
        .{
            id,
            seconds,
            width,
            height,
            if (hd) "true" else "false",
            if (std.mem.eql(u8, id, sku.default_sku_id)) "true" else "false",
        },
    );
}
