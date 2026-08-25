const std = @import("std");

const config = @import("../core/config.zig");
const draft = @import("draft.zig");
const grid = @import("grid.zig");
const media = @import("../runtime/media.zig");
const repair_mod = @import("repair.zig");
const validate_mod = @import("validate.zig");

const log = std.log.scoped(.minimax_h3_ir);

const max_fix_rounds: u32 = 2;
const thumb_side: u32 = 384;

const compose_base = @embedFile("prompts/compose_base.v1.txt");
const compose_ref = @embedFile("prompts/compose.v2.txt");

const repair_system =
    \\Fix the listed ERRORS in this MiniMax-H3 brief. Keep every required section and every attached media label.
    \\Do not invent media. Output only the corrected brief. No markdown fences, no preface.
;

pub const Creativity = enum {
    restrained,
    balanced,
    bold,
    extreme,

    pub fn temperature(self: Creativity) f32 {
        return switch (self) {
            .restrained => 0.2,
            .balanced => 0.5,
            .bold => 0.8,
            .extreme => 1.1,
        };
    }

    pub fn magnitude(self: Creativity) []const u8 {
        return switch (self) {
            .restrained => "plain",
            .balanced => "measured",
            .bold => "assertive",
            .extreme => "maximal",
        };
    }
};

pub const Asset = struct {
    kind: []const u8,
    path: []const u8,
    role: ?[]const u8 = null,
    paired_video_path: ?[]const u8 = null,
};

pub const Card = struct {
    asset: Asset,
    label: []const u8,
    width: u32 = 0,
    height: u32 = 0,
    seconds: f32 = 0,
};

pub const Finding = validate_mod.Finding;

pub const Request = struct {
    prompt: []const u8,
    variant: config.Variant = .t2va,
    duration_s: f32 = 5.0,
    aspect: []const u8 = "16:9",
    llm_url: ?[]const u8 = null,
    llm_model: ?[]const u8 = null,
    image: []const u8 = "",
    last_image: []const u8 = "",
    refs: []const u8 = "",
    seed: u64 = 0,
    creativity: Creativity = .balanced,
    director: []const u8 = "",
    shots: ?u32 = null,
    silent: bool = false,
    dialogue: []const []const u8 = &.{},
    http: ?*std.http.Client = null,
};

pub const Brief = struct {
    text: []u8,
    source: enum { prompting_guidance, openh3_ir },
    via: enum { none, llm } = .none,

    pub fn deinit(self: Brief, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

const ImagePart = struct { mime: []const u8, b64: []const u8 };

const ChatResponse = struct {
    choices: []const struct {
        message: ?struct { content: ?[]const u8 = null } = null,
        text: ?[]const u8 = null,
    } = &.{},
};

pub fn alreadyCompiled(prompt: []const u8) bool {
    return std.mem.indexOf(u8, prompt, "integrated_multimodal_description:") != null or
        std.mem.indexOf(u8, prompt, "detailed_description:") != null;
}

pub fn hasLlm(req: Request) bool {
    const url = req.llm_url orelse return false;
    return std.mem.trim(u8, url, " \t").len != 0;
}

pub fn effectiveSeconds(duration_s: f32) f32 {
    return grid.effectiveSeconds(duration_s);
}

pub fn shotCount(duration_s: f32, variant: config.Variant) u32 {
    return grid.shotCount(duration_s, variant, null);
}

pub fn checkCapacity(assets: []const Asset) !void {
    if (assets.len > grid.max_ref_files) return error.TooManyRefs;
    var n_img: u32 = 0;
    var n_vid: u32 = 0;
    var n_aud: u32 = 0;
    for (assets) |asset| {
        switch (asset.kind[0]) {
            'i' => n_img += 1,
            'v' => n_vid += 1,
            else => n_aud += 1,
        }
    }
    if (n_img > grid.max_ref_images) return error.TooManyRefImages;
    if (n_vid > grid.max_ref_videos) return error.TooManyRefVideos;
    if (n_aud > grid.max_ref_audios) return error.TooManyRefAudios;
}

pub fn checkRequest(req: Request) !void {
    if (std.mem.trim(u8, req.prompt, " \t\r\n").len == 0) return error.IntentEmpty;
    if (req.duration_s <= 0) return error.DurationInvalid;
    const aspect = std.mem.trim(u8, req.aspect, " \t");
    if (std.mem.indexOfScalar(u8, aspect, ':') == null and std.mem.indexOfAny(u8, aspect, "xX") == null)
        return error.AspectInvalid;
    if (req.shots) |n| {
        if (n < 1 or n > grid.max_shots) return error.ShotsInvalid;
        const floor = @as(f32, @floatFromInt(n)) * @as(f32, @floatFromInt(grid.min_shot_ms)) / 1000.0;
        if (req.duration_s < floor) return error.ShotsDoNotFit;
    }
}

pub fn collectAssets(allocator: std.mem.Allocator, req: Request) ![]Asset {
    var out: std.ArrayList(Asset) = .empty;
    errdefer out.deinit(allocator);
    switch (req.variant) {
        .t2va => {},
        .fl2va => {
            if (req.image.len != 0)
                try out.append(allocator, .{ .kind = "image", .path = req.image, .role = "frame_anchor_first" });
            if (req.last_image.len != 0)
                try out.append(allocator, .{ .kind = "image", .path = req.last_image, .role = "frame_anchor_last" });
        },
        .ref2va => {
            var it = std.mem.splitScalar(u8, req.refs, ',');
            var pending_video: ?[]const u8 = null;
            while (it.next()) |part| {
                const path = std.mem.trim(u8, part, " \t");
                if (path.len == 0) continue;
                switch (media.guessKind(path)) {
                    .image => try out.append(allocator, .{ .kind = "image", .path = path }),
                    .video => {
                        if (pending_video) |prev|
                            try out.append(allocator, .{ .kind = "video", .path = prev });
                        pending_video = path;
                    },
                    .audio, .video_audio => {
                        if (pending_video) |prev| {
                            try out.append(allocator, .{ .kind = "video", .path = prev });
                            try out.append(allocator, .{
                                .kind = "audio",
                                .path = path,
                                .paired_video_path = prev,
                            });
                            pending_video = null;
                        } else {
                            try out.append(allocator, .{ .kind = "audio", .path = path });
                        }
                    },
                }
            }
            if (pending_video) |prev|
                try out.append(allocator, .{ .kind = "video", .path = prev });
        },
    }
    return out.toOwnedSlice(allocator);
}

pub fn labelAssets(allocator: std.mem.Allocator, assets: []const Asset) ![]Card {
    const cards = try allocator.alloc(Card, assets.len);
    @memset(cards, .{ .asset = .{ .kind = "", .path = "" }, .label = "" });
    errdefer {
        for (cards) |card| {
            if (card.label.len != 0) allocator.free(card.label);
        }
        allocator.free(cards);
    }
    var picture: u32 = 0;
    var video: u32 = 0;
    var audio: u32 = 0;
    for (assets, cards) |asset, *card| {
        const n, const word = switch (asset.kind[0]) {
            'i' => blk: {
                picture += 1;
                break :blk .{ picture, "Picture" };
            },
            'v' => blk: {
                video += 1;
                break :blk .{ video, "Video" };
            },
            else => blk: {
                audio += 1;
                break :blk .{ audio, "Audio" };
            },
        };
        card.* = .{
            .asset = asset,
            .label = try std.fmt.allocPrint(allocator, "{s} {d}", .{ word, n }),
        };
    }
    return cards;
}

pub fn freeCards(allocator: std.mem.Allocator, cards: []Card) void {
    for (cards) |card| allocator.free(card.label);
    allocator.free(cards);
}

pub fn measureCards(allocator: std.mem.Allocator, io: std.Io, cards: []Card) !void {
    for (cards) |*card| {
        switch (card.asset.kind[0]) {
            'a' => {
                const n = try media.wavSampleCount(allocator, io, card.asset.path);
                card.seconds = @as(f32, @floatFromInt(n)) / 32000.0;
            },
            'i' => {
                const size = try media.imageSize(allocator, io, card.asset.path);
                card.width = size.w;
                card.height = size.h;
            },
            else => {
                const meta = try media.probeVideo(allocator, io, card.asset.path);
                card.width = meta.w;
                card.height = meta.h;
            },
        }
    }
}

pub fn freeFindings(allocator: std.mem.Allocator, findings: []Finding) void {
    validate_mod.freeFindings(allocator, findings);
}

pub fn countErrors(findings: []const Finding) u32 {
    return validate_mod.countErrors(findings);
}

pub fn instructionLine(allocator: std.mem.Allocator, variant: config.Variant, assets: []const Asset, last_shot: u32, duration_s: f32) ![]u8 {
    var n_pic: u32 = 0;
    var first = false;
    var last = false;
    for (assets) |asset| {
        if (asset.kind[0] != 'i') continue;
        n_pic += 1;
        const role = asset.role orelse "";
        if (std.mem.eql(u8, role, "frame_anchor_first")) first = true;
        if (std.mem.eql(u8, role, "frame_anchor_last")) last = true;
    }
    return grid.instructionLine(allocator, variant, last_shot, duration_s, n_pic, last and !first);
}

pub fn promptingGuidance(allocator: std.mem.Allocator, req: Request) ![]u8 {
    if (alreadyCompiled(req.prompt)) return allocator.dupe(u8, req.prompt);
    return renderDraft(allocator, req);
}

pub fn userMessage(allocator: std.mem.Allocator, req: Request, assets: []const Asset) ![]u8 {
    const cards = try labelAssets(allocator, assets);
    defer freeCards(allocator, cards);
    const shots = grid.shotCount(req.duration_s, req.variant, req.shots);
    const align_line = try instructionLine(allocator, req.variant, assets, shots, req.duration_s);
    defer allocator.free(align_line);
    return writeUser(allocator, req, cards, "", shots, align_line, &.{});
}

pub fn compile(allocator: std.mem.Allocator, io: std.Io, req: Request) !Brief {
    if (alreadyCompiled(req.prompt)) {
        return .{ .text = try allocator.dupe(u8, req.prompt), .source = .prompting_guidance };
    }
    try checkRequest(req);
    const assets = try collectAssets(allocator, req);
    defer allocator.free(assets);
    try checkCapacity(assets);
    const brief: Brief = if (hasLlm(req)) try compileLlm(allocator, io, req) else blk: {
        log.info("ir: no LLM URL, using deterministic draft", .{});
        break :blk try compileDraft(allocator, req);
    };
    log.info("ir source={s} via={s} chars={d} variant={s}", .{
        @tagName(brief.source),
        @tagName(brief.via),
        brief.text.len,
        @tagName(req.variant),
    });
    return brief;
}

pub fn resolveChatUrl(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    const trimmed = std.mem.trimEnd(u8, std.mem.trim(u8, raw, " \t"), "/");
    if (trimmed.len == 0) return error.H3irLlmMissing;
    if (std.mem.endsWith(u8, trimmed, "/chat/completions")) return allocator.dupe(u8, trimmed);
    if (std.mem.endsWith(u8, trimmed, "/v1"))
        return std.fmt.allocPrint(allocator, "{s}/chat/completions", .{trimmed});
    return std.fmt.allocPrint(allocator, "{s}/v1/chat/completions", .{trimmed});
}

pub fn parseChatContent(allocator: std.mem.Allocator, json_text: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(ChatResponse, allocator, json_text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value.choices.len == 0) return error.H3irEmpty;
    const choice = parsed.value.choices[0];
    const raw = if (choice.message) |msg| msg.content orelse choice.text orelse "" else choice.text orelse "";
    const text = repair_mod.stripFences(std.mem.trim(u8, raw, " \t\r\n"));
    if (text.len == 0) return error.H3irEmpty;
    return allocator.dupe(u8, text);
}

pub fn validate(allocator: std.mem.Allocator, text: []const u8, variant: config.Variant, cards: []const Card, duration_s: f32) ![]Finding {
    var role_buf: [16]validate_mod.RoleDecl = undefined;
    return validate_mod.validate(allocator, text, contextFrom(variant, cards, duration_s, &role_buf));
}

fn compileDraft(allocator: std.mem.Allocator, req: Request) !Brief {
    const text = try renderDraft(allocator, req);
    errdefer allocator.free(text);
    const assets = try collectAssets(allocator, req);
    defer allocator.free(assets);
    const cards = try labelAssets(allocator, assets);
    defer freeCards(allocator, cards);
    const have = countsFromCards(cards);
    var role_buf: [16]validate_mod.RoleDecl = undefined;
    const findings = try validate_mod.validate(allocator, text, .{
        .variant = req.variant,
        .duration_s = req.duration_s,
        .n_pictures = have.picture,
        .n_videos = have.video,
        .n_audios = have.audio,
        .declared_roles = rolesFromCards(cards, &role_buf),
        .creativity = @tagName(req.creativity),
        .pinned_shots = req.shots,
        .dialogue = req.dialogue,
        .forbids_score = req.silent,
    });
    defer freeFindings(allocator, findings);
    if (countErrors(findings) != 0) {
        log.err("ir draft failed its own validator: {s}", .{findings[0].code});
        return error.CompilerInvariant;
    }
    return .{ .text = text, .source = .prompting_guidance };
}

fn renderDraft(allocator: std.mem.Allocator, req: Request) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const assets = try collectAssets(a, req);
    const draft_assets = try a.alloc(draft.DraftAsset, assets.len);
    for (assets, draft_assets) |asset, *da| {
        da.* = .{
            .kind = asset.kind,
            .path = asset.path,
            .role = asset.role,
            .paired_video_path = asset.paired_video_path,
        };
    }
    const plan = try draft.buildPlan(
        a,
        req.variant,
        req.duration_s,
        req.prompt,
        draft_assets,
        "",
        req.creativity.magnitude(),
        req.shots,
    );
    const text = try draft.render(a, plan);
    return allocator.dupe(u8, text);
}

fn compileLlm(allocator: std.mem.Allocator, io: std.Io, req: Request) !Brief {
    const client = req.http orelse return error.H3irHttpMissing;
    const raw_url = req.llm_url orelse return error.H3irLlmMissing;
    const url = try resolveChatUrl(allocator, raw_url);
    defer allocator.free(url);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const assets = try collectAssets(a, req);
    try checkCapacity(assets);
    const cards = try labelAssets(a, assets);
    try measureCards(a, io, cards);
    const thumbs = try loadThumbs(a, io, cards);
    const draft_text = try renderDraft(a, req);
    const draft_findings = try validate(a, draft_text, req.variant, cards, req.duration_s);
    if (countErrors(draft_findings) != 0) return error.CompilerInvariant;

    const shots = grid.shotCount(req.duration_s, req.variant, req.shots);
    const align_line = try instructionLine(a, req.variant, assets, shots, req.duration_s);
    const tasks = try taskTypesOf(a, assets);
    const user = try writeUser(a, req, cards, draft_text, shots, align_line, tasks);
    const system = if (req.variant == .ref2va) compose_ref else compose_base;
    const model = modelName(req);
    log.info("ir llm: compose shots={d} cards={d} thumbs={d} model={s}", .{ shots, cards.len, thumbs.len, model });

    var text = try chat(allocator, client, url, model, req, system, user, thumbs, req.creativity.temperature());
    errdefer allocator.free(text);

    const counts = countsFromCards(cards);
    var last_only = false;
    var first = false;
    for (assets) |asset| {
        const role = asset.role orelse "";
        if (std.mem.eql(u8, role, "frame_anchor_first")) first = true;
        if (std.mem.eql(u8, role, "frame_anchor_last")) last_only = true;
    }
    last_only = last_only and !first;

    var rounds: u32 = 0;
    while (true) {
        const repaired = try repair_mod.repair(allocator, text, req.variant, counts.picture, counts.video, counts.audio, last_only, req.duration_s, tasks, req.dialogue);
        allocator.free(text);
        text = repaired;
        const findings = try validate(allocator, text, req.variant, cards, req.duration_s);
        defer freeFindings(allocator, findings);
        const errors = countErrors(findings);
        if (errors == 0) break;
        if (rounds >= max_fix_rounds) {
            log.err("ir validate: {d} error(s) after {d} repair round(s)", .{ errors, rounds });
            return error.CompilerInvariant;
        }
        rounds += 1;
        log.info("ir repair: round={d} errors={d}", .{ rounds, errors });
        const fix_user = try repairUser(a, text, findings);
        const fixed = try chat(allocator, client, url, model, req, repair_system, fix_user, &.{}, 0.2);
        allocator.free(text);
        text = fixed;
    }
    return .{ .text = text, .source = .openh3_ir, .via = .llm };
}

fn writeUser(
    allocator: std.mem.Allocator,
    req: Request,
    cards: []const Card,
    draft_text: []const u8,
    shots: u32,
    align_line: []const u8,
    tasks: []const []const u8,
) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("intent: {s}\n", .{std.mem.trim(u8, req.prompt, " \t\r\n")});
    try w.print("seconds: {d:.3}\n", .{effectiveSeconds(req.duration_s)});
    try w.print("aspect: {s}\n", .{req.aspect});
    try w.print("variant: {s}\n", .{@tagName(req.variant)});
    try w.print("creativity: {s}\n", .{@tagName(req.creativity)});
    try w.print("magnitude: {s}\n", .{req.creativity.magnitude()});
    try w.print("shots: {d}\n", .{shots});
    if (req.director.len != 0) try w.print("director: {s}\n", .{req.director});
    if (req.seed != 0) try w.print("seed: {d}\n", .{req.seed});
    if (req.silent) try w.writeAll("silent: true\n");
    if (tasks.len != 0) {
        try w.writeAll("task_types: ");
        for (tasks, 0..) |t, i| {
            if (i != 0) try w.writeAll(" + ");
            try w.writeAll(t);
        }
        try w.writeAll("\n");
    }
    if (align_line.len != 0) {
        try w.writeAll("instruction_line:\n");
        try w.writeAll(align_line);
        try w.writeAll("\n");
    }
    try w.writeAll("assets:\n");
    if (cards.len == 0) {
        try w.writeAll("- none\n");
    } else {
        for (cards) |card| {
            try w.print("- {s}: {s}", .{ card.label, card.asset.kind });
            if (card.asset.role) |role| try w.print(" role={s}", .{role});
            if (card.asset.paired_video_path != null) try w.writeAll(" paired_to_previous_video");
            if (card.width != 0) try w.print(" {d}x{d}", .{ card.width, card.height });
            if (card.seconds != 0) try w.print(" {d:.2}s", .{card.seconds});
            try w.writeAll("\n");
        }
    }
    if (draft_text.len != 0) {
        try w.writeAll("\ndeterministic_draft:\n");
        try w.writeAll(draft_text);
        if (draft_text[draft_text.len - 1] != '\n') try w.writeAll("\n");
    }
    return aw.toOwnedSlice();
}

fn loadThumbs(allocator: std.mem.Allocator, io: std.Io, cards: []const Card) ![]ImagePart {
    var out: std.ArrayList(ImagePart) = .empty;
    errdefer out.deinit(allocator);
    for (cards) |card| {
        if (card.asset.kind[0] == 'a') continue;
        const jpeg = try media.loadJpegThumb(allocator, io, card.asset.path, thumb_side);
        defer allocator.free(jpeg);
        const Encoder = std.base64.standard.Encoder;
        const b64 = try allocator.alloc(u8, Encoder.calcSize(jpeg.len));
        _ = Encoder.encode(b64, jpeg);
        try out.append(allocator, .{ .mime = "image/jpeg", .b64 = b64 });
    }
    return out.toOwnedSlice(allocator);
}

fn modelName(req: Request) []const u8 {
    const named = req.llm_model orelse return "default";
    const trimmed = std.mem.trim(u8, named, " \t");
    return if (trimmed.len == 0) "default" else trimmed;
}

fn chat(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    model: []const u8,
    req: Request,
    system: []const u8,
    user: []const u8,
    images: []const ImagePart,
    temperature: f32,
) ![]u8 {
    const body = try stringifyChat(allocator, model, system, user, images, temperature, req.seed);
    defer allocator.free(body);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .{ .override = "identity" },
        },
        .response_writer = &aw.writer,
    }) catch return error.H3irLlmFailed;
    const raw = aw.written();
    if (result.status != .ok) {
        const preview = raw[0..@min(raw.len, 400)];
        log.err("ir llm: HTTP {s}: {s}", .{ @tagName(result.status), preview });
        return error.H3irLlmFailed;
    }
    return parseChatContent(allocator, raw);
}

fn stringifyChat(
    allocator: std.mem.Allocator,
    model: []const u8,
    system: []const u8,
    user: []const u8,
    images: []const ImagePart,
    temperature: f32,
    seed: u64,
) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var jw: std.json.Stringify = .{ .writer = &aw.writer };
    try jw.beginObject();
    try jw.objectField("model");
    try jw.write(model);
    try jw.objectField("temperature");
    try jw.write(temperature);
    if (seed != 0) {
        try jw.objectField("seed");
        try jw.write(seed);
    }
    try jw.objectField("messages");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("role");
    try jw.write("system");
    try jw.objectField("content");
    try jw.write(system);
    try jw.endObject();
    try jw.beginObject();
    try jw.objectField("role");
    try jw.write("user");
    try jw.objectField("content");
    if (images.len == 0) {
        try jw.write(user);
    } else {
        try jw.beginArray();
        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("text");
        try jw.objectField("text");
        try jw.write(user);
        try jw.endObject();
        for (images) |img| {
            const data = try std.fmt.allocPrint(allocator, "data:{s};base64,{s}", .{ img.mime, img.b64 });
            defer allocator.free(data);
            try jw.beginObject();
            try jw.objectField("type");
            try jw.write("image_url");
            try jw.objectField("image_url");
            try jw.beginObject();
            try jw.objectField("url");
            try jw.write(data);
            try jw.endObject();
            try jw.endObject();
        }
        try jw.endArray();
    }
    try jw.endObject();
    try jw.endArray();
    try jw.endObject();
    return aw.toOwnedSlice();
}

fn repairUser(allocator: std.mem.Allocator, text: []const u8, findings: []const Finding) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("errors:\n");
    for (findings) |finding| {
        if (!finding.err) continue;
        try w.print("- {s}: {s}\n", .{ finding.code, finding.msg });
    }
    try w.writeAll("\nbrief:\n");
    try w.writeAll(text);
    return aw.toOwnedSlice();
}

fn taskTypesOf(allocator: std.mem.Allocator, assets: []const Asset) ![]const []const u8 {
    const draft_assets = try allocator.alloc(draft.DraftAsset, assets.len);
    for (assets, draft_assets) |asset, *da| {
        da.* = .{
            .kind = asset.kind,
            .path = asset.path,
            .role = asset.role,
            .paired_video_path = asset.paired_video_path,
        };
    }
    const manifest = try draft.buildManifest(allocator, draft_assets);
    return draft.deriveTaskTypes(allocator, manifest);
}

fn countsFromCards(cards: []const Card) struct { picture: u32, video: u32, audio: u32 } {
    var picture: u32 = 0;
    var video: u32 = 0;
    var audio: u32 = 0;
    for (cards) |card| {
        switch (card.asset.kind[0]) {
            'i' => picture += 1,
            'v' => video += 1,
            else => audio += 1,
        }
    }
    return .{ .picture = picture, .video = video, .audio = audio };
}

fn rolesFromCards(cards: []const Card, buf: []validate_mod.RoleDecl) []validate_mod.RoleDecl {
    var n: usize = 0;
    for (cards) |card| {
        const role = card.asset.role orelse continue;
        if (n == buf.len) break;
        buf[n] = .{ .label = card.label, .role = role };
        n += 1;
    }
    return buf[0..n];
}

fn contextFrom(variant: config.Variant, cards: []const Card, duration_s: f32, role_buf: []validate_mod.RoleDecl) validate_mod.Context {
    const have = countsFromCards(cards);
    return .{
        .variant = variant,
        .duration_s = duration_s,
        .n_pictures = have.picture,
        .n_videos = have.video,
        .n_audios = have.audio,
        .declared_roles = rolesFromCards(cards, role_buf),
    };
}
