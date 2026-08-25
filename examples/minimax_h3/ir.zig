const std = @import("std");

const config = @import("config.zig");
const media = @import("media.zig");

const log = std.log.scoped(.minimax_h3_ir);

const max_ref_files: u32 = 12;
const max_ref_images: u32 = 9;
const max_ref_videos: u32 = 3;
const max_ref_audios: u32 = 3;
const max_fix_rounds: u32 = 2;
const thumb_side: u32 = 384;

const base_sections = [_][]const u8{
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
};
const ref_sections = [_][]const u8{
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
};

const compose_system =
    \\You write MiniMax-H3 Prompting Guidance. Output only the brief. No markdown fences, no preface.
    \\
    \\T2VA and FL2VA fields, in this order:
    \\integrated_multimodal_description:
    \\overall_soundscape:
    \\non_diegetic_music:
    \\
    \\Ref2VA fields, in this order, and the brief must begin with subject_definitions:
    \\subject_definitions:
    \\summary:
    \\retention_analysis:
    \\detailed_description:
    \\overall_soundscape:
    \\non_diegetic_music:
    \\
    \\Rules:
    \\- Use only the fields for the stated variant. Do not mix the other mode's description field.
    \\- FL2VA must open with the supplied alignment line, then one blank line, then the fields.
    \\- FL2VA cites pictures bare (Picture 1). Ref2VA cites <Picture N>, <Video N>, <Audio N>.
    \\- Name only attached media. Do not invent references. Duration and aspect are fixed.
    \\- overall_soundscape covers the full clip. non_diegetic_music is N/A unless the intent asks for score.
    \\- Every required field has a real sentence. Images attached to this request are the references, in label order.
;

const beat_system =
    \\Write a beat sheet for MiniMax-H3. One line per shot: Shot N (start–end): action. Camera: ... Sound: ...
    \\No markdown, no preface, no extra shots.
;

const repair_system =
    \\Fix the listed ERRORS in this MiniMax-H3 brief. Keep every required section and every attached media label.
    \\Output only the corrected brief. No markdown fences, no preface.
;

pub const Mode = enum {
    off,
    prompt,
    h3ir,
    auto,

    pub fn parse(text: []const u8) ?Mode {
        return std.meta.stringToEnum(Mode, text);
    }
};

pub const Creativity = enum {
    restrained,
    balanced,
    bold,
    extreme,

    pub fn parse(text: []const u8) ?Creativity {
        return std.meta.stringToEnum(Creativity, text);
    }

    pub fn temperature(self: Creativity) f32 {
        return switch (self) {
            .restrained => 0.2,
            .balanced => 0.5,
            .bold => 0.8,
            .extreme => 1.1,
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

pub const Finding = struct {
    code: []const u8,
    err: bool,
    msg: []const u8,
};

pub const Request = struct {
    prompt: []const u8,
    variant: config.Variant = .t2va,
    duration_s: f32 = 5.0,
    aspect: []const u8 = "16:9",
    mode: Mode = .auto,
    llm_url: ?[]const u8 = null,
    llm_model: ?[]const u8 = null,
    image: []const u8 = "",
    last_image: []const u8 = "",
    refs: []const u8 = "",
    seed: u64 = 0,
    creativity: Creativity = .balanced,
    director: []const u8 = "",
    http: ?*std.http.Client = null,
};

pub const Brief = struct {
    text: []u8,
    source: enum { raw, prompting_guidance, openh3_ir },
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
    const frames = config.alignFrameCount(config.frameCount(duration_s));
    return @as(f32, @floatFromInt(frames)) / 24.0;
}

pub fn shotCount(duration_s: f32, creativity: Creativity) u32 {
    if (duration_s < 8.0) return 1;
    if (duration_s < 12.0) return if (creativity == .extreme) 3 else 2;
    return 3;
}

pub fn checkCapacity(assets: []const Asset) !void {
    if (assets.len > max_ref_files) return error.TooManyRefs;
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
    if (n_img > max_ref_images) return error.TooManyRefImages;
    if (n_vid > max_ref_videos) return error.TooManyRefVideos;
    if (n_aud > max_ref_audios) return error.TooManyRefAudios;
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

pub fn measureCards(allocator: std.mem.Allocator, io: std.Io, cards: []Card) void {
    for (cards) |*card| {
        if (card.asset.kind[0] == 'a') {
            const n = media.wavSampleCount(allocator, io, card.asset.path) catch continue;
            card.seconds = @as(f32, @floatFromInt(n)) / 32000.0;
        } else {
            const size = media.imageSize(allocator, io, card.asset.path) catch continue;
            card.width = size.w;
            card.height = size.h;
        }
    }
}

pub fn freeFindings(allocator: std.mem.Allocator, findings: []Finding) void {
    for (findings) |finding| allocator.free(finding.msg);
    allocator.free(findings);
}

pub fn countErrors(findings: []const Finding) u32 {
    var n: u32 = 0;
    for (findings) |finding| {
        if (finding.err) n += 1;
    }
    return n;
}

pub fn instructionLine(allocator: std.mem.Allocator, variant: config.Variant, assets: []const Asset, last_shot: u32, duration_s: f32) ![]u8 {
    if (variant != .fl2va) return allocator.dupe(u8, "");
    const s_ss = effectiveSeconds(duration_s);
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
    if (n_pic == 1 and last and !first) {
        return std.fmt.allocPrint(allocator, "How the reference pictures align with the target video — Picture 1 (from Shot {d}) aligns with the {d:.2}-second mark of the target video.", .{ last_shot, s_ss });
    }
    if (n_pic == 1) {
        return allocator.dupe(u8, "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video.");
    }
    return std.fmt.allocPrint(allocator, "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {d}) aligns with the {d:.2}-second mark of the target video.", .{ last_shot, s_ss });
}

pub fn promptingGuidance(allocator: std.mem.Allocator, req: Request) ![]u8 {
    if (alreadyCompiled(req.prompt)) return allocator.dupe(u8, req.prompt);
    const assets = try collectAssets(allocator, req);
    defer allocator.free(assets);
    return renderGuidance(allocator, req, assets);
}

pub fn userMessage(allocator: std.mem.Allocator, req: Request, assets: []const Asset) ![]u8 {
    const cards = try labelAssets(allocator, assets);
    defer freeCards(allocator, cards);
    return writeUser(allocator, req, cards, "", shotCount(req.duration_s, req.creativity), null);
}

pub fn compile(allocator: std.mem.Allocator, io: std.Io, req: Request) !Brief {
    if (req.mode != .off and alreadyCompiled(req.prompt)) {
        return .{ .text = try allocator.dupe(u8, req.prompt), .source = .prompting_guidance };
    }
    const mode = switch (req.mode) {
        .auto => if (hasLlm(req)) Mode.h3ir else Mode.prompt,
        else => req.mode,
    };
    if (mode != .off) {
        const assets = try collectAssets(allocator, req);
        defer allocator.free(assets);
        try checkCapacity(assets);
    }
    if (req.mode == .auto and mode != .h3ir) {
        log.info("ir auto: no LLM URL, using prompting guidance", .{});
    }
    const brief: Brief = switch (mode) {
        .off => .{ .text = try allocator.dupe(u8, req.prompt), .source = .raw },
        .prompt => .{ .text = try promptingGuidance(allocator, req), .source = .prompting_guidance },
        .h3ir => compileLlm(allocator, io, req, req.mode == .auto) catch |err| switch (err) {
            error.TooManyRefs, error.TooManyRefImages, error.TooManyRefVideos, error.TooManyRefAudios => return err,
            else => if (req.mode == .auto) blk: {
                log.warn("ir llm failed ({s}); using prompting guidance", .{@errorName(err)});
                break :blk .{ .text = try promptingGuidance(allocator, req), .source = .prompting_guidance };
            } else return err,
        },
        .auto => unreachable,
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
    const text = stripFences(std.mem.trim(u8, raw, " \t\r\n"));
    if (text.len == 0) return error.H3irEmpty;
    return allocator.dupe(u8, text);
}

pub fn validate(allocator: std.mem.Allocator, text: []const u8, variant: config.Variant, cards: []const Card, duration_s: f32) ![]Finding {
    var out: std.ArrayList(Finding) = .empty;
    errdefer {
        for (out.items) |finding| allocator.free(finding.msg);
        out.deinit(allocator);
    }
    const names: []const []const u8 = if (variant == .ref2va) &ref_sections else &base_sections;
    const main_field = if (variant == .ref2va) "detailed_description" else "integrated_multimodal_description";
    const wrong_field = if (variant == .ref2va) "integrated_multimodal_description" else "detailed_description";

    var last_pos: usize = 0;
    var seen_section = false;
    var order_ok = true;
    for (names) |name| {
        if (headerPos(text, name)) |pos| {
            if (seen_section and pos < last_pos) order_ok = false;
            last_pos = pos;
            seen_section = true;
        } else {
            try addFinding(allocator, &out, "S1-missing-section", true, "'{s}:' is absent", .{name});
        }
    }
    if (!order_ok) {
        try addFinding(allocator, &out, "S2-section-order", true, "sections out of mandated order", .{});
    }
    const body_start = std.mem.trimStart(u8, text, " \t\r\n");
    if (variant == .ref2va and !std.mem.startsWith(u8, body_start, "subject_definitions:")) {
        try addFinding(allocator, &out, "S3-preamble", true, "brief must begin with 'subject_definitions:'", .{});
    }
    if (std.mem.indexOf(u8, text, "```") != null) {
        try addFinding(allocator, &out, "S4-code-fence", true, "output contains a markdown code fence", .{});
    }
    if (headerPos(text, wrong_field) != null) {
        try addFinding(allocator, &out, "S5-mode-field-crossed", true, "'{s}' is the other mode's field; {s} uses '{s}'", .{ wrong_field, @tagName(variant), main_field });
    }
    for (names) |name| {
        const body = sectionBody(text, name, names) orelse continue;
        if (std.mem.trim(u8, body, " \t\r\n").len == 0) {
            try addFinding(allocator, &out, "S9-section-empty", true, "'{s}:' is present but empty", .{name});
        }
    }

    if (variant == .fl2va) {
        const assets = try cardsToAssets(allocator, cards);
        defer allocator.free(assets);
        const want = try instructionLine(allocator, .fl2va, assets, lastShotIn(text), duration_s);
        defer allocator.free(want);
        const lines = firstLines(text);
        if (!std.mem.startsWith(u8, lines.first, "How the reference pictures align with the target video")) {
            try addFinding(allocator, &out, "I1-instruction-line-missing", true, "FL2VA brief must open with the alignment instruction", .{});
        } else {
            if (!std.mem.eql(u8, lines.first, want)) {
                try addFinding(allocator, &out, "I2-instruction-line-not-exact", true, "instruction line differs from the mandated one", .{});
            }
            if (lines.second_present and lines.second.len != 0) {
                try addFinding(allocator, &out, "I3-instruction-line-no-blank-line", true, "the instruction line must be followed by one blank line", .{});
            }
        }
    }

    var used = Used{};
    var unknown: std.ArrayList([]const u8) = .empty;
    defer unknown.deinit(allocator);
    try scanLabels(allocator, text, variant == .fl2va, &used, &unknown);
    for (unknown.items) |kind| {
        try addFinding(allocator, &out, "L1-unknown-label", true, "<{s} N> is not in the label vocabulary", .{kind});
    }
    const have = countsFromCards(cards);
    try mediaFindings(allocator, &out, "Picture", have.picture, used.picture);
    try mediaFindings(allocator, &out, "Video", have.video, used.video);
    try mediaFindings(allocator, &out, "Audio", have.audio, used.audio);
    return out.toOwnedSlice(allocator);
}

fn compileLlm(allocator: std.mem.Allocator, io: std.Io, req: Request, fallback: bool) !Brief {
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
    measureCards(a, io, cards);
    const thumbs = try loadThumbs(a, io, cards);
    const shots = shotCount(req.duration_s, req.creativity);
    const model = modelName(req);
    const beats = if (shots > 1)
        chat(a, client, url, model, req, beat_system, try beatUser(a, req, cards, shots), &.{}, 0.2) catch try deterministicBeats(a, req, shots)
    else
        try deterministicBeats(a, req, shots);
    const align_line = try instructionLine(a, req.variant, assets, shots, req.duration_s);
    const user = try writeUser(a, req, cards, beats, shots, align_line);
    log.info("ir llm: compose shots={d} cards={d} thumbs={d} model={s}", .{ shots, cards.len, thumbs.len, model });
    var text = chat(allocator, client, url, model, req, compose_system, user, thumbs, req.creativity.temperature()) catch |err| {
        if (fallback) {
            log.warn("ir compose failed ({s}); using prompting guidance", .{@errorName(err)});
            return .{ .text = try promptingGuidance(allocator, req), .source = .prompting_guidance };
        }
        return err;
    };
    errdefer allocator.free(text);

    var rounds: u32 = 0;
    while (true) {
        const repaired = try mechanicalRepair(allocator, text, req.variant, assets, req.duration_s);
        if (repaired.ptr != text.ptr) {
            allocator.free(text);
            text = repaired;
        } else {
            allocator.free(repaired);
        }
        const findings = try validate(allocator, text, req.variant, cards, req.duration_s);
        defer freeFindings(allocator, findings);
        const errors = countErrors(findings);
        if (errors == 0) break;
        if (rounds >= max_fix_rounds) {
            log.warn("ir validate: {d} error(s) after {d} repair round(s)", .{ errors, rounds });
            if (!alreadyCompiled(text)) {
                const wrapped = try promptingGuidance(allocator, req);
                allocator.free(text);
                return .{ .text = wrapped, .source = .prompting_guidance, .via = .llm };
            }
            break;
        }
        rounds += 1;
        log.info("ir repair: round={d} errors={d}", .{ rounds, errors });
        const fix_user = try repairUser(a, text, findings);
        const fixed = chat(allocator, client, url, model, req, repair_system, fix_user, &.{}, 0.2) catch break;
        allocator.free(text);
        text = fixed;
    }

    if (!alreadyCompiled(text)) {
        var wrap_req = req;
        wrap_req.prompt = text;
        const wrapped = try promptingGuidance(allocator, wrap_req);
        allocator.free(text);
        text = wrapped;
    }
    return .{ .text = text, .source = .openh3_ir, .via = .llm };
}

fn renderGuidance(allocator: std.mem.Allocator, req: Request, assets: []const Asset) ![]u8 {
    const body = std.mem.trim(u8, req.prompt, " \t\r\n");
    const duration = effectiveSeconds(req.duration_s);
    const shots = shotCount(req.duration_s, req.creativity);
    return switch (req.variant) {
        .t2va => std.fmt.allocPrint(allocator,
            \\integrated_multimodal_description: [Shot 1] Live-action, cinematic, {s}
            \\
            \\overall_soundscape: Ambient sound follows the scene described above for the full {d:.2}-second clip, including room tone and physical action.
            \\
            \\non_diegetic_music: N/A
            \\
        , .{ body, duration }),
        .fl2va => renderFl2va(allocator, body, req.duration_s, duration, assets, shots),
        .ref2va => renderRef2va(allocator, body, duration, assets),
    };
}

fn renderFl2va(allocator: std.mem.Allocator, body: []const u8, duration_s: f32, duration: f32, assets: []const Asset, shots: u32) ![]u8 {
    const line = try instructionLine(allocator, .fl2va, assets, shots, duration_s);
    defer allocator.free(line);
    var n_pic: u32 = 0;
    var last_only = false;
    for (assets) |asset| {
        if (asset.kind[0] != 'i') continue;
        n_pic += 1;
        last_only = std.mem.eql(u8, asset.role orelse "", "frame_anchor_last");
    }
    const develop = if (n_pic == 1 and last_only)
        try std.fmt.allocPrint(allocator, "the shot develops continuously until the composition of Picture 1 at {d:.2} seconds", .{duration})
    else if (n_pic == 1)
        try allocator.dupe(u8, "the opening matches Picture 1 and develops continuously")
    else
        try std.fmt.allocPrint(allocator, "the opening matches Picture 1 and develops continuously until the composition of Picture 2 at {d:.2} seconds", .{duration});
    defer allocator.free(develop);
    return std.fmt.allocPrint(allocator,
        \\{s}
        \\
        \\integrated_multimodal_description: [Shot 1] Live-action, cinematic, {s}. {s}
        \\
        \\overall_soundscape: Ambient sound follows the scene described above for the full {d:.2}-second clip, including room tone and physical action.
        \\
        \\non_diegetic_music: N/A
        \\
    , .{ line, develop, body, duration });
}

fn renderRef2va(allocator: std.mem.Allocator, body: []const u8, duration: f32, assets: []const Asset) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    const cards = try labelAssets(allocator, assets);
    defer freeCards(allocator, cards);
    try w.writeAll("subject_definitions:\n");
    if (cards.len == 0) {
        try w.writeAll("<Picture 1> is the first supplied reference used for appearance and style of the target video.\n");
    } else {
        for (cards) |card| {
            try w.print("<{s}> is the attached {s}", .{ card.label, card.asset.kind });
            if (card.asset.role) |role| try w.print(" ({s})", .{role});
            if (card.asset.paired_video_path != null) try w.writeAll(" soundtrack");
            try w.writeAll(" used for appearance and style of the target video.\n");
        }
    }
    try w.print("\nsummary:\n[reference generation] The target video follows the request: {s}\n", .{body});
    try w.writeAll("\nretention_analysis:\n");
    if (cards.len == 0) {
        try w.writeAll("<Picture 1> (appears in [Shot 1]): fully_preserved — appearance, clothing, and visual style stay consistent.\n");
    } else {
        for (cards) |card| {
            try w.print("<{s}> (appears in [Shot 1]): fully_preserved — appearance and style stay consistent.\n", .{card.label});
        }
    }
    try w.print("\ndetailed_description:\n[Shot 1] Live-action, cinematic, {s}", .{body});
    if (cards.len == 0) {
        try w.writeAll(" The shot follows <Picture 1>.");
    } else {
        for (cards) |card| try w.print(" The shot follows <{s}>.", .{card.label});
    }
    try w.print(
        \\
        \\
        \\overall_soundscape: Ambient sound follows the scene described above for the full {d:.2}-second clip, including room tone and physical action.
        \\
        \\non_diegetic_music: N/A
        \\
    , .{duration});
    return aw.toOwnedSlice();
}

fn writeUser(allocator: std.mem.Allocator, req: Request, cards: []const Card, beats: []const u8, shots: u32, align_line: ?[]const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("intent: {s}\n", .{std.mem.trim(u8, req.prompt, " \t\r\n")});
    try w.print("seconds: {d:.2}\n", .{effectiveSeconds(req.duration_s)});
    try w.print("aspect: {s}\n", .{req.aspect});
    try w.print("variant: {s}\n", .{@tagName(req.variant)});
    try w.print("creativity: {s}\n", .{@tagName(req.creativity)});
    try w.print("shots: {d}\n", .{shots});
    if (req.director.len != 0) try w.print("director: {s}\n", .{req.director});
    if (req.seed != 0) try w.print("seed: {d}\n", .{req.seed});
    if (align_line) |line| {
        if (line.len != 0) {
            try w.writeAll("instruction_line:\n");
            try w.writeAll(line);
            try w.writeAll("\n");
        }
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
    if (beats.len != 0) {
        try w.writeAll("beats:\n");
        try w.writeAll(beats);
        if (beats[beats.len - 1] != '\n') try w.writeAll("\n");
    }
    return aw.toOwnedSlice();
}

fn beatUser(allocator: std.mem.Allocator, req: Request, cards: []const Card, shots: u32) ![]u8 {
    return writeUser(allocator, req, cards, "", shots, null);
}

fn deterministicBeats(allocator: std.mem.Allocator, req: Request, shots: u32) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    const total = effectiveSeconds(req.duration_s);
    const intent = std.mem.trim(u8, req.prompt, " \t\r\n");
    var i: u32 = 0;
    while (i < shots) : (i += 1) {
        const start = total * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shots));
        const end = total * @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(shots));
        try w.print("Shot {d} ({d:.2}–{d:.2}s): {s}. Camera: continuous medium. Sound: scene-sync ambient.\n", .{ i + 1, start, end, intent });
    }
    return aw.toOwnedSlice();
}

fn loadThumbs(allocator: std.mem.Allocator, io: std.Io, cards: []const Card) ![]ImagePart {
    var out: std.ArrayList(ImagePart) = .empty;
    errdefer out.deinit(allocator);
    for (cards) |card| {
        if (card.asset.kind[0] == 'a') continue;
        const jpeg = media.loadJpegThumb(allocator, io, card.asset.path, thumb_side) catch |err| {
            log.warn("ir analyse: thumb failed for {s} ({s})", .{ card.label, @errorName(err) });
            continue;
        };
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

fn mechanicalRepair(allocator: std.mem.Allocator, text: []const u8, variant: config.Variant, assets: []const Asset, duration_s: f32) ![]u8 {
    const cur = stripFences(std.mem.trim(u8, text, " \t\r\n"));
    if (variant != .fl2va) return allocator.dupe(u8, cur);
    const want = try instructionLine(allocator, .fl2va, assets, lastShotIn(cur), duration_s);
    defer allocator.free(want);
    const lines = firstLines(cur);
    if (!std.mem.startsWith(u8, lines.first, "How the reference pictures align with the target video")) {
        return std.fmt.allocPrint(allocator, "{s}\n\n{s}\n", .{ want, cur });
    }
    if (!std.mem.eql(u8, lines.first, want) or (lines.second_present and lines.second.len != 0)) {
        const rest = afterFirstLine(cur);
        const body = if (rest.len != 0 and rest[0] == '\n') rest[1..] else rest;
        return std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ want, std.mem.trimStart(u8, body, "\r\n") });
    }
    return allocator.dupe(u8, cur);
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

fn mediaFindings(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), kind: []const u8, n_have: u32, bits: u16) !void {
    var over = false;
    var n: u32 = 1;
    while (n <= 15) : (n += 1) {
        if (hasBit(bits, n) and n > n_have) over = true;
    }
    if (over) {
        try addFinding(allocator, out, "L3-phantom-media", true, "a {s} is cited that was not attached", .{kind});
    }
    if (n_have == 0) return;
    var missing: u32 = 0;
    n = 1;
    while (n <= n_have) : (n += 1) {
        if (!hasBit(bits, n)) missing += 1;
    }
    if (missing != 0) {
        try addFinding(allocator, out, "L4-unused-media", bits == 0, "attached {s}(s) never referenced", .{kind});
    }
}

fn addFinding(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), code: []const u8, err: bool, comptime fmt: []const u8, args: anytype) !void {
    try out.append(allocator, .{
        .code = code,
        .err = err,
        .msg = try std.fmt.allocPrint(allocator, fmt, args),
    });
}

fn headerPos(text: []const u8, name: []const u8) ?usize {
    var i: usize = 0;
    while (i + name.len < text.len) : (i += 1) {
        if (i != 0 and text[i - 1] != '\n') continue;
        if (!std.mem.startsWith(u8, text[i..], name)) continue;
        if (text[i + name.len] == ':') return i;
    }
    return null;
}

fn sectionBody(text: []const u8, name: []const u8, names: []const []const u8) ?[]const u8 {
    const start = headerPos(text, name) orelse return null;
    const after = start + name.len + 1;
    var end = text.len;
    for (names) |other| {
        const pos = headerPos(text, other) orelse continue;
        if (pos > start and pos < end) end = pos;
    }
    return text[after..end];
}

const Used = struct { picture: u16 = 0, video: u16 = 0, audio: u16 = 0 };

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

fn cardsToAssets(allocator: std.mem.Allocator, cards: []const Card) ![]Asset {
    const assets = try allocator.alloc(Asset, cards.len);
    for (cards, assets) |card, *asset| asset.* = card.asset;
    return assets;
}

fn hasBit(bits: u16, n: u32) bool {
    if (n == 0 or n > 15) return false;
    return bits & (@as(u16, 1) << @intCast(n - 1)) != 0;
}

fn setBit(bits: *u16, n: u32) void {
    if (n == 0 or n > 15) return;
    bits.* |= @as(u16, 1) << @intCast(n - 1);
}

fn scanLabels(allocator: std.mem.Allocator, text: []const u8, fl2va_bare: bool, used: *Used, unknown: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '<') continue;
        const close = std.mem.indexOfScalarPos(u8, text, i + 1, '>') orelse continue;
        const inner = std.mem.trim(u8, text[i + 1 .. close], " \t");
        const sp = std.mem.lastIndexOfScalar(u8, inner, ' ') orelse continue;
        const kind = inner[0..sp];
        const n = std.fmt.parseInt(u32, inner[sp + 1 ..], 10) catch continue;
        if (std.mem.eql(u8, kind, "Picture")) setBit(&used.picture, n) else if (std.mem.eql(u8, kind, "Video")) setBit(&used.video, n) else if (std.mem.eql(u8, kind, "Audio")) setBit(&used.audio, n) else if (!std.mem.eql(u8, kind, "Subject")) {
            var seen = false;
            for (unknown.items) |prev| {
                if (std.mem.eql(u8, prev, kind)) seen = true;
            }
            if (!seen) try unknown.append(allocator, kind);
        }
        i = close;
    }
    if (!fl2va_bare) return;
    i = 0;
    while (std.mem.indexOfPos(u8, text, i, "Picture ")) |pos| {
        const ok_prev = pos == 0 or !std.ascii.isAlphanumeric(text[pos - 1]);
        const not_angle = pos == 0 or text[pos - 1] != '<';
        if (ok_prev and not_angle) {
            const rest = text[pos + 8 ..];
            var n: u32 = 0;
            var j: usize = 0;
            while (j < rest.len and rest[j] >= '0' and rest[j] <= '9') : (j += 1) {
                n = n * 10 + (rest[j] - '0');
            }
            if (j != 0) setBit(&used.picture, n);
        }
        i = pos + 8;
    }
}

fn lastShotIn(text: []const u8) u32 {
    var max_n: u32 = 1;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, "[Shot ")) |pos| {
        const rest = text[pos + 6 ..];
        var n: u32 = 0;
        var j: usize = 0;
        while (j < rest.len and rest[j] >= '0' and rest[j] <= '9') : (j += 1) {
            n = n * 10 + (rest[j] - '0');
        }
        if (j != 0 and j < rest.len and rest[j] == ']' and n > max_n) max_n = n;
        i = pos + 6;
    }
    return max_n;
}

fn firstLines(text: []const u8) struct { first: []const u8, second: []const u8, second_present: bool } {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return .{ .first = trimmed, .second = "", .second_present = false };
    const first = std.mem.trimEnd(u8, trimmed[0..nl], " \t\r");
    const rest = trimmed[nl + 1 ..];
    const nl2 = std.mem.indexOfScalar(u8, rest, '\n') orelse return .{ .first = first, .second = std.mem.trimEnd(u8, rest, " \t\r"), .second_present = true };
    return .{ .first = first, .second = std.mem.trimEnd(u8, rest[0..nl2], " \t\r"), .second_present = true };
}

fn afterFirstLine(text: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return "";
    return trimmed[nl + 1 ..];
}

fn stripFences(text: []const u8) []const u8 {
    var out = text;
    if (std.mem.startsWith(u8, out, "```")) {
        if (std.mem.indexOfScalar(u8, out, '\n')) |nl| out = out[nl + 1 ..];
        if (std.mem.endsWith(u8, out, "```")) out = out[0 .. out.len - 3];
        out = std.mem.trim(u8, out, " \t\r\n");
    }
    return out;
}
