const std = @import("std");

const config = @import("../core/config.zig");
const grid = @import("grid.zig");

pub const Role = enum {
    frame_anchor_first,
    frame_anchor_last,
    subject,
    environment,
    style,
    storyboard,
    structure,
    edit_source,
    continuation_source,
    placed_subject,
    replacement_subject,
    voice_timbre,
    bgm,
    music_style,
    beat_reference,
    sfx,

    pub fn isAudioRef(self: Role) bool {
        return self == .voice_timbre or self == .sfx or self == .music_style or self == .beat_reference;
    }

    pub fn isSwap(self: Role) bool {
        return self == .placed_subject or self == .replacement_subject;
    }
};

pub const ManifestEntry = struct {
    kind: []const u8,
    label: []const u8,
    path: []const u8,
    role: Role,
    paired_with: ?[]const u8 = null,
    characterisation: []const u8 = "",
};

pub const SubjectPlan = struct {
    label: []const u8,
    sources: []const []const u8,
    descriptor: []const u8,
    attributes: []const []const u8 = &.{},
    retention: []const u8 = "fully_preserved",
    appears_in: []const u32 = &.{},
    retention_note: []const u8 = "",
    taken_over_by: ?[]const u8 = null,
};

pub const CameraMove = struct {
    type: []const u8,
    amplitude: ?[]const u8 = null,
    speed: ?[]const u8 = null,

    pub fn phrase(self: CameraMove, allocator: std.mem.Allocator) ![]u8 {
        const verb = cameraVerb(self.type);
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        try aw.writer.print("The camera {s}", .{verb});
        if (self.amplitude) |amp| try aw.writer.print(" with {s} amplitude", .{amp});
        if (self.speed) |spd| try aw.writer.print(" at {s} speed", .{spd});
        return aw.toOwnedSlice();
    }
};

pub const ShotPlan = struct {
    n: u32,
    start_ms: u32,
    end_ms: u32,
    camera: CameraMove,
    subjects: []const []const u8 = &.{},
    body: []const u8 = "",
    word_target: u32 = 120,
};

pub const Plan = struct {
    variant: config.Variant,
    duration_s: f32,
    manifest: []const ManifestEntry,
    subjects: []const SubjectPlan,
    shots: []const ShotPlan,
    task_types: []const []const u8,
    style_phrase: []const u8,
    summary: []const u8,
    ambient: []const u8,
    music: []const u8,
};

pub const DraftAsset = struct {
    kind: []const u8,
    path: []const u8,
    role: ?[]const u8 = null,
    paired_video_path: ?[]const u8 = null,
    note: []const u8 = "",
};

const draft_camera = [_]CameraMove{
    .{ .type = "Push In", .amplitude = "small", .speed = "slow" },
    .{ .type = "Truck Right", .amplitude = "small", .speed = "slow" },
    .{ .type = "Static Shot", .amplitude = null, .speed = null },
    .{ .type = "Pull Out", .amplitude = "small", .speed = "slow" },
};
const maximal_camera = [_]CameraMove{
    .{ .type = "Push In", .amplitude = "large", .speed = "fast" },
    .{ .type = "Truck Right", .amplitude = "large", .speed = "fast" },
    .{ .type = "Pull Out", .amplitude = "large", .speed = "fast" },
    .{ .type = "Tilt Up", .amplitude = "large", .speed = "fast" },
};
const assertive_camera = [_]CameraMove{
    .{ .type = "Push In", .amplitude = "large", .speed = null },
    .{ .type = "Truck Right", .amplitude = "large", .speed = null },
    .{ .type = "Static Shot", .amplitude = null, .speed = null },
    .{ .type = "Pull Out", .amplitude = "large", .speed = null },
};

const task_order = [_][]const u8{
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
};

pub fn parseRole(text: []const u8) ?Role {
    return std.meta.stringToEnum(Role, text);
}

pub fn defaultRole(kind: []const u8, stated: ?[]const u8) Role {
    if (stated) |s| {
        if (parseRole(s)) |r| return r;
    }
    return switch (kind[0]) {
        'i' => .subject,
        'v' => .subject,
        else => .sfx,
    };
}

pub fn buildManifest(allocator: std.mem.Allocator, assets: []const DraftAsset) ![]ManifestEntry {
    var images: std.ArrayList(DraftAsset) = .empty;
    defer images.deinit(allocator);
    var videos: std.ArrayList(DraftAsset) = .empty;
    defer videos.deinit(allocator);
    var audios: std.ArrayList(DraftAsset) = .empty;
    defer audios.deinit(allocator);
    for (assets) |a| {
        switch (a.kind[0]) {
            'i' => try images.append(allocator, a),
            'v' => try videos.append(allocator, a),
            else => try audios.append(allocator, a),
        }
    }
    std.mem.sort(DraftAsset, images.items, {}, struct {
        fn less(_: void, a: DraftAsset, b: DraftAsset) bool {
            return anchorRank(a) < anchorRank(b);
        }
    }.less);

    var out: std.ArrayList(ManifestEntry) = .empty;
    errdefer out.deinit(allocator);
    var n_pic: u32 = 0;
    var n_vid: u32 = 0;
    var n_aud: u32 = 0;
    var n_standalone: u32 = 0;
    for (images.items) |a| {
        n_pic += 1;
        try out.append(allocator, .{
            .kind = "image",
            .label = try std.fmt.allocPrint(allocator, "<Picture {d}>", .{n_pic}),
            .path = a.path,
            .role = defaultRole(a.kind, a.role),
            .characterisation = a.note,
        });
    }
    for (videos.items) |v| {
        n_vid += 1;
        if (pairedAudio(audios.items, v.path)) |snd| {
            n_aud += 1;
            try out.append(allocator, .{
                .kind = "audio",
                .label = try std.fmt.allocPrint(allocator, "<Audio {d}>", .{n_aud}),
                .path = snd.path,
                .role = defaultRole(snd.kind, snd.role),
                .paired_with = try std.fmt.allocPrint(allocator, "<Video {d}>", .{n_vid}),
                .characterisation = snd.note,
            });
        }
        try out.append(allocator, .{
            .kind = "video",
            .label = try std.fmt.allocPrint(allocator, "<Video {d}>", .{n_vid}),
            .path = v.path,
            .role = defaultRole(v.kind, v.role),
            .characterisation = v.note,
        });
    }
    for (audios.items) |a| {
        if (a.paired_video_path != null) continue;
        n_aud += 1;
        n_standalone += 1;
        try out.append(allocator, .{
            .kind = "audio",
            .label = try std.fmt.allocPrint(allocator, "<Audio {d}>", .{n_aud}),
            .path = a.path,
            .role = defaultRole(a.kind, a.role),
            .characterisation = a.note,
        });
    }
    return out.toOwnedSlice(allocator);
}

pub fn deriveTaskTypes(allocator: std.mem.Allocator, manifest: []const ManifestEntry) ![]const []const u8 {
    var flags = [_]bool{false} ** task_order.len;
    var has_audio = false;
    var has_bgm = false;
    var has_audio_ref = false;
    for (manifest) |m| {
        if (m.kind[0] == 'a') has_audio = true;
        if (m.role == .bgm) has_bgm = true;
        if (m.role.isAudioRef()) has_audio_ref = true;
        if (m.role == .frame_anchor_first or m.role == .frame_anchor_last) flags[0] = true;
        if (m.role == .subject or m.role == .environment or m.role == .style or m.role == .storyboard or m.role == .structure or m.role == .voice_timbre or m.role.isSwap()) flags[1] = true;
        if (m.role == .edit_source) flags[2] = true;
        if (m.role == .continuation_source) flags[3] = true;
    }
    if (has_audio and has_bgm) flags[4] = true;
    if (has_audio and has_audio_ref) flags[5] = true;
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(allocator);
    for (task_order, flags) |name, on| {
        if (on) try out.append(allocator, name);
    }
    if (out.items.len == 0) try out.append(allocator, "reference generation");
    return out.toOwnedSlice(allocator);
}

pub fn buildPlan(
    allocator: std.mem.Allocator,
    variant: config.Variant,
    duration_s: f32,
    intent: []const u8,
    assets: []const DraftAsset,
    style_phrase: []const u8,
    magnitude: []const u8,
    pinned_shots: ?u32,
) !Plan {
    const manifest = try buildManifest(allocator, assets);
    var subjects: std.ArrayList(SubjectPlan) = .empty;
    errdefer subjects.deinit(allocator);
    var n_subj: u32 = 0;
    for (manifest) |m| {
        if (m.kind[0] == 'a') continue;
        if (m.role == .storyboard or m.role == .style or m.role == .structure) continue;
        n_subj += 1;
        const desc = try std.fmt.allocPrint(allocator, "the attached {s}", .{m.kind});
        const sources = try allocator.alloc([]const u8, 1);
        sources[0] = m.label;
        try subjects.append(allocator, .{
            .label = try std.fmt.allocPrint(allocator, "<Subject {d}>", .{n_subj}),
            .sources = sources,
            .descriptor = desc,
            .retention = markerFor(m.role),
            .retention_note = try std.fmt.allocPrint(allocator, "{s} is retained", .{desc}),
        });
    }

    var n = grid.shotCount(duration_s, variant, pinned_shots);
    for (manifest) |m| {
        if (m.role == .edit_source or m.role == .continuation_source) n = 1;
    }
    const bounds = grid.cutBounds(duration_s, n);
    const rotation = cameraRotation(magnitude);
    const shots = try allocator.alloc(ShotPlan, n);
    const subj_labels = try allocator.alloc([]const u8, subjects.items.len);
    for (subjects.items, subj_labels) |s, *l| l.* = s.label;
    const total_words: u32 = switch (variant) {
        .ref2va => 400,
        .fl2va => 380,
        .t2va => 330,
    };
    const total_ms = bounds[n];
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const start = bounds[i];
        const end = bounds[i + 1];
        const share = if (total_ms == 0) 1.0 else @as(f32, @floatFromInt(end - start)) / @as(f32, @floatFromInt(total_ms));
        shots[i] = .{
            .n = i + 1,
            .start_ms = start,
            .end_ms = end,
            .camera = rotation[i % rotation.len],
            .subjects = subj_labels,
            .word_target = @max(90, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(total_words)) * share)))),
        };
    }
    for (subjects.items) |*s| {
        const appears = try allocator.alloc(u32, n);
        var k: u32 = 0;
        while (k < n) : (k += 1) appears[k] = k + 1;
        s.appears_in = appears;
    }

    const style = if (style_phrase.len != 0) style_phrase else styleFromIntent(intent);
    const tasks = try deriveTaskTypes(allocator, manifest);
    const summary = if (subj_labels.len != 0)
        try std.fmt.allocPrint(allocator, "The target video shows {s} in the scene described by the request.", .{try joinLabels(allocator, subj_labels)})
    else
        try allocator.dupe(u8, "The target video shows the scene described by the request.");

    var plan: Plan = .{
        .variant = variant,
        .duration_s = duration_s,
        .manifest = manifest,
        .subjects = try subjects.toOwnedSlice(allocator),
        .shots = shots,
        .task_types = tasks,
        .style_phrase = style,
        .summary = summary,
        .ambient = "Room tone continues throughout the video.",
        .music = "N/A",
    };
    try fillShotBodies(allocator, &plan, intent);
    return plan;
}

pub fn render(allocator: std.mem.Allocator, plan: Plan) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    if (plan.variant == .ref2va) {
        try w.writeAll("subject_definitions:\n");
        try writeDefinitions(allocator, w, plan);
        try w.writeAll("\n\nsummary:\n");
        const tasks = try joinTasks(allocator, plan.task_types);
        defer allocator.free(tasks);
        try w.print("[{s}] {s}", .{ tasks, plan.summary });
        try w.writeAll("\n\nretention_analysis:\n");
        try writeRetention(allocator, w, plan);
        try w.writeAll("\n\ndetailed_description:\n");
        try writeDescription(allocator, w, plan);
        try w.print("\n\noverall_soundscape:\n{s}\n\nnon_diegetic_music:\n{s}\n", .{ plan.ambient, plan.music });
    } else {
        const n_pic = countKind(plan.manifest, 'i');
        const last_only = n_pic == 1 and lastOnly(plan.manifest);
        const last_shot = if (plan.shots.len != 0) plan.shots[plan.shots.len - 1].n else 1;
        const line = try grid.instructionLine(allocator, plan.variant, last_shot, plan.duration_s, n_pic, last_only);
        defer allocator.free(line);
        if (line.len != 0) {
            try w.writeAll(line);
            try w.writeAll("\n\n");
        }
        try w.writeAll("integrated_multimodal_description: ");
        try writeDescription(allocator, w, plan);
        try w.print("\n\noverall_soundscape: {s}\n\nnon_diegetic_music: {s}\n", .{ plan.ambient, plan.music });
    }
    return aw.toOwnedSlice();
}

fn fillShotBodies(allocator: std.mem.Allocator, plan: *Plan, intent: []const u8) !void {
    const shots = @constCast(plan.shots);
    for (shots) |*shot| {
        shot.body = try draftBody(allocator, plan.*, shot.*, intent);
    }
}

fn draftBody(allocator: std.mem.Allocator, plan: Plan, shot: ShotPlan, intent: []const u8) ![]u8 {
    const cam = try shot.camera.phrase(allocator);
    defer allocator.free(cam);
    const trimmed = std.mem.trim(u8, intent, " \t\r\n");
    if (plan.variant == .fl2va and shot.n == 1) {
        const n_pic = countKind(plan.manifest, 'i');
        const last_only = n_pic == 1 and lastOnly(plan.manifest);
        if (n_pic == 1 and last_only) {
            return std.fmt.allocPrint(allocator, "The shot develops continuously until the composition of Picture 1 at the end. {s}. {s}.", .{ lowerFirst(trimmed), cam });
        }
        if (n_pic == 1) {
            return std.fmt.allocPrint(allocator, "The shot opens in the composition established by Picture 1. {s}. {s}.", .{ lowerFirst(trimmed), cam });
        }
        return std.fmt.allocPrint(allocator, "The shot opens in the composition established by Picture 1. Across the shot, {s}. {s}. The framing narrows toward the pose, spacing and composition established by Picture 2, which the shot settles into at the end.", .{ lowerFirst(trimmed), cam });
    }
    if (shot.n == 1) {
        if (plan.variant == .ref2va and plan.subjects.len != 0) {
            return std.fmt.allocPrint(allocator, "The frame holds {s}. {s}. Across the shot, {s}.", .{ plan.subjects[0].label, cam, lowerFirst(trimmed) });
        }
        return std.fmt.allocPrint(allocator, "The frame holds the scene described as: {s}. {s}.", .{ trimmed, cam });
    }
    const ts = grid.msToTimestamp(shot.start_ms);
    _ = ts;
    if (plan.variant == .ref2va and plan.subjects.len != 0) {
        return std.fmt.allocPrint(allocator, "The shot cuts to {s}, still in frame. {s}.", .{ plan.subjects[0].label, cam });
    }
    return std.fmt.allocPrint(allocator, "The shot cuts to another view of the same scene. {s}.", .{cam});
}

fn writeDescription(allocator: std.mem.Allocator, w: anytype, plan: Plan) !void {
    _ = allocator;
    const prefix = std.mem.trim(u8, plan.style_phrase, " \t.,");
    for (plan.shots, 0..) |shot, i| {
        if (i != 0) {
            if (plan.variant == .ref2va) try w.writeAll("\n") else try w.writeAll(" ");
        }
        if (shot.n == 1) {
            if (plan.variant == .ref2va) {
                try w.writeAll(styleSentence(prefix));
                try w.writeAll("\n");
                try w.print("[Shot 1] {s}", .{shot.body});
            } else if (prefix.len != 0) {
                try w.print("[Shot 1] {s}, ", .{prefix});
                try writeLowerFirst(w, shot.body);
            } else {
                try w.print("[Shot 1] {s}", .{shot.body});
            }
        } else {
            const ts = grid.msToTimestamp(shot.start_ms);
            try w.print("[Shot {d}] At {s}, ", .{ shot.n, ts[0..9] });
            try writeLowerFirst(w, shot.body);
        }
    }
}

fn writeDefinitions(allocator: std.mem.Allocator, w: anytype, plan: Plan) !void {
    var first = true;
    for (plan.subjects) |s| {
        if (!first) try w.writeAll("\n");
        first = false;
        const src = try joinLabels(allocator, s.sources);
        defer allocator.free(src);
        try w.print("{s} is {s} in {s}.", .{ s.label, s.descriptor, src });
    }
    for (plan.manifest) |m| {
        if (m.role == .storyboard) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} is a storyboard reference for [Shot 1], defining their viewpoint, subject placement, and shot order.", .{m.label});
        } else if (m.role == .style) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} is the style and composition reference for the target video, defining its medium, line, palette, shading and composition.", .{m.label});
        } else if (m.role == .structure) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} is the source video providing the camera movement and cutting rhythm for the target video.", .{m.label});
        } else if (m.role == .edit_source) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} is the source video for the target video edit.", .{m.label});
        } else if (m.role == .continuation_source) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} is the source video the target video continues from.", .{m.label});
        } else if (m.kind[0] == 'a') {
            if (!first) try w.writeAll("\n");
            first = false;
            const detail = if (m.characterisation.len != 0) m.characterisation else "containing a spoken vocal layer";
            try w.print("{s} is a sound-texture reference for the target video — {s}.", .{ m.label, detail });
        }
    }
    if (first) {
        try w.writeAll("<Subject 1> is the scene described by the request.");
    }
}

fn writeRetention(allocator: std.mem.Allocator, w: anytype, plan: Plan) !void {
    var first = true;
    for (plan.subjects) |s| {
        if (!first) try w.writeAll("\n");
        first = false;
        const appears = try shotList(allocator, s.appears_in);
        defer allocator.free(appears);
        try w.print("{s} (appears in {s}): {s} - {s}.", .{ s.label, appears, s.retention, s.retention_note });
    }
    for (plan.manifest) |m| {
        if (m.role == .frame_anchor_first) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} ([Shot 1] first frame): fully_preserved - the composition, subject placement and lighting of the opening frame are held.", .{m.label});
        } else if (m.role == .frame_anchor_last) {
            if (!first) try w.writeAll("\n");
            first = false;
            const last = if (plan.shots.len != 0) plan.shots[plan.shots.len - 1].n else 1;
            try w.print("{s} ([Shot {d}] last frame): fully_preserved - the final composition and subject placement are reached.", .{ m.label, last });
        } else if (m.role == .storyboard) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} (storyboard reference): weak_reference - the viewpoint, subject placement and shot order are followed, while the drawing itself is not reproduced.", .{m.label});
        } else if (m.role == .style) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} (style and composition): weak_reference - the target video adheres to the reference's aesthetic, while its contents are not reproduced.", .{m.label});
        } else if (m.role == .structure) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} (camera movement and cutting rhythm): weak_reference - the camera moves and edit timing are followed, while the video's contents are not reproduced.", .{m.label});
        } else if (m.role == .edit_source) {
            if (!first) try w.writeAll("\n");
            first = false;
            try w.print("{s} (source video editing): fully_preserved - the original framing, lighting and setting are maintained while the edit is applied.", .{m.label});
        } else if (m.kind[0] == 'a') {
            if (!first) try w.writeAll("\n");
            first = false;
            const marker = if (m.role == .bgm) "partially_copy" else "reference";
            try w.print("{s}: {s} - its sound texture is referenced, not copied as the final track.", .{ m.label, marker });
        }
    }
    if (first) {
        try w.writeAll("<Subject 1> (appears in [Shot 1]): fully_preserved - the requested scene is retained.");
    }
}

fn cameraVerb(type_name: []const u8) []const u8 {
    const pairs = [_]struct { k: []const u8, v: []const u8 }{
        .{ .k = "Zoom In", .v = "zooms in" },
        .{ .k = "Zoom Out", .v = "zooms out" },
        .{ .k = "Push In", .v = "pushes in" },
        .{ .k = "Pull Out", .v = "pulls out" },
        .{ .k = "Pan Left", .v = "pans left" },
        .{ .k = "Pan Right", .v = "pans right" },
        .{ .k = "Truck Left", .v = "trucks left" },
        .{ .k = "Truck Right", .v = "trucks right" },
        .{ .k = "Tilt Up", .v = "tilts up" },
        .{ .k = "Tilt Down", .v = "tilts down" },
        .{ .k = "Pedestal Up", .v = "rises on a pedestal move" },
        .{ .k = "Pedestal Down", .v = "lowers on a pedestal move" },
        .{ .k = "Arc Shot", .v = "arcs around the subject" },
        .{ .k = "Tracking Shot", .v = "tracks with the subject" },
        .{ .k = "Static Shot", .v = "holds a static shot" },
        .{ .k = "Shake Slightly", .v = "shakes slightly" },
        .{ .k = "Shake Strongly", .v = "shakes strongly" },
        .{ .k = "POV", .v = "takes the subject's point of view" },
        .{ .k = "Roll Clockwise", .v = "rolls clockwise" },
        .{ .k = "Roll Counterclockwise", .v = "rolls counterclockwise" },
    };
    for (pairs) |p| {
        if (std.mem.eql(u8, p.k, type_name)) return p.v;
    }
    return type_name;
}

fn cameraRotation(magnitude: []const u8) []const CameraMove {
    if (std.mem.eql(u8, magnitude, "maximal")) return &maximal_camera;
    if (std.mem.eql(u8, magnitude, "assertive")) return &assertive_camera;
    return &draft_camera;
}

fn markerFor(role: Role) []const u8 {
    return switch (role) {
        .continuation_source => "partially_preserved",
        .style, .storyboard, .structure => "weak_reference",
        else => "fully_preserved",
    };
}

fn styleFromIntent(intent: []const u8) []const u8 {
    const low_terms = [_][]const u8{ "anime", "manga", "3d", "cgi", "watercolor", "watercolour", "stop-motion" };
    const lower = intent;
    for (low_terms) |t| {
        if (containsIgnoreCase(lower, t)) {
            if (std.mem.eql(u8, t, "anime") or std.mem.eql(u8, t, "manga")) return "anime, cinematic";
            if (std.mem.eql(u8, t, "3d") or std.mem.eql(u8, t, "cgi")) return "3D CG, cinematic";
            if (std.mem.eql(u8, t, "watercolor") or std.mem.eql(u8, t, "watercolour")) return "watercolour, cinematic";
            if (std.mem.eql(u8, t, "stop-motion")) return "stop-motion, cinematic";
        }
    }
    return "Live-action, cinematic";
}

fn styleSentence(phrase: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, phrase, " \t.,");
    if (trimmed.len == 0) return "The target video is in live-action style.";
    if (std.ascii.eqlIgnoreCase(trimmed, "Live-action, cinematic"))
        return "The target video is in live-action style, cinematic.";
    if (std.ascii.eqlIgnoreCase(trimmed, "Live-action"))
        return "The target video is in live-action style.";
    return "The target video is in live-action style, cinematic.";
}

fn lowerFirst(text: []const u8) []const u8 {
    return text;
}

fn containsIgnoreCase(hay: []const u8, needle: []const u8) bool {
    if (needle.len > hay.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(hay[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn countKind(manifest: []const ManifestEntry, first: u8) u32 {
    var n: u32 = 0;
    for (manifest) |m| {
        if (m.kind[0] == first) n += 1;
    }
    return n;
}

fn lastOnly(manifest: []const ManifestEntry) bool {
    var last = false;
    var first = false;
    for (manifest) |m| {
        if (m.role == .frame_anchor_first) first = true;
        if (m.role == .frame_anchor_last) last = true;
    }
    return last and !first;
}

fn pairedAudio(audios: []const DraftAsset, video_path: []const u8) ?DraftAsset {
    for (audios) |a| {
        if (a.paired_video_path) |p| {
            if (std.mem.eql(u8, p, video_path)) return a;
        }
    }
    return null;
}

fn anchorRank(a: DraftAsset) u8 {
    const role = a.role orelse return 2;
    if (std.mem.eql(u8, role, "frame_anchor_first")) return 0;
    if (std.mem.eql(u8, role, "frame_anchor_last")) return 1;
    return 2;
}

fn joinTasks(allocator: std.mem.Allocator, labels: []const []const u8) ![]u8 {
    if (labels.len == 0) return allocator.dupe(u8, "");
    if (labels.len == 1) return allocator.dupe(u8, labels[0]);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    for (labels, 0..) |l, i| {
        if (i != 0) try aw.writer.writeAll(" + ");
        try aw.writer.writeAll(l);
    }
    return aw.toOwnedSlice();
}

fn joinLabels(allocator: std.mem.Allocator, labels: []const []const u8) ![]u8 {
    if (labels.len == 0) return allocator.dupe(u8, "");
    if (labels.len == 1) return allocator.dupe(u8, labels[0]);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    for (labels, 0..) |l, i| {
        if (i != 0) try aw.writer.writeAll(", ");
        try aw.writer.writeAll(l);
    }
    return aw.toOwnedSlice();
}

fn writeLowerFirst(w: anytype, text: []const u8) !void {
    if (text.len == 0) return;
    if (text[0] >= 'A' and text[0] <= 'Z') {
        try w.writeByte(text[0] + 32);
        try w.writeAll(text[1..]);
    } else {
        try w.writeAll(text);
    }
}

fn shotList(allocator: std.mem.Allocator, shots: []const u32) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    for (shots, 0..) |n, i| {
        if (i != 0) try aw.writer.writeAll(", ");
        try aw.writer.print("[Shot {d}]", .{n});
    }
    return aw.toOwnedSlice();
}
