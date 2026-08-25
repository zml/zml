const std = @import("std");

const config = @import("../core/config.zig");
const grid = @import("grid.zig");

pub const Finding = struct {
    code: []const u8,
    err: bool,
    msg: []const u8,
};

pub const RoleDecl = struct {
    label: []const u8,
    role: []const u8,
};

pub const Context = struct {
    variant: config.Variant,
    duration_s: f32,
    n_pictures: u32 = 0,
    n_videos: u32 = 0,
    n_audios: u32 = 0,
    task_types: []const []const u8 = &.{},
    declared_roles: []const RoleDecl = &.{},
    creativity: []const u8 = "balanced",
    pinned_shots: ?u32 = null,
    dialogue: []const []const u8 = &.{},
    forbids_speech: bool = false,
    forbids_score: bool = false,
    forbids_text: bool = false,
};

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
const task_types = [_][]const u8{
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
};
const visual_markers = [_][]const u8{ "fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference" };
const audio_markers = [_][]const u8{ "fully_copy", "partially_copy", "reference", "weak_reference" };
const camera_types = [_][]const u8{
    "Zoom In",        "Zoom Out",      "Push In",        "Pull Out",
    "Pan Left",       "Pan Right",     "Truck Left",     "Truck Right",
    "Tilt Up",        "Tilt Down",     "Pedestal Up",    "Pedestal Down",
    "Arc Shot",       "Tracking Shot", "Static Shot",    "Shake Slightly",
    "Shake Strongly", "POV",           "Roll Clockwise", "Roll Counterclockwise",
};
const camera_stems = [_][]const u8{
    "zooms in",        "zooms out",       "pushes in",   "pulls out",   "pans left",      "pans right",
    "trucks left",     "trucks right",    "tilts up",    "tilts down",  "holds a static", "static shot",
    "shakes slightly", "shakes strongly", "arcs around", "tracks with",
};
const unicode_hazards = [_]u21{ '‘', '’', '“', '”', 0x00A0, '〈', '〉', '＜', '＞', '（', '）', '［', '］' };
const leak_markers = [_][]const u8{ "<think>", "</think>", "<reasoning>", "</reasoning>", "assistantfinal" };

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

pub fn hasCode(findings: []const Finding, code: []const u8) bool {
    for (findings) |finding| {
        if (std.mem.eql(u8, finding.code, code)) return true;
    }
    return false;
}

pub fn validate(allocator: std.mem.Allocator, text: []const u8, ctx: Context) ![]Finding {
    var out: std.ArrayList(Finding) = .empty;
    errdefer {
        for (out.items) |finding| allocator.free(finding.msg);
        out.deinit(allocator);
    }
    const is_ref = ctx.variant == .ref2va;
    const names: []const []const u8 = if (is_ref) &ref_sections else &base_sections;
    const main_field = if (is_ref) "detailed_description" else "integrated_multimodal_description";
    const wrong_field = if (is_ref) "integrated_multimodal_description" else "detailed_description";
    const duration = grid.effectiveSeconds(ctx.duration_s);

    var last_pos: usize = 0;
    var seen_section = false;
    var order_ok = true;
    for (names) |name| {
        if (headerPos(text, name)) |pos| {
            if (seen_section and pos < last_pos) order_ok = false;
            last_pos = pos;
            seen_section = true;
        } else {
            try add(allocator, &out, "S1-missing-section", true, "'{s}:' is absent", .{name});
        }
    }
    if (!order_ok) try add(allocator, &out, "S2-section-order", true, "sections out of mandated order", .{});
    const body_start = std.mem.trimStart(u8, text, " \t\r\n");
    if (is_ref and !std.mem.startsWith(u8, body_start, "subject_definitions:")) {
        try add(allocator, &out, "S3-preamble", true, "brief must begin with 'subject_definitions:'", .{});
    }
    if (std.mem.indexOf(u8, text, "```") != null) {
        try add(allocator, &out, "S4-code-fence", true, "output contains a markdown code fence", .{});
    }
    if (headerPos(text, wrong_field) != null) {
        try add(allocator, &out, "S5-mode-field-crossed", true, "'{s}' is the other mode's field", .{wrong_field});
    }
    for (names) |name| {
        const body = sectionBody(text, name, names) orelse continue;
        if (std.mem.trim(u8, body, " \t\r\n").len == 0) {
            try add(allocator, &out, "S9-section-empty", true, "'{s}:' is present but empty", .{name});
        }
    }

    if (ctx.variant == .fl2va) {
        const n_pic = if (ctx.n_pictures == 0) 2 else ctx.n_pictures;
        const last_only = roleIs(ctx, "frame_anchor_last") and !roleIs(ctx, "frame_anchor_first") and ctx.n_pictures == 1;
        const want = try grid.instructionLine(allocator, .fl2va, lastShotIn(text), ctx.duration_s, n_pic, last_only);
        defer allocator.free(want);
        const lines = firstLines(text);
        if (!grid.startsWithInstruction(lines.first)) {
            try add(allocator, &out, "I1-instruction-line-missing", true, "FL2VA brief must open with the alignment instruction", .{});
        } else {
            if (!std.mem.eql(u8, lines.first, want)) {
                try add(allocator, &out, "I2-instruction-line-not-exact", true, "instruction line differs from the mandated one", .{});
            }
            if (lines.second_present and lines.second.len != 0) {
                try add(allocator, &out, "I3-instruction-line-no-blank-line", true, "the instruction line must be followed by one blank line", .{});
            }
        }
    }

    var used = Used{};
    var unknown: std.ArrayList([]const u8) = .empty;
    defer unknown.deinit(allocator);
    try scanLabels(allocator, text, ctx.variant == .fl2va, &used, &unknown);
    for (unknown.items) |kind| {
        try add(allocator, &out, "L1-unknown-label", true, "<{s} N> is not in the label vocabulary", .{kind});
    }
    if (openLabel(text)) |kind| {
        try add(allocator, &out, "L6-label-not-closed", true, "<{s} N is opened and never closed", .{kind});
    }
    const n_pic_have = if (ctx.variant == .fl2va and ctx.n_pictures == 0) 2 else ctx.n_pictures;
    try mediaFindings(allocator, &out, "Picture", n_pic_have, used.picture);
    try mediaFindings(allocator, &out, "Video", ctx.n_videos, used.video);
    try mediaFindings(allocator, &out, "Audio", ctx.n_audios, used.audio);

    const defs = sectionBody(text, "subject_definitions", &ref_sections) orelse "";
    const summ = sectionBody(text, "summary", &ref_sections) orelse "";
    const ret = sectionBody(text, "retention_analysis", &ref_sections) orelse "";
    const desc = sectionBody(text, main_field, names) orelse "";
    const music = std.mem.trim(u8, sectionBody(text, "non_diegetic_music", names) orelse "", " \t\r\n");
    const sound = sectionBody(text, "overall_soundscape", names) orelse "";

    if (is_ref) {
        try refStructure(allocator, &out, defs, summ, ret, ctx);
    }

    try timeline(allocator, &out, desc, duration, ctx);
    try cameraAndProse(allocator, &out, desc, text, is_ref);
    try dialogueRules(allocator, &out, desc, text, ctx);
    try audioRules(allocator, &out, desc, sound, music, ctx);
    try creativityRules(allocator, &out, desc, text, music, ctx);
    try hygiene(allocator, &out, text, desc);
    return out.toOwnedSlice(allocator);
}

fn refStructure(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), defs: []const u8, summ: []const u8, ret: []const u8, ctx: Context) !void {
    var defined = Used{};
    scanDefinedSubjects(defs, &defined);
    var used_summ = Used{};
    var ignore: std.ArrayList([]const u8) = .empty;
    defer ignore.deinit(allocator);
    try scanLabels(allocator, summ, false, &used_summ, &ignore);
    var n: u32 = 1;
    while (n <= 15) : (n += 1) {
        if (hasBit(used_summ.subject, n) and !hasBit(defined.subject, n)) {
            try add(allocator, out, "L2-undefined-subject", true, "<Subject {d}> is used but never defined", .{n});
        }
    }

    const pref = firstBrackets(summ);
    if (pref == null) {
        try add(allocator, out, "M1-task-prefix", true, "summary must open with a [task type] prefix", .{});
    } else {
        var parts_ok = true;
        var it = std.mem.splitScalar(u8, pref.?, '+');
        var seen_task: u8 = 0;
        while (it.next()) |raw| {
            const p = std.mem.trim(u8, raw, " \t");
            if (!isTaskType(p)) {
                try add(allocator, out, "M2-task-type", true, "'{s}' is not a legal task type", .{p});
                parts_ok = false;
            }
            const bit: u8 = taskBit(p);
            if (bit != 0 and seen_task & bit != 0) {
                try add(allocator, out, "M3-task-dupe", true, "task type repeated in the prefix", .{});
            }
            seen_task |= bit;
        }
        if (parts_ok and ctx.task_types.len != 0 and !taskSetEqual(pref.?, ctx.task_types)) {
            try add(allocator, out, "M16-task-prefix-not-derived", true, "the summary task types do not match the attachments", .{});
        }
        if (containsPhrase(pref.?, "video editing") and ctx.n_videos == 0) {
            try add(allocator, out, "M5-editing-without-video", true, "'video editing' claimed but no <Video N> is referenced", .{});
        }
        if (containsPhrase(pref.?, "video editing") and std.mem.indexOf(u8, summ, "The target video is an edited version of <Video") == null) {
            try add(allocator, out, "M6-editing-opening", true, "a video-editing summary must contain the mandated opening sentence", .{});
        }
        if (containsPhrase(pref.?, "audio reuse") and ctx.n_audios == 0) {
            try add(allocator, out, "M7-audio-reuse-without-audio", true, "the summary claims 'audio reuse' and no <Audio N> is attached", .{});
        }
        if (containsPhrase(pref.?, "audio reference") and ctx.n_audios == 0) {
            try add(allocator, out, "M8-audio-reference-without-audio", true, "the summary claims 'audio reference' and no <Audio N> is attached", .{});
        }
        if (containsPhrase(pref.?, "keyframe completion") and ctx.n_pictures == 0) {
            try add(allocator, out, "M11-keyframe-without-picture", true, "the summary claims 'keyframe completion' and no picture is attached", .{});
        }
        if (containsPhrase(pref.?, "video continuation") and ctx.n_videos == 0) {
            try add(allocator, out, "M12-continuation-without-video", true, "the summary claims 'video continuation' and no video is attached", .{});
        }
        var later = false;
        var si: usize = 0;
        while (std.mem.indexOfPos(u8, summ, si, "[")) |pos| {
            const close = std.mem.indexOfScalarPos(u8, summ, pos + 1, ']') orelse break;
            const inner = summ[pos + 1 .. close];
            if (later and allTaskParts(inner)) {
                try add(allocator, out, "M9-task-prefix-repeated", true, "the task-type prefix appears again inside the summary", .{});
                break;
            }
            if (allTaskParts(inner)) later = true;
            si = close + 1;
        }
    }

    var analysed = Used{};
    var ret_it = std.mem.splitScalar(u8, ret, '\n');
    while (ret_it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        if (trimmed[0] != '<') {
            try add(allocator, out, "R1-malformed-entry", true, "unparseable retention line", .{});
            continue;
        }
        const close = std.mem.indexOfScalar(u8, trimmed, '>') orelse continue;
        const inner = trimmed[1..close];
        const kind = kindOf(inner);
        const colon = std.mem.indexOfScalar(u8, trimmed[close..], ':') orelse {
            try add(allocator, out, "R1-malformed-entry", true, "unparseable retention line", .{});
            continue;
        };
        var marker = std.mem.trim(u8, trimmed[close + colon + 1 ..], " \t");
        if (std.mem.indexOfScalar(u8, marker, ' ') != null or std.mem.indexOfScalar(u8, marker, '-') != null) {
            const end = std.mem.indexOfAny(u8, marker, " -") orelse marker.len;
            marker = marker[0..end];
        }
        if (std.mem.eql(u8, kind, "Subject")) {
            if (parseN(inner)) |num| setBit(&analysed.subject, num);
            if (std.mem.indexOf(u8, trimmed[0 .. close + colon], "appears in") == null) {
                try add(allocator, out, "R3-missing-appears-in", true, "subject is missing the mandated '(appears in [Shot N])'", .{});
            }
        }
        const legal = if (std.mem.eql(u8, kind, "Audio")) audio_markers[0..] else visual_markers[0..];
        if (!inList(marker, legal)) {
            try add(allocator, out, "R2-illegal-marker", true, "'{s}' is not a legal relationship marker", .{marker});
        }
        if (std.mem.indexOf(u8, trimmed, "(S") != null) {
            try add(allocator, out, "R4-speaker-in-retention", true, "speaker IDs must not appear in retention_analysis", .{});
        }
        if (std.mem.eql(u8, marker, "attribute_transfer") and indexOfLabel(trimmed[close + colon ..]) == null) {
            try add(allocator, out, "R32-transfer-target-unnamed", true, "attribute_transfer names no target subject", .{});
        }
    }
    n = 1;
    while (n <= 15) : (n += 1) {
        if (hasBit(defined.subject, n) and !hasBit(analysed.subject, n)) {
            try add(allocator, out, "R5-unanalysed-subject", false, "<Subject {d}> defined but absent from retention_analysis", .{n});
        }
    }
    if (countLabelLines(ret)) |dup| {
        try add(allocator, out, "R24-label-analysed-twice", true, "<{s}> has more than one retention line", .{dup});
    }
    var def_it = std.mem.splitScalar(u8, defs, '\n');
    while (def_it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (!std.mem.startsWith(u8, t, "<Picture ") and !std.mem.startsWith(u8, t, "<Video ")) continue;
        if (std.mem.indexOf(u8, t, " is") == null and std.mem.indexOf(u8, t, ":") == null) continue;
        if (!containsIgnoreCase(t, "reference for") and !containsIgnoreCase(t, "source of") and !containsIgnoreCase(t, "the reference image for")) continue;
        const close = std.mem.indexOfScalar(u8, t, '>') orelse continue;
        const lab = t[1..close];
        if (std.mem.indexOf(u8, ret, lab) == null) {
            try add(allocator, out, "L5-redundant-source-line", true, "<{s}> is cited only as a source and has no retention_analysis entry", .{lab});
        }
    }
    for (ctx.declared_roles) |decl| {
        if (std.mem.eql(u8, decl.role, "storyboard") and containsLabel(defs, decl.label) and hasSubjectLineWith(defs, decl.label)) {
            try add(allocator, out, "R28-storyboard-cited-as-content", true, "{s} is a storyboard cited as content", .{decl.label});
        }
        if (std.mem.eql(u8, decl.role, "style") and hasSubjectLineWith(defs, decl.label)) {
            try add(allocator, out, "R29-style-cited-as-content", true, "{s} is a style plate cited as content", .{decl.label});
        }
        if (std.mem.eql(u8, decl.role, "structure") and hasSubjectLineWith(defs, decl.label)) {
            try add(allocator, out, "R30-structure-cited-as-content", true, "{s} is a structure clip cited as content", .{decl.label});
        }
        if ((std.mem.eql(u8, decl.role, "frame_anchor_first") or std.mem.eql(u8, decl.role, "frame_anchor_last")) and ctx.variant == .ref2va) {
            try add(allocator, out, "R10-mode-role-contamination", true, "a Ref2VA brief declares a keyframe anchor", .{});
        }
    }
    if (hasRole(ctx, "replacement_subject") and std.mem.indexOf(u8, ret, "attribute_transfer") == null) {
        try add(allocator, out, "R31-replacement-not-recorded", true, "a replacement_subject is attached and retention_analysis has no attribute_transfer", .{});
    }
}

fn timeline(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), desc: []const u8, duration: f32, ctx: Context) !void {
    var nums: [8]u32 = undefined;
    var times: [8]f32 = undefined;
    var n_shots: u32 = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, desc, i, "[Shot ")) |pos| {
        const rest = desc[pos + 6 ..];
        var n: u32 = 0;
        var j: usize = 0;
        while (j < rest.len and rest[j] >= '0' and rest[j] <= '9') : (j += 1) {
            n = n * 10 + (rest[j] - '0');
        }
        if (j == 0 or j >= rest.len or rest[j] != ']') {
            i = pos + 6;
            continue;
        }
        const body = rest[j + 1 ..];
        const next = std.mem.indexOf(u8, body, "[Shot ") orelse body.len;
        const head = body[0..@min(next, 80)];
        if (n_shots < nums.len) {
            nums[n_shots] = n;
            times[n_shots] = -1;
            const at = skipScenetrans(head);
            if (std.mem.indexOf(u8, at, "At ")) |ap| {
                const ts_src = std.mem.trimStart(u8, at[ap + 3 ..], " ");
                if (n == 1) {
                    try add(allocator, out, "T2-shot1-timestamp", true, "[Shot 1] must not carry a timestamp", .{});
                } else if (grid.parseTimestamp(ts_src)) |t| {
                    times[n_shots] = t;
                } else {
                    try add(allocator, out, "T3-timestamp-format", true, "[Shot {d}] time must be MM:SS.mmm", .{n});
                }
            } else if (n != 1) {
                try add(allocator, out, "T4-missing-cut-time", true, "[Shot {d}] has no 'At MM:SS.mmm' cut time", .{n});
            }
            n_shots += 1;
        }
        i = pos + 6;
    }
    if (n_shots == 0) {
        try add(allocator, out, "T1-no-shot", true, "description has no [Shot N] marker", .{});
        return;
    }
    var contiguous = true;
    var k: u32 = 0;
    while (k < n_shots) : (k += 1) {
        if (nums[k] != k + 1) contiguous = false;
    }
    if (!contiguous) try add(allocator, out, "T8-shot-numbering", true, "shot numbers must be contiguous from 1", .{});
    if (ctx.pinned_shots) |pin| {
        if (n_shots != pin) try add(allocator, out, "T11-shot-count-pinned", true, "the caller asked for exactly {d} shot(s)", .{pin});
    }
    var prev: f32 = -1;
    k = 0;
    while (k < n_shots) : (k += 1) {
        if (times[k] < 0) continue;
        if (prev >= 0 and times[k] <= prev) {
            try add(allocator, out, "T5-non-increasing", true, "cut times are not strictly increasing", .{});
        }
        if (times[k] >= duration) {
            try add(allocator, out, "T6-time-past-end", true, "[Shot {d}] cuts at or beyond the real duration", .{nums[k]});
        }
        prev = times[k];
    }
    var marks: [10]f32 = undefined;
    var n_marks: u32 = 1;
    marks[0] = 0;
    k = 0;
    while (k < n_shots) : (k += 1) {
        if (times[k] >= 0 and n_marks < marks.len) {
            marks[n_marks] = times[k];
            n_marks += 1;
        }
    }
    if (n_marks < marks.len) {
        marks[n_marks] = duration;
        n_marks += 1;
    }
    k = 1;
    while (k < n_marks) : (k += 1) {
        const gap = marks[k] - marks[k - 1];
        if (gap >= 0 and gap < grid.min_shot_floor_s) {
            try add(allocator, out, "T12-cut-inside-the-floor", true, "a shot is shorter than 1.2s", .{});
            break;
        }
    }
    if (!grid.isOnGrid(duration) and !grid.isOnGrid(ctx.duration_s)) {
        try add(allocator, out, "T7-illegal-duration", true, "duration is not on the 17k+5 alignment grid", .{});
    }
}

fn cameraAndProse(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), desc: []const u8, text: []const u8, is_ref: bool) !void {
    if (is_ref) {
        const shot = std.mem.indexOf(u8, desc, "[Shot 1]");
        const style_line = if (shot) |s| std.mem.trim(u8, desc[0..s], " \t\r\n") else "";
        if (style_line.len == 0 or std.mem.indexOf(u8, style_line, "style") == null) {
            try add(allocator, out, "P1-no-style-opening", true, "Ref2VA needs a style sentence before [Shot 1]", .{});
        }
    }
    const has_camera_word = containsIgnoreCase(desc, "camera") or containsIgnoreCase(desc, "shot");
    const has_motion = hasCameraMotion(desc);
    if (!has_camera_word) {
        try add(allocator, out, "P4-no-camera-at-all", true, "the camera is never described", .{});
    } else if (!has_motion) {
        try add(allocator, out, "P5-camera-no-motion-type", true, "framing is described but no closed-vocabulary motion type appears", .{});
    }
    if (std.mem.indexOf(u8, desc, "Camera:") != null or std.mem.indexOf(u8, desc, "Camera :") != null) {
        try add(allocator, out, "R18-camera-as-label-stack", true, "camera is stated as a metadata header", .{});
    }
    if (bareSubject(desc)) {
        try add(allocator, out, "R19-bare-subject-name", true, "a subject is named without angle brackets", .{});
    }
    if (cameraContradicts(desc)) {
        try add(allocator, out, "R20-camera-contradiction", true, "the camera move contradicts its description", .{});
    }
    if (std.mem.indexOf(u8, desc, "with ") != null) {
        if (offAmplitude(desc) or offSpeed(desc)) {
            try add(allocator, out, "P8-camera-modifier-off-vocabulary", true, "the camera carries a modifier outside the closed vocabulary", .{});
        }
    }
    _ = text;
}

fn dialogueRules(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), desc: []const u8, text: []const u8, ctx: Context) !void {
    const open_n = countSub(desc, "<d>");
    const close_n = countSub(desc, "</d>");
    if (open_n != close_n) {
        try add(allocator, out, "D1-unbalanced-d", true, "<d> and </d> counts differ", .{});
    }
    for ([_][]const u8{ "<D>", "</D>", "< d >", "<d >", "< d>" }) |bad| {
        if (std.mem.indexOf(u8, text, bad) != null) {
            try add(allocator, out, "D5-marker-not-byte-exact", true, "dialogue marker is not exactly '<d>' / '</d>'", .{});
            break;
        }
    }
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, desc, i, "<d>")) |pos| {
        const end = std.mem.indexOfPos(u8, desc, pos + 3, "</d>") orelse break;
        const blk = desc[pos + 3 .. end];
        const t = std.mem.trimStart(u8, blk, " \t");
        if (t.len == 0 or t[0] != '[') {
            try add(allocator, out, "D2-missing-lang-tag", true, "<d> block lacks a [Language] tag", .{});
        }
        i = end + 4;
    }
    if (ctx.dialogue.len != 0) {
        for (ctx.dialogue) |line| {
            if (std.mem.indexOf(u8, desc, line) == null) {
                try add(allocator, out, "D4-dialogue-not-verbatim", true, "caller dialogue does not appear verbatim", .{});
                break;
            }
        }
    }
    if (containsIgnoreCase(desc, "voiceover") and std.mem.indexOf(u8, desc, "lips") == null) {
        try add(allocator, out, "D9-voiceover-no-lips-clause", true, "a voiceover must say the lips remain closed", .{});
    }
}

fn audioRules(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), desc: []const u8, sound: []const u8, music: []const u8, ctx: Context) !void {
    if (std.mem.indexOf(u8, sound, "<d>") != null or std.mem.indexOf(u8, music, "<d>") != null) {
        try add(allocator, out, "A3-dialogue-outside-desc", true, "dialogue markup belongs in the description, not the soundscape", .{});
    }
    const wants_score = containsIgnoreCase(desc, "score") or containsIgnoreCase(desc, "music") or ctx.creativity.len == 0;
    _ = wants_score;
    const silent_ask = ctx.forbids_score;
    if (silent_ask and !std.mem.eql(u8, music, "N/A") and music.len != 0) {
        try add(allocator, out, "A4-music-should-be-na", true, "the request forbids score so non_diegetic_music must be N/A", .{});
    }
}

fn creativityRules(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), desc: []const u8, text: []const u8, music: []const u8, ctx: Context) !void {
    const has_speech = std.mem.indexOf(u8, text, "<d>") != null;
    const has_score = !std.mem.eql(u8, music, "N/A") and music.len != 0;
    const has_text = countChar(desc, '"') >= 2;
    const speech_ok = permitsSpeech(ctx.creativity) or ctx.dialogue.len != 0 or ctx.n_audios != 0;
    if (has_speech and ctx.forbids_speech) {
        try add(allocator, out, "Q1-forbidden-element-present", true, "the request ruled out speech and the brief contains it", .{});
    } else if (has_speech and !speech_ok) {
        try add(allocator, out, "Q2-unlicensed-addition", true, "the brief adds speech the creativity setting does not license", .{});
    }
    if (has_score and ctx.forbids_score) {
        try add(allocator, out, "Q1-forbidden-element-present", true, "the request ruled out score and the brief contains it", .{});
    } else if (has_score and !permitsScore(ctx.creativity)) {
        try add(allocator, out, "Q2-unlicensed-addition", true, "the brief adds score the creativity setting does not license", .{});
    }
    if (has_text and ctx.forbids_text) {
        try add(allocator, out, "Q1-forbidden-element-present", true, "the request ruled out on-screen text and the brief contains it", .{});
    }
    if (std.mem.eql(u8, ctx.creativity, "extreme") and containsIgnoreCase(desc, "camera")) {
        if (std.mem.indexOf(u8, desc, "with large amplitude") == null and std.mem.indexOf(u8, desc, "at fast speed") == null) {
            try add(allocator, out, "Q3-extreme-not-honoured", true, "extreme requires 'with large amplitude' or 'at fast speed'", .{});
        }
    }
}

fn hygiene(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), text: []const u8, desc: []const u8) !void {
    for (leak_markers) |m| {
        if (std.mem.indexOf(u8, text, m) != null) {
            try add(allocator, out, "G1-reasoning-leaked", true, "leaked reasoning marker in the brief", .{});
            break;
        }
    }
    if (hasUnicodeHazard(text)) {
        try add(allocator, out, "H1-unicode-hazard", true, "structural text contains characters that change tokenization", .{});
    }
    _ = desc;
}

fn add(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), code: []const u8, err: bool, comptime fmt: []const u8, args: anytype) !void {
    try out.append(allocator, .{ .code = code, .err = err, .msg = try std.fmt.allocPrint(allocator, fmt, args) });
}

const Used = struct { picture: u16 = 0, video: u16 = 0, audio: u16 = 0, subject: u16 = 0 };

fn hasBit(bits: u16, n: u32) bool {
    if (n == 0 or n > 15) return false;
    return bits & (@as(u16, 1) << @intCast(n - 1)) != 0;
}

fn setBit(bits: *u16, n: u32) void {
    if (n == 0 or n > 15) return;
    bits.* |= @as(u16, 1) << @intCast(n - 1);
}

fn mediaFindings(allocator: std.mem.Allocator, out: *std.ArrayList(Finding), kind: []const u8, n_have: u32, bits: u16) !void {
    var over = false;
    var n: u32 = 1;
    while (n <= 15) : (n += 1) {
        if (hasBit(bits, n) and n > n_have) over = true;
    }
    if (over) try add(allocator, out, "L3-phantom-media", true, "a {s} is cited that was not attached", .{kind});
    if (n_have == 0) return;
    var missing: u32 = 0;
    n = 1;
    while (n <= n_have) : (n += 1) {
        if (!hasBit(bits, n)) missing += 1;
    }
    if (missing != 0) try add(allocator, out, "L4-unused-media", bits == 0, "attached {s}(s) never referenced", .{kind});
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
        if (std.mem.eql(u8, kind, "Picture")) setBit(&used.picture, n) else if (std.mem.eql(u8, kind, "Video")) setBit(&used.video, n) else if (std.mem.eql(u8, kind, "Audio")) setBit(&used.audio, n) else if (std.mem.eql(u8, kind, "Subject")) setBit(&used.subject, n) else {
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

fn firstLines(text: []const u8) struct { first: []const u8, second: []const u8, second_present: bool } {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return .{ .first = trimmed, .second = "", .second_present = false };
    const first = std.mem.trimEnd(u8, trimmed[0..nl], " \t\r");
    const rest = trimmed[nl + 1 ..];
    const nl2 = std.mem.indexOfScalar(u8, rest, '\n') orelse return .{ .first = first, .second = std.mem.trimEnd(u8, rest, " \t\r"), .second_present = true };
    return .{ .first = first, .second = std.mem.trimEnd(u8, rest[0..nl2], " \t\r"), .second_present = true };
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

fn openLabel(text: []const u8) ?[]const u8 {
    const kinds = [_][]const u8{ "Subject", "Picture", "Video", "Audio" };
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '<') continue;
        const rest = text[i + 1 ..];
        for (kinds) |k| {
            if (!std.mem.startsWith(u8, rest, k)) continue;
            if (rest.len <= k.len or rest[k.len] != ' ') continue;
            var j = k.len + 1;
            if (j >= rest.len or rest[j] < '0' or rest[j] > '9') continue;
            while (j < rest.len and rest[j] >= '0' and rest[j] <= '9') j += 1;
            if (j < rest.len and rest[j] >= '0' and rest[j] <= '9') continue;
            var k2 = j;
            while (k2 < rest.len and (rest[k2] == ' ' or rest[k2] == '\t')) k2 += 1;
            if (k2 < rest.len and rest[k2] == '>') continue;
            return k;
        }
    }
    return null;
}

fn firstBrackets(text: []const u8) ?[]const u8 {
    const t = std.mem.trimStart(u8, text, " \t\r\n");
    if (t.len == 0 or t[0] != '[') return null;
    const end = std.mem.indexOfScalar(u8, t, ']') orelse return null;
    return t[1..end];
}

fn isTaskType(p: []const u8) bool {
    return inList(p, &task_types);
}

fn taskBit(p: []const u8) u8 {
    for (task_types, 0..) |t, i| {
        if (std.mem.eql(u8, t, p)) return @as(u8, 1) << @intCast(i);
    }
    return 0;
}

fn allTaskParts(inner: []const u8) bool {
    var it = std.mem.splitScalar(u8, inner, '+');
    var any = false;
    while (it.next()) |raw| {
        const p = std.mem.trim(u8, raw, " \t");
        if (p.len == 0) continue;
        if (!isTaskType(p)) return false;
        any = true;
    }
    return any;
}

fn taskSetEqual(got: []const u8, want: []const []const u8) bool {
    for (want) |w| {
        if (!containsPhrase(got, w)) return false;
    }
    var it = std.mem.splitScalar(u8, got, '+');
    while (it.next()) |raw| {
        const p = std.mem.trim(u8, raw, " \t");
        if (p.len == 0) continue;
        var found = false;
        for (want) |w| {
            if (std.mem.eql(u8, w, p)) found = true;
        }
        if (!found) return false;
    }
    return true;
}

fn containsPhrase(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

fn scanDefinedSubjects(defs: []const u8, used: *Used) void {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, defs, i, "<Subject ")) |pos| {
        const rest = defs[pos + 9 ..];
        var n: u32 = 0;
        var j: usize = 0;
        while (j < rest.len and rest[j] >= '0' and rest[j] <= '9') : (j += 1) {
            n = n * 10 + (rest[j] - '0');
        }
        if (j != 0 and j < rest.len and rest[j] == '>' and std.mem.indexOf(u8, rest[j..@min(rest.len, j + 8)], " is") != null) {
            setBit(&used.subject, n);
        }
        i = pos + 9;
    }
}

fn kindOf(inner: []const u8) []const u8 {
    const sp = std.mem.indexOfScalar(u8, inner, ' ') orelse return inner;
    return inner[0..sp];
}

fn parseN(inner: []const u8) ?u32 {
    const sp = std.mem.indexOfScalar(u8, inner, ' ') orelse return null;
    return std.fmt.parseInt(u32, inner[sp + 1 ..], 10) catch null;
}

fn inList(item: []const u8, list: []const []const u8) bool {
    for (list) |x| {
        if (std.mem.eql(u8, x, item)) return true;
    }
    return false;
}

fn indexOfLabel(text: []const u8) ?usize {
    const kinds = [_][]const u8{ "<Subject ", "<Picture ", "<Video ", "<Audio " };
    for (kinds) |k| {
        if (std.mem.indexOf(u8, text, k)) |p| return p;
    }
    return null;
}

fn countLabelLines(ret: []const u8) ?[]const u8 {
    var seen: [16]u8 = .{0} ** 16;
    var labels: [16][]const u8 = .{""} ** 16;
    var n_lab: usize = 0;
    var it = std.mem.splitScalar(u8, ret, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len < 3 or t[0] != '<') continue;
        const close = std.mem.indexOfScalar(u8, t, '>') orelse continue;
        const lab = t[1..close];
        var idx: ?usize = null;
        for (labels[0..n_lab], 0..) |prev, i| {
            if (std.mem.eql(u8, prev, lab)) idx = i;
        }
        if (idx) |i| {
            seen[i] += 1;
            if (seen[i] > 1) return lab;
        } else if (n_lab < labels.len) {
            labels[n_lab] = lab;
            seen[n_lab] = 1;
            n_lab += 1;
        }
    }
    return null;
}

fn containsLabel(text: []const u8, label: []const u8) bool {
    return std.mem.indexOf(u8, text, label) != null;
}

fn hasSubjectLineWith(defs: []const u8, label: []const u8) bool {
    var it = std.mem.splitScalar(u8, defs, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, t, "<Subject ") and std.mem.indexOf(u8, t, label) != null) return true;
    }
    return false;
}

fn hasRole(ctx: Context, role: []const u8) bool {
    for (ctx.declared_roles) |d| {
        if (std.mem.eql(u8, d.role, role)) return true;
    }
    return false;
}

fn roleIs(ctx: Context, role: []const u8) bool {
    return hasRole(ctx, role);
}

fn skipScenetrans(head: []const u8) []const u8 {
    var t = std.mem.trimStart(u8, head, " \t,");
    while (std.mem.startsWith(u8, t, "<scenetrans>")) {
        t = std.mem.trimStart(u8, t["<scenetrans>".len..], " \t");
    }
    return t;
}

fn hasCameraMotion(desc: []const u8) bool {
    for (camera_stems) |stem| {
        if (containsIgnoreCase(desc, stem)) return true;
    }
    for (camera_types) |t| {
        if (containsIgnoreCase(desc, t)) return true;
    }
    return false;
}

fn containsIgnoreCase(hay: []const u8, needle: []const u8) bool {
    if (needle.len > hay.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(hay[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn bareSubject(desc: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, desc, i, "Subject ")) |pos| {
        if (pos == 0 or desc[pos - 1] != '<') {
            const rest = desc[pos + 8 ..];
            if (rest.len != 0 and rest[0] >= '0' and rest[0] <= '9') return true;
        }
        i = pos + 8;
    }
    return false;
}

fn cameraContradicts(desc: []const u8) bool {
    if (containsIgnoreCase(desc, "pushes in") and containsIgnoreCase(desc, "backward")) return true;
    if (containsIgnoreCase(desc, "pulls out") and containsIgnoreCase(desc, "closer")) return true;
    if (containsIgnoreCase(desc, "static shot") and (containsIgnoreCase(desc, "pans") or containsIgnoreCase(desc, "trucks") or containsIgnoreCase(desc, "tilts"))) return true;
    return false;
}

fn offAmplitude(desc: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, desc, i, "with ")) |pos| {
        const rest = desc[pos + 5 ..];
        const amp_at = std.mem.indexOf(u8, rest[0..@min(rest.len, 24)], " amplitude") orelse {
            i = pos + 5;
            continue;
        };
        const word = std.mem.trim(u8, rest[0..amp_at], " ");
        if (!std.mem.eql(u8, word, "small") and !std.mem.eql(u8, word, "large")) return true;
        i = pos + 5;
    }
    return false;
}

fn offSpeed(desc: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, desc, i, "at ")) |pos| {
        const rest = desc[pos + 3 ..];
        const sp = std.mem.indexOf(u8, rest[0..@min(rest.len, 20)], " speed") orelse {
            i = pos + 3;
            continue;
        };
        const word = std.mem.trim(u8, rest[0..sp], " ");
        const prev = desc[0..pos];
        if (std.mem.endsWith(u8, std.mem.trimEnd(u8, prev, " "), "amplitude") or hasCameraMotion(prev[prev.len -| 80..])) {
            if (!std.mem.eql(u8, word, "slow") and !std.mem.eql(u8, word, "fast") and !std.mem.eql(u8, word, "a slow") and !std.mem.eql(u8, word, "a fast")) return true;
        }
        i = pos + 3;
    }
    return false;
}

fn countSub(hay: []const u8, needle: []const u8) u32 {
    var n: u32 = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, hay, i, needle)) |pos| {
        n += 1;
        i = pos + needle.len;
    }
    return n;
}

fn countChar(hay: []const u8, c: u8) u32 {
    var n: u32 = 0;
    for (hay) |ch| {
        if (ch == c) n += 1;
    }
    return n;
}

fn hasUnicodeHazard(text: []const u8) bool {
    var it = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
    while (it.nextCodepoint()) |cp| {
        for (unicode_hazards) |h| {
            if (cp == h) return true;
        }
    }
    return false;
}

fn permitsSpeech(creativity: []const u8) bool {
    return std.mem.eql(u8, creativity, "bold") or std.mem.eql(u8, creativity, "extreme");
}

fn permitsScore(creativity: []const u8) bool {
    return !std.mem.eql(u8, creativity, "restrained");
}

fn permitsText(creativity: []const u8) bool {
    return std.mem.eql(u8, creativity, "bold") or std.mem.eql(u8, creativity, "extreme");
}
