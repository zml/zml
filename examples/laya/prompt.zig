//! Laya prompt construction and output calibration.
//!
//! Mirrors the reference runtime (convaiinnovations/laya, `rl_common.py`):
//!   [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! Each option's [MASK] position is the marker the decision head scores.

const std = @import("std");

const zml = @import("zml");

const AgentConfig = @import("model.zig").AgentConfig;

pub const QuestionType = enum(u32) {
    choice = 0,
    score = 1,
    noul = 2,
};

pub const Question = struct {
    id: []const u8,
    type: QuestionType,
    instructions: []const u8,
    /// Choice labels, in option order. Empty for score and noul.
    labels: []const []const u8,
    /// Option texts shown to the model, in option order.
    options: []const []const u8,
    /// Score level descriptions, echoed back as the legend.
    legend: []const []const u8,
};

pub const Encoded = struct {
    ids: []u32,
    markers: []u32,
};

pub const SpecialTokens = struct {
    cls: u32,
    sep: u32,
    pad: u32,
    mask: u32,

    pub fn init(tokenizer: *const zml.tokenizer.Tokenizer) !SpecialTokens {
        return .{
            .cls = tokenizer.tokenId("[CLS]") orelse return error.MissingClsToken,
            .sep = tokenizer.tokenId("[SEP]") orelse return error.MissingSepToken,
            .pad = tokenizer.tokenId("[PAD]") orelse return error.MissingPadToken,
            .mask = tokenizer.tokenId("[MASK]") orelse return error.MissingMaskToken,
        };
    }
};

/// Parses `{ "<id>": { "type": ..., "instructions": ..., "criteria": ... }, ... }`.
pub fn parseQuestions(arena: std.mem.Allocator, value: std.json.Value) ![]Question {
    if (value != .object) return error.QuestionsMustBeAnObject;
    const questions = try arena.alloc(Question, value.object.count());
    var it = value.object.iterator();
    var i: usize = 0;
    while (it.next()) |entry| : (i += 1) {
        questions[i] = try parseQuestion(arena, entry.key_ptr.*, entry.value_ptr.*);
    }
    return questions;
}

fn parseQuestion(arena: std.mem.Allocator, id: []const u8, def: std.json.Value) !Question {
    if (def != .object) return error.QuestionMustBeAnObject;
    const type_value = def.object.get("type") orelse return error.QuestionMissingType;
    if (type_value != .string) return error.InvalidQuestionType;
    const qtype = std.meta.stringToEnum(QuestionType, type_value.string) orelse return error.InvalidQuestionType;

    const instructions_value = def.object.get("instructions") orelse return error.QuestionMissingInstructions;
    const instructions = switch (instructions_value) {
        .string => |s| s,
        else => try renderJson(arena, instructions_value),
    };

    const criteria = def.object.get("criteria") orelse .null;
    var labels: std.ArrayList([]const u8) = .empty;
    var options: std.ArrayList([]const u8) = .empty;
    var legend: std.ArrayList([]const u8) = .empty;

    switch (qtype) {
        .choice => switch (criteria) {
            .array => |items| for (items.items) |item| {
                if (item != .string) return error.ChoiceLabelsMustBeStrings;
                try labels.append(arena, item.string);
                try options.append(arena, item.string);
            },
            .object => |object| {
                var it = object.iterator();
                while (it.next()) |entry| {
                    const label = entry.key_ptr.*;
                    try labels.append(arena, label);
                    const description = entry.value_ptr.*;
                    const empty = description == .null or (description == .string and description.string.len == 0);
                    try options.append(arena, if (empty) label else try std.fmt.allocPrint(arena, "{s}: {s}", .{ label, try renderCriterion(arena, description) }));
                }
            },
            else => return error.ChoiceCriteriaMustBeAListOrObject,
        },
        .score => {
            if (criteria != .array or criteria.array.items.len == 0) return error.ScoreCriteriaMustBeAList;
            for (criteria.array.items, 0..) |level, index| {
                const text = try renderCriterion(arena, level);
                try legend.append(arena, text);
                try options.append(arena, try std.fmt.allocPrint(arena, "level {d}: {s}", .{ index, text }));
            }
        },
        .noul => {
            const object: ?std.json.ObjectMap = switch (criteria) {
                .null => null,
                .object => |o| o,
                else => return error.NoulCriteriaMustBeAnObject,
            };
            const Default = struct { key: []const u8, text: []const u8 };
            const defaults = [_]Default{
                .{ .key = "false", .text = "no, the statement does not hold" },
                .{ .key = "true", .text = "yes, the statement holds" },
            };
            for (defaults) |d| {
                const given: ?std.json.Value = if (object) |o| o.get(d.key) else null;
                const empty = given == null or given.? == .null or (given.? == .string and given.?.string.len == 0);
                const text = if (empty) d.text else try renderCriterion(arena, given.?);
                try options.append(arena, try std.fmt.allocPrint(arena, "{s}: {s}", .{ d.key, text }));
            }
        },
    }
    if (options.items.len == 0) return error.QuestionHasNoOptions;

    return .{
        .id = id,
        .type = qtype,
        .instructions = instructions,
        .labels = labels.items,
        .options = options.items,
        .legend = legend.items,
    };
}

/// Serializes the state the same way as Python `json.dumps(state, ensure_ascii=False)`.
pub fn renderState(arena: std.mem.Allocator, state: std.json.Value) ![]const u8 {
    return switch (state) {
        .string => |s| s,
        else => renderJson(arena, state),
    };
}

fn renderCriterion(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => renderJson(arena, value),
    };
}

/// JSON with Python's default `", "` / `": "` separators and raw UTF-8.
fn renderJson(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    try writeJson(&out.writer, value);
    return out.written();
}

fn writeJson(w: *std.Io.Writer, value: std.json.Value) !void {
    switch (value) {
        .object => |object| {
            try w.writeByte('{');
            var it = object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try w.writeAll(", ");
                first = false;
                try w.print("{f}: ", .{std.json.fmt(entry.key_ptr.*, .{})});
                try writeJson(w, entry.value_ptr.*);
            }
            try w.writeByte('}');
        },
        .array => |array| {
            try w.writeByte('[');
            for (array.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try writeJson(w, item);
            }
            try w.writeByte(']');
        },
        else => try w.print("{f}", .{std.json.fmt(value, .{})}),
    }
}

pub const Builder = struct {
    arena: std.mem.Allocator,
    encoder: *zml.tokenizer.Tokenizer.Encoder,
    special: SpecialTokens,
    max_len: usize,
    head_max_len: usize,

    const max_option_tokens = 48;

    /// Tokenizes without special tokens; [MASK] in user text is neutralized.
    fn tokenize(self: Builder, text: []const u8) ![]const u32 {
        const clean = try std.mem.replaceOwned(u8, self.arena, text, "[MASK]", " ");
        self.encoder.reset();
        var ids = try self.encoder.encodeAlloc(self.arena, clean);
        // Drop any post-processor specials so we control the layout.
        if (ids.len > 0 and ids[0] == self.special.cls) ids = ids[1..];
        if (ids.len > 0 and ids[ids.len - 1] == self.special.sep) ids = ids[0 .. ids.len - 1];
        return ids;
    }

    pub fn build(self: Builder, question: Question, state: []const u8) !Encoded {
        const head_text = try std.fmt.allocPrint(self.arena, "{s} question: {s}", .{ @tagName(question.type), question.instructions });
        var head_ids = try self.tokenize(head_text);

        const option_ids = try self.arena.alloc([]const u32, question.options.len);
        var options_len: usize = 0;
        for (question.options, option_ids) |option, *ids| {
            const text = try std.fmt.allocPrint(self.arena, " {s}", .{option});
            const tokens = try self.tokenize(text);
            const body = tokens[0..@min(tokens.len, max_option_tokens)];
            const with_marker = try self.arena.alloc(u32, body.len + 1);
            with_marker[0] = self.special.mask;
            @memcpy(with_marker[1..], body);
            ids.* = with_marker;
            options_len += with_marker.len;
        }

        // Keep at least 16 tokens for the instructions when options are long.
        const head_max: i64 = @intCast(self.head_max_len);
        var budget: i64 = head_max - @as(i64, @intCast(options_len));
        if (budget < 16) {
            const per: usize = @intCast(@max(4, @divFloor(head_max - 16, @as(i64, @intCast(@max(1, option_ids.len))))));
            options_len = 0;
            for (option_ids) |*ids| {
                ids.* = ids.*[0..@min(ids.len, per)];
                options_len += ids.len;
            }
            budget = head_max - @as(i64, @intCast(options_len));
        }
        head_ids = head_ids[0..@min(head_ids.len, @as(usize, @intCast(@max(8, budget))))];

        var ids: std.ArrayList(u32) = .empty;
        var markers: std.ArrayList(u32) = .empty;
        try ids.append(self.arena, self.special.cls);
        try ids.appendSlice(self.arena, head_ids);
        try ids.append(self.arena, self.special.sep);
        for (option_ids) |option| {
            try markers.append(self.arena, @intCast(ids.items.len));
            try ids.appendSlice(self.arena, option);
        }
        try ids.append(self.arena, self.special.sep);

        const room = self.max_len -| (ids.items.len + 1);
        const state_ids = try self.tokenize(state);
        try ids.appendSlice(self.arena, state_ids[0..@min(state_ids.len, room)]);
        try ids.append(self.arena, self.special.sep);

        const final_len = @min(ids.items.len, self.max_len);
        var kept: usize = 0;
        for (markers.items) |m| {
            if (m < final_len) kept += 1;
        }
        if (kept != question.options.len) return error.TooManyOptionsForTokenBudget;

        return .{ .ids = ids.items[0..final_len], .markers = markers.items };
    }
};

pub const Calibration = struct {
    temperature: [3]f32,
    by_options: std.json.ArrayHashMap(f32),

    // Temperatures < 1 sharpen instead of soften; the shipped `choice:11+` bucket (0.10)
    // would turn a coin flip into 99% confidence. The reference runtime clamps too.
    const min_temperature = 0.5;
    const max_temperature = 5.0;

    pub fn init(config: AgentConfig) Calibration {
        return .{ .temperature = config.temperature, .by_options = config.temperature_by_options };
    }

    fn clamp(t: f32) f32 {
        if (!std.math.isFinite(t)) return 1.0;
        return std.math.clamp(t, min_temperature, max_temperature);
    }

    pub fn temperatureFor(self: Calibration, qtype: QuestionType, k: usize) f32 {
        const size = if (k <= 2) "2" else if (k <= 5) "3-5" else if (k <= 10) "6-10" else "11+";
        var buf: [32]u8 = undefined;
        const bucket = std.fmt.bufPrint(&buf, "{s}:{s}", .{ @tagName(qtype), size }) catch unreachable;
        const t = self.by_options.map.get(bucket) orelse self.temperature[@intFromEnum(qtype)];
        return clamp(t);
    }
};

pub const Answer = struct {
    question: Question,
    probabilities: []f32,
    confidence: f32,
    act_probability: f32,
    /// Expected level for score questions.
    score: f32 = 0,
    /// P(true) for noul questions.
    noul: f32 = 0,
    /// Arg-max option index.
    best: usize,
};

/// Turns raw marker logits and act logits into a calibrated answer.
pub fn decide(arena: std.mem.Allocator, calibration: Calibration, question: Question, logits: []const f32, act_logits: []const f32) !Answer {
    const k = question.options.len;
    const scale = calibration.temperatureFor(question.type, k);
    const p = try arena.alloc(f32, k);
    softmaxScaled(logits[0..k], scale, p);

    var best: usize = 0;
    for (p, 0..) |v, i| {
        if (v > p[best]) best = i;
    }

    const act = try arena.alloc(f32, act_logits.len);
    softmaxScaled(act_logits, 1, act);

    var answer: Answer = .{
        .question = question,
        .probabilities = p,
        .confidence = entropyConfidence(p),
        .act_probability = act[0],
        .best = best,
    };
    switch (question.type) {
        .choice => {},
        .score => for (p, 0..) |v, i| {
            answer.score += @as(f32, @floatFromInt(i)) * v;
        },
        .noul => {
            answer.noul = p[1];
            answer.confidence = @max(p[1], 1 - p[1]);
        },
    }
    return answer;
}

fn softmaxScaled(logits: []const f32, temperature: f32, out: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (logits) |v| max = @max(max, v / temperature);
    var total: f32 = 0;
    for (logits, out) |v, *o| {
        o.* = @exp(v / temperature - max);
        total += o.*;
    }
    for (out) |*o| o.* /= total;
}

/// 1 - H(p) / log(k): 1 when certain, 0 when uniform.
fn entropyConfidence(p: []const f32) f32 {
    if (p.len < 2) return 1;
    var entropy: f32 = 0;
    for (p) |v| entropy -= v * @log(std.math.clamp(v, 1e-12, 1));
    return std.math.clamp(1 - entropy / @log(@as(f32, @floatFromInt(p.len))), 0, 1);
}

/// Writes the answers in the reference runtime's response format.
pub fn writeResponse(w: *std.Io.Writer, answers: []const Answer, input_tokens: usize, latency_ms: f64) !void {
    try w.writeAll("{\"model\": \"laya-rl-agent\", \"answers\": {");
    for (answers, 0..) |a, i| {
        if (i > 0) try w.writeAll(", ");
        const q = a.question;
        try w.print("{f}: {{\"type\": \"{s}\", \"confidence\": {d:.4}, \"action\": {{\"act_probability\": {d:.4}}}", .{
            std.json.fmt(q.id, .{}), @tagName(q.type), a.confidence, a.act_probability,
        });
        switch (q.type) {
            .choice => {
                try w.print(", \"choice\": {f}, \"probabilities\": {{", .{std.json.fmt(q.labels[a.best], .{})});
                for (q.labels, a.probabilities, 0..) |label, p, j| {
                    if (j > 0) try w.writeAll(", ");
                    try w.print("{f}: {d:.4}", .{ std.json.fmt(label, .{}), p });
                }
                try w.writeByte('}');
            },
            .score => {
                try w.print(", \"score\": {d:.4}, \"legend\": {{", .{a.score});
                for (q.legend, 0..) |text, j| {
                    if (j > 0) try w.writeAll(", ");
                    try w.print("\"{d}\": {f}", .{ j, std.json.fmt(text, .{}) });
                }
                try w.writeAll("}, \"probabilities\": {");
                for (a.probabilities, 0..) |p, j| {
                    if (j > 0) try w.writeAll(", ");
                    try w.print("\"{d}\": {d:.4}", .{ j, p });
                }
                try w.writeByte('}');
            },
            .noul => try w.print(", \"noul\": {d:.4}", .{a.noul}),
        }
        try w.writeByte('}');
    }
    try w.print("}}, \"usage\": {{\"input_tokens\": {d}, \"output_tokens\": 0}}, \"latency_ms\": {d:.1}}}", .{ input_tokens, latency_ms });
}

test {
    // Pulls in the runfiles initializer exported by zml's bazel module.
    _ = zml.bazel;
}

test entropyConfidence {
    try std.testing.expectApproxEqAbs(@as(f32, 0), entropyConfidence(&.{ 0.5, 0.5 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), entropyConfidence(&.{ 1, 0 }), 1e-4);
}

test "calibration clamps sharpening temperatures" {
    var by_options: std.json.ArrayHashMap(f32) = .{};
    defer by_options.deinit(std.testing.allocator);
    try by_options.map.put(std.testing.allocator, "choice:11+", 0.1);
    const calibration: Calibration = .{ .temperature = .{ 1.6, 1.25, 2.0 }, .by_options = by_options };
    try std.testing.expectEqual(@as(f32, 0.5), calibration.temperatureFor(.choice, 12));
    try std.testing.expectEqual(@as(f32, 1.6), calibration.temperatureFor(.choice, 4));
}

test parseQuestions {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"sentiment": {"type": "choice", "instructions": "Tone?", "criteria": {"positive": null, "negative": "unhappy"}},
        \\ "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["none", "high"]},
        \\ "refund": {"type": "noul", "instructions": "Asks for a refund"}}
    ;
    const value = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{});
    const questions = try parseQuestions(arena.allocator(), value);
    try std.testing.expectEqual(3, questions.len);
    try std.testing.expectEqualStrings("positive", questions[0].options[0]);
    try std.testing.expectEqualStrings("negative: unhappy", questions[0].options[1]);
    try std.testing.expectEqualStrings("level 1: high", questions[1].options[1]);
    try std.testing.expectEqualStrings("true: yes, the statement holds", questions[2].options[1]);
}
