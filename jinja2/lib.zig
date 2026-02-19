const std = @import("std");

pub const Environment = @import("environment.zig").Environment;
const expr = @import("expr.zig");
pub const Template = @import("template.zig").Template;
pub const Value = @import("value.zig").Value;

fn testAllocator() std.mem.Allocator {
    return std.testing.allocator;
}

fn jsonToValue(allocator: std.mem.Allocator, json_value: std.json.Value) !Value {
    return switch (json_value) {
        .null => .null,
        .bool => |v| Value.fromBool(v),
        .integer => |v| Value.fromInt(v),
        .float => |v| .{ .float = v },
        .string => |v| Value.fromString(v),
        .number_string => |v| Value.fromString(v),
        .array => |arr| blk: {
            const out = try allocator.alloc(Value, arr.items.len);
            for (arr.items, 0..) |item, i| {
                out[i] = try jsonToValue(allocator, item);
            }
            break :blk Value.fromList(out);
        },
        .object => |obj| blk: {
            const out = try allocator.alloc(Value.Entry, obj.count());
            var it = obj.iterator();
            var i: usize = 0;
            while (it.next()) |entry| {
                out[i] = .{
                    .key = entry.key_ptr.*,
                    .value = try jsonToValue(allocator, entry.value_ptr.*),
                };
                i += 1;
            }
            break :blk Value.fromMap(out);
        },
    };
}

fn freeNameList(allocator: std.mem.Allocator, list: *std.ArrayList([]const u8)) void {
    for (list.items) |name| allocator.free(name);
    list.deinit(allocator);
}

fn listFixtureFiles(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    extension: []const u8,
) !std.ArrayList([]const u8) {
    var out = std.ArrayList([]const u8).empty;
    errdefer freeNameList(allocator, &out);

    var dir = try std.Io.Dir.openDir(.cwd(), io, dir_path, .{ .iterate = true });
    defer dir.close(io);

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, extension)) continue;
        try out.append(allocator, try allocator.dupe(u8, entry.name));
    }

    std.mem.sort([]const u8, out.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    return out;
}

fn filePrefixId(name: []const u8) []const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse name.len;
    const stem = name[0..dot];
    const underscore = std.mem.indexOfScalar(u8, stem, '_') orelse stem.len;
    return stem[0..underscore];
}

fn parseExpectedCaseIds(name: []const u8) !struct { template_id: []const u8, data_id: []const u8 } {
    if (!std.mem.endsWith(u8, name, ".txt")) return error.InvalidExpectedFixture;
    const stem = name[0 .. name.len - 4];
    const sep = std.mem.indexOfScalar(u8, stem, '_') orelse return error.InvalidExpectedFixture;
    const template_id = stem[0..sep];
    const data_id = stem[sep + 1 ..];
    if (template_id.len == 0 or data_id.len == 0) return error.InvalidExpectedFixture;
    return .{ .template_id = template_id, .data_id = data_id };
}

fn findFileById(files: []const []const u8, wanted_id: []const u8) ?[]const u8 {
    for (files) |name| {
        if (std.mem.eql(u8, filePrefixId(name), wanted_id)) return name;
    }
    return null;
}

fn resolveSuiteRootPath(
    io: std.Io,
    allocator: std.mem.Allocator,
    suite: []const u8,
) !struct { path: []const u8, owned: ?[]u8 } {
    const p1 = try std.fmt.allocPrint(allocator, "test_cases/{s}", .{suite});
    errdefer allocator.free(p1);
    if (std.Io.Dir.openDir(.cwd(), io, p1, .{})) |dir| {
        dir.close(io);
        return .{ .path = p1, .owned = p1 };
    } else |_| {
        allocator.free(p1);
    }

    const p2 = try std.fmt.allocPrint(allocator, "jinja2/test_cases/{s}", .{suite});
    errdefer allocator.free(p2);
    if (std.Io.Dir.openDir(.cwd(), io, p2, .{})) |dir| {
        dir.close(io);
        return .{ .path = p2, .owned = p2 };
    } else |_| {
        allocator.free(p2);
    }

    const test_srcdir = std.c.getenv("TEST_SRCDIR") orelse return error.FileNotFound;
    const test_workspace = std.c.getenv("TEST_WORKSPACE") orelse return error.FileNotFound;
    const p3 = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}/jinja2/test_cases/{s}",
        .{ std.mem.span(test_srcdir), std.mem.span(test_workspace), suite },
    );
    errdefer allocator.free(p3);
    if (std.Io.Dir.openDir(.cwd(), io, p3, .{})) |dir| {
        dir.close(io);
        return .{ .path = p3, .owned = p3 };
    } else |_| {
        allocator.free(p3);
    }

    return error.FileNotFound;
}

fn resolveExpectedDirPath(
    io: std.Io,
    allocator: std.mem.Allocator,
    suite: []const u8,
    suite_root_path: []const u8,
) !struct { path: []const u8, owned: ?[]u8 } {
    const p1 = try std.fmt.allocPrint(allocator, "test_cases_generated/{s}", .{suite});
    errdefer allocator.free(p1);
    if (std.Io.Dir.openDir(.cwd(), io, p1, .{})) |dir| {
        dir.close(io);
        return .{ .path = p1, .owned = p1 };
    } else |_| {
        allocator.free(p1);
    }

    const p2 = try std.fmt.allocPrint(allocator, "jinja2/test_cases_generated/{s}", .{suite});
    errdefer allocator.free(p2);
    if (std.Io.Dir.openDir(.cwd(), io, p2, .{})) |dir| {
        dir.close(io);
        return .{ .path = p2, .owned = p2 };
    } else |_| {
        allocator.free(p2);
    }

    if (std.c.getenv("TEST_SRCDIR")) |test_srcdir| {
        if (std.c.getenv("TEST_WORKSPACE")) |test_workspace| {
            const p3 = try std.fmt.allocPrint(
                allocator,
                "{s}/{s}/jinja2/test_cases_generated/{s}",
                .{ std.mem.span(test_srcdir), std.mem.span(test_workspace), suite },
            );
            errdefer allocator.free(p3);
            if (std.Io.Dir.openDir(.cwd(), io, p3, .{})) |dir| {
                dir.close(io);
                return .{ .path = p3, .owned = p3 };
            } else |_| {
                allocator.free(p3);
            }
        }
    }

    const fallback_with_expected = try std.fmt.allocPrint(allocator, "{s}/expected", .{suite_root_path});
    errdefer allocator.free(fallback_with_expected);
    if (std.Io.Dir.openDir(.cwd(), io, fallback_with_expected, .{})) |dir| {
        dir.close(io);
        return .{ .path = fallback_with_expected, .owned = fallback_with_expected };
    } else |_| {
        allocator.free(fallback_with_expected);
    }

    const fallback = try allocator.dupe(u8, suite_root_path);
    return .{ .path = fallback, .owned = fallback };
}

fn runChatLikeFixtureSuite(suite: []const u8) !void {
    var threaded: std.Io.Threaded = .init(testAllocator(), .{});
    defer threaded.deinit();
    const io = threaded.io();

    const suite_root = try resolveSuiteRootPath(io, testAllocator(), suite);
    defer if (suite_root.owned) |p| testAllocator().free(p);

    const templates_dir = try std.fmt.allocPrint(testAllocator(), "{s}/templates", .{suite_root.path});
    defer testAllocator().free(templates_dir);
    const data_dir = try std.fmt.allocPrint(testAllocator(), "{s}/data", .{suite_root.path});
    defer testAllocator().free(data_dir);
    const expected_dir = try resolveExpectedDirPath(io, testAllocator(), suite, suite_root.path);
    defer if (expected_dir.owned) |p| testAllocator().free(p);

    var template_names = try listFixtureFiles(io, testAllocator(), templates_dir, ".jinja");
    defer freeNameList(testAllocator(), &template_names);
    var data_names = try listFixtureFiles(io, testAllocator(), data_dir, ".json");
    defer freeNameList(testAllocator(), &data_names);
    var expected_names = try listFixtureFiles(io, testAllocator(), expected_dir.path, ".txt");
    defer freeNameList(testAllocator(), &expected_names);

    for (expected_names.items) |expected_name| {
        const ids = parseExpectedCaseIds(expected_name) catch {
            std.debug.print("{s} fixture {s}: expected filename must be <templateid>_<dataid>.txt\n", .{ suite, expected_name });
            return error.InvalidExpectedFixture;
        };

        const template_name = findFileById(template_names.items, ids.template_id) orelse {
            std.debug.print("{s} fixture {s}: missing template id {s}\n", .{ suite, expected_name, ids.template_id });
            return error.InvalidExpectedFixture;
        };
        const data_name = findFileById(data_names.items, ids.data_id) orelse {
            std.debug.print("{s} fixture {s}: missing data id {s}\n", .{ suite, expected_name, ids.data_id });
            return error.InvalidExpectedFixture;
        };

        var arena = std.heap.ArenaAllocator.init(testAllocator());
        defer arena.deinit();
        const allocator = arena.allocator();

        const template_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ templates_dir, template_name });
        const data_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ data_dir, data_name });
        const expected_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ expected_dir.path, expected_name });

        const template_src = try std.Io.Dir.cwd().readFileAlloc(io, template_path, allocator, .limited(1 << 20));
        const data_src = try std.Io.Dir.cwd().readFileAlloc(io, data_path, allocator, .limited(1 << 20));
        const expected_src = try std.Io.Dir.cwd().readFileAlloc(io, expected_path, allocator, .limited(1 << 20));

        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, allocator, data_src, .{});
        if (parsed != .object) {
            std.debug.print("{s} fixture {s}: data JSON root must be object\n", .{ suite, data_name });
            return error.InvalidChatDataFixture;
        }

        const ctx_value = try jsonToValue(allocator, parsed);
        if (ctx_value != .map) {
            std.debug.print("{s} fixture {s}: data must convert to map\n", .{ suite, data_name });
            return error.InvalidChatDataFixture;
        }

        var env = Environment.init(allocator);
        defer env.deinit();
        env.addTemplate("fixture", template_src) catch |err| {
            std.debug.print("{s} fixture {s}: addTemplate failed ({s})\n", .{ suite, expected_name, @errorName(err) });
            return err;
        };

        const rendered = env.renderTemplate(testAllocator(), "fixture", ctx_value.map) catch |err| {
            std.debug.print("{s} fixture {s}: renderTemplate failed ({s})\n", .{ suite, expected_name, @errorName(err) });
            return err;
        };
        defer testAllocator().free(rendered);

        if (!std.mem.eql(u8, expected_src, rendered)) {
            std.debug.print("{s} fixture mismatch: {s} (template={s} data={s})\n", .{ suite, expected_name, template_name, data_name });
        }
        try std.testing.expectEqualStrings(expected_src, rendered);
    }
}

test "basic fixture corpus renders as expected" {
    var threaded: std.Io.Threaded = .init(testAllocator(), .{});
    defer threaded.deinit();
    const io = threaded.io();

    var basic_dir: std.Io.Dir = undefined;
    var base_path: []const u8 = "";
    var owned_base_path: ?[]u8 = null;
    defer if (owned_base_path) |p| testAllocator().free(p);
    if (std.Io.Dir.openDir(.cwd(), io, "test_cases/basic", .{ .iterate = true })) |dir| {
        basic_dir = dir;
        base_path = "test_cases/basic";
    } else |_| if (std.Io.Dir.openDir(.cwd(), io, "jinja2/test_cases/basic", .{ .iterate = true })) |dir| {
        basic_dir = dir;
        base_path = "jinja2/test_cases/basic";
    } else |_| {
        const test_srcdir = std.c.getenv("TEST_SRCDIR") orelse return error.FileNotFound;
        const test_workspace = std.c.getenv("TEST_WORKSPACE") orelse return error.FileNotFound;
        const runfiles_path = try std.fmt.allocPrint(
            testAllocator(),
            "{s}/{s}/jinja2/test_cases/basic",
            .{ std.mem.span(test_srcdir), std.mem.span(test_workspace) },
        );
        errdefer testAllocator().free(runfiles_path);
        basic_dir = try std.Io.Dir.openDir(.cwd(), io, runfiles_path, .{ .iterate = true });
        owned_base_path = runfiles_path;
        base_path = runfiles_path;
    }
    defer basic_dir.close(io);

    const expected_dir = try resolveExpectedDirPath(io, testAllocator(), "basic", base_path);
    defer if (expected_dir.owned) |p| testAllocator().free(p);

    var names = std.ArrayList([]const u8).empty;
    defer names.deinit(testAllocator());

    var iter = basic_dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        try names.append(testAllocator(), try testAllocator().dupe(u8, entry.name));
    }
    defer {
        for (names.items) |name| testAllocator().free(name);
    }

    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    for (names.items) |name| {
        var arena = std.heap.ArenaAllocator.init(testAllocator());
        defer arena.deinit();
        const allocator = arena.allocator();

        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ base_path, name });
        const case_src = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 20));

        const expected_name = blk: {
            const stem = name[0 .. name.len - ".json".len];
            break :blk try std.fmt.allocPrint(allocator, "{s}.txt", .{stem});
        };
        const expected_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ expected_dir.path, expected_name });
        const expected_src = try std.Io.Dir.cwd().readFileAlloc(io, expected_path, allocator, .limited(1 << 20));

        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, allocator, case_src, .{});
        if (parsed != .object) {
            std.debug.print("basic fixture {s}: expected root JSON object\n", .{name});
            return error.InvalidBasicFixture;
        }

        const template_json = parsed.object.get("template") orelse {
            std.debug.print("basic fixture {s}: missing template\n", .{name});
            return error.InvalidBasicFixture;
        };
        const data_json = parsed.object.get("data") orelse {
            std.debug.print("basic fixture {s}: missing data\n", .{name});
            return error.InvalidBasicFixture;
        };
        if (template_json != .string) {
            std.debug.print("basic fixture {s}: template must be string\n", .{name});
            return error.InvalidBasicFixture;
        }

        const ctx_value = try jsonToValue(allocator, data_json);
        if (ctx_value != .map) {
            std.debug.print("basic fixture {s}: data must convert to map\n", .{name});
            return error.InvalidBasicFixture;
        }

        var env = Environment.init(allocator);
        defer env.deinit();
        env.addTemplate("fixture", template_json.string) catch |err| {
            std.debug.print("basic fixture {s}: addTemplate failed: {s}\n", .{ name, @errorName(err) });
            return err;
        };

        const rendered = env.renderTemplate(testAllocator(), "fixture", ctx_value.map) catch |err| {
            std.debug.print("basic fixture {s}: renderTemplate failed: {s}\n", .{ name, @errorName(err) });
            return err;
        };
        defer testAllocator().free(rendered);

        if (!std.mem.eql(u8, expected_src, rendered)) {
            std.debug.print("basic fixture mismatch: {s}\n", .{name});
        }
        try std.testing.expectEqualStrings(expected_src, rendered);
    }
}

test "chat fixture corpus renders as expected" {
    try runChatLikeFixtureSuite("chat");
}

test "multimodal fixture corpus renders as expected" {
    try runChatLikeFixtureSuite("multimodal");
}

test "expression lexer and parser basic arithmetic" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    const parsed = try expr.parseExpr(allocator, "1 + 2 * 3");

    var scope = expr.Scope.init(allocator);
    defer scope.deinit();
    const runtime = expr.Runtime{};

    const result = try expr.evalExpr(allocator, parsed, &scope, &runtime);
    try std.testing.expectEqual(@as(i64, 7), result.int);
}

test "render text and variable" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var env = Environment.init(allocator);
    defer env.deinit();

    try env.addTemplate("hello", "Hello {{ name }}!");

    const ctx = [_]Value.Entry{
        .{ .key = "name", .value = Value.fromString("World") },
    };

    const output = try env.renderTemplate(allocator, "hello", &ctx);
    defer allocator.free(output);

    try std.testing.expectEqualStrings("Hello World!", output);
}

test "if else rendering" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var env = Environment.init(allocator);
    defer env.deinit();

    try env.addTemplate("if", "{% if enabled %}on{% else %}off{% endif %}");

    const true_ctx = [_]Value.Entry{
        .{ .key = "enabled", .value = Value.fromBool(true) },
    };
    const false_ctx = [_]Value.Entry{
        .{ .key = "enabled", .value = Value.fromBool(false) },
    };

    const out_true = try env.renderTemplate(allocator, "if", &true_ctx);
    defer allocator.free(out_true);
    const out_false = try env.renderTemplate(allocator, "if", &false_ctx);
    defer allocator.free(out_false);

    try std.testing.expectEqualStrings("on", out_true);
    try std.testing.expectEqualStrings("off", out_false);
}

test "for loop over list" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var env = Environment.init(allocator);
    defer env.deinit();

    try env.addTemplate("loop", "{% for n in nums %}[{{ n }}]{% endfor %}");

    const nums = [_]Value{
        Value.fromInt(1),
        Value.fromInt(2),
        Value.fromInt(3),
    };
    const ctx = [_]Value.Entry{
        .{ .key = "nums", .value = Value.fromList(&nums) },
    };

    const output = try env.renderTemplate(allocator, "loop", &ctx);
    defer allocator.free(output);

    try std.testing.expectEqualStrings("[1][2][3]", output);
}

test "attribute lookup via dot" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var env = Environment.init(allocator);
    defer env.deinit();

    try env.addTemplate("dot", "{{ user.name }}");

    const user_map = [_]Value.Entry{
        .{ .key = "name", .value = Value.fromString("Ada") },
    };
    const ctx = [_]Value.Entry{
        .{ .key = "user", .value = Value.fromMap(&user_map) },
    };

    const output = try env.renderTemplate(allocator, "dot", &ctx);
    defer allocator.free(output);

    try std.testing.expectEqualStrings("Ada", output);
}

test "expression in if condition" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var env = Environment.init(allocator);
    defer env.deinit();

    try env.addTemplate("cond", "{% if score >= 10 and score < 20 %}ok{% else %}bad{% endif %}");

    const ctx = [_]Value.Entry{
        .{ .key = "score", .value = Value.fromInt(12) },
    };

    const output = try env.renderTemplate(allocator, "cond", &ctx);
    defer allocator.free(output);

    try std.testing.expectEqualStrings("ok", output);
}

test "renderTemplateStruct zero-alloc nested context" {
    var arena = std.heap.ArenaAllocator.init(testAllocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var env = Environment.init(allocator);
    defer env.deinit();

    try env.addTemplate(
        "struct_ctx",
        "{{ user.name }}|{{ user.stats.score }}|{% for t in tags %}{{ t }}{% endfor %}|{{ active }}|{{ note is none }}",
    );

    const Stats = struct { score: i32 };
    const User = struct {
        name: []const u8,
        stats: Stats,
    };
    const Ctx = struct {
        user: User,
        tags: [2][]const u8,
        active: bool,
        note: ?[]const u8,
    };

    const ctx = Ctx{
        .user = .{ .name = "Ada", .stats = .{ .score = 7 } },
        .tags = .{ "x", "y" },
        .active = true,
        .note = null,
    };

    const output = try env.renderTemplateStruct(allocator, "struct_ctx", ctx);
    defer allocator.free(output);

    try std.testing.expectEqualStrings("Ada|7|xy|true|true", output);
}
