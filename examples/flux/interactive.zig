const std = @import("std");
const c_interface = @import("c");

// Callback for completion
fn completion(buf: [*c]const u8, lc: *c_interface.Completions) callconv(.c) void {
    if (buf[0] == 'h') {
        _ = c_interface.linenoiseAddCompletion(lc, "hello");
        _ = c_interface.linenoiseAddCompletion(lc, "hello there");
    }
}

// Callback for hints
fn hints(buf: [*c]const u8, color: *c_int, bold: *c_int) callconv(.c) [*c]u8 {
    const str = std.mem.span(buf);
    if (std.ascii.eqlIgnoreCase(str, "hello")) {
        color.* = 35;
        bold.* = 0;
        // Hints must be static strings or allocated, but here we return a string literal.
        // linenoise does not free hints.
        return @constCast(" World");
    }
    return null;
}

pub fn interactive(allocator: std.mem.Allocator) !void {
    _ = allocator; // autofix
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();

    // var args_iter = try @TypeOf(init.args).Iterator.initAllocator(init.args, allocator);
    // defer args_iter.deinit();

    // const prgname = args_iter.next() orelse "linenoise";

    // Parse options
    // while (args_iter.next()) |arg| {
    //     if (std.mem.eql(u8, arg, "--multiline")) {
    //         c_interface.linenoiseSetMultiLine(1);
    //         std.debug.print("Multi-line mode enabled.\n", .{});
    //     } else if (std.mem.eql(u8, arg, "--keycodes")) {
    //         c_interface.linenoisePrintKeyCodes();
    //         std.process.exit(0);
    //     } else if (std.mem.eql(u8, arg, "--async")) {
    //         async_mode = true;
    //     } else {
    //         std.debug.print("Usage: {s} [--multiline] [--keycodes] [--async]\n", .{prgname});
    //         std.process.exit(1);
    //     }
    // }

    // Set callbacks
    // c_interface.linenoiseSetCompletionCallback(completion);
    // c_interface.linenoiseSetHintsCallback(hints);

    // Load history
    _ = c_interface.linenoiseHistoryLoad("history.txt");

    while (true) {
        const line = c_interface.linenoise("hello> ");
        if (line == null) break;
        defer c_interface.linenoiseFree(line);

        if (line[0] != 0 and line[0] != '/') {
            std.debug.print("echo: '{s}'\n", .{std.mem.span(line)});
            _ = c_interface.linenoiseHistoryAdd(line);
            _ = c_interface.linenoiseHistorySave("history.txt");
        } else if (std.mem.startsWith(u8, std.mem.span(line), "/historylen")) {
            const len_str = std.mem.span(line)[11..];
            if (std.fmt.parseInt(c_int, std.mem.trim(u8, len_str, " "), 10)) |len| {
                _ = c_interface.linenoiseHistorySetMaxLen(len);
            } else |_| {}
        } else if (std.mem.startsWith(u8, std.mem.span(line), "/mask")) {
            c_interface.linenoiseMaskModeEnable();
        } else if (std.mem.startsWith(u8, std.mem.span(line), "/unmask")) {
            c_interface.linenoiseMaskModeDisable();
        } else if (line[0] == '/') {
            std.debug.print("Unrecognized command: {s}\n", .{std.mem.span(line)});
        }
    }
}
