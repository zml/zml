const std = @import("std");
const c_interface = @import("c");

// Callback for completion
fn completion(buf: [*c]const u8, lc: [*c]c_interface.linenoiseCompletions) callconv(.c) void {
    if (buf[0] == 'h') {
        _ = c_interface.linenoiseAddCompletion(lc, "hello");
        _ = c_interface.linenoiseAddCompletion(lc, "Hello World");
    }
}

// Callback for hints
fn hints(buf: [*c]const u8, color: [*c]c_int, bold: [*c]c_int) callconv(.c) [*c]u8 {
    const str = std.mem.span(buf);
    if (std.ascii.eqlIgnoreCase(str, "hello")) {
        color.* = 35;
        bold.* = 0;
        return @constCast(" World");
    }
    return null;
}

pub fn interactive(allocator: std.mem.Allocator) !void {
    _ = allocator; // autofix

    // Set callbacks
    c_interface.linenoiseSetCompletionCallback(completion);
    c_interface.linenoiseSetHintsCallback(hints);

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
