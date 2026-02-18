const std = @import("std");
const c_interface = @import("c");
const FluxPipeline = @import("pipeline.zig").FluxPipeline;
const utils = @import("utils.zig");

// Callback for completion
fn completion(buf: [*c]const u8, lc: [*c]c_interface.linenoiseCompletions) callconv(.c) void {
    const str = std.mem.span(buf);
    if (str.len > 0 and str[0] == '/') {
        if (std.mem.startsWith(u8, "/ge", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/generator_type");
        }
        if (std.mem.startsWith(u8, "/ra", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/random_seed");
        }
        if (std.mem.startsWith(u8, "/nu", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/num_inference_steps");
        }
        if (std.mem.startsWith(u8, "/hi", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/historylen");
        }
        if (std.mem.startsWith(u8, "/ma", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/mask");
        }
        if (std.mem.startsWith(u8, "/un", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/unmask");
        }
        if (std.mem.startsWith(u8, "/he", str)) {
            _ = c_interface.linenoiseAddCompletion(lc, "/help");
        }
    } else {
        _ = c_interface.linenoiseAddCompletion(lc, "/help");
    }
}

// Callback for hints
fn hints(buf: [*c]const u8, color: [*c]c_int, bold: [*c]c_int) callconv(.c) [*c]u8 {
    const str = std.mem.span(buf);
    color.* = 35;
    bold.* = 0;

    if (str.len == 0) {
        return @constCast("/help");
    }
    if (std.ascii.eqlIgnoreCase(str, "hello")) {
        return @constCast(" World");
    }

    if (std.mem.startsWith(u8, str, "/")) {
        if (std.mem.eql(u8, str, "/generator_type")) return @constCast(" <type>");
        if (std.mem.eql(u8, str, "/random_seed")) return @constCast(" <int>");
        if (std.mem.eql(u8, str, "/num_inference_steps")) return @constCast(" <int>");
        if (std.mem.eql(u8, str, "/historylen")) return @constCast(" <int>");
    }

    return null;
}

pub fn interactive(allocator: std.mem.Allocator, pipeline: *FluxPipeline) !void {
    _ = allocator; // autofix
    // Set callbacks
    c_interface.linenoiseSetCompletionCallback(completion);
    c_interface.linenoiseSetHintsCallback(hints);

    // Load history
    _ = c_interface.linenoiseHistoryLoad("history.txt");

    while (true) {
        const line = c_interface.linenoise("flux> ");
        if (line == null) break;
        defer c_interface.linenoiseFree(line);
        const line_slice = std.mem.span(line);

        if (line_slice.len > 0) {
            _ = c_interface.linenoiseHistoryAdd(line);
            _ = c_interface.linenoiseHistorySave("history.txt");
        }

        if (line_slice.len > 0 and line_slice[0] != '/') {
            std.debug.print("Generating for: '{s}'\n", .{line_slice});
            var options = pipeline.config;
            options.prompt = line_slice;
            pipeline.generate(options) catch |err| {
                std.log.err("Generation failed: {}", .{err});
            };
        } else if (std.mem.startsWith(u8, line_slice, "/generator_type")) {
            const type_str = std.mem.trim(u8, line_slice[15..], " ");
            if (std.meta.stringToEnum(utils.GeneratorType, type_str)) |gt| {
                pipeline.config.generator_type = gt;
                std.debug.print("Generator Type set to: {s}\n", .{type_str});
            } else {
                std.debug.print("Invalid Generator Type: {s}\n", .{type_str});
                std.debug.print("Available types: torch, accelerator_box_muller, accelerator_marsaglia\n", .{});
            }
        } else if (std.mem.startsWith(u8, line_slice, "/random_seed")) {
            const seed_str = std.mem.trim(u8, line_slice[12..], " ");
            if (std.fmt.parseInt(u64, seed_str, 10)) |seed| {
                pipeline.config.random_seed = seed;
                std.debug.print("Random Seed set to: {}\n", .{seed});
            } else |_| {
                std.debug.print("Invalid Random Seed: {s}\n", .{seed_str});
            }
        } else if (std.mem.startsWith(u8, line_slice, "/num_inference_steps")) {
            const steps_str = std.mem.trim(u8, line_slice[20..], " ");
            if (std.fmt.parseInt(usize, steps_str, 10)) |steps| {
                pipeline.config.num_inference_steps = steps;
                std.debug.print("Num Inference Steps set to: {}\n", .{steps});
            } else |_| {
                std.debug.print("Invalid Num Inference Steps: {s}\n", .{steps_str});
            }
        } else if (std.mem.startsWith(u8, line_slice, "/historylen")) {
            const len_str = line_slice[11..];
            if (std.fmt.parseInt(c_int, std.mem.trim(u8, len_str, " "), 10)) |len| {
                _ = c_interface.linenoiseHistorySetMaxLen(len);
            } else |_| {}
        } else if (std.mem.startsWith(u8, line_slice, "/mask")) {
            c_interface.linenoiseMaskModeEnable();
        } else if (std.mem.startsWith(u8, line_slice, "/unmask")) {
            c_interface.linenoiseMaskModeDisable();
        } else if (std.mem.startsWith(u8, line_slice, "/help")) {
            std.debug.print("Available commands:\n", .{});
            std.debug.print("  /generator_type <type>    Set generator type (torch, accelerator_box_muller, accelerator_marsaglia)\n", .{});
            std.debug.print("  /random_seed <int>        Set random seed\n", .{});
            std.debug.print("  /num_inference_steps <int> Set number of inference steps\n", .{});
            std.debug.print("  /historylen <int>         Set history length\n", .{});
            std.debug.print("  /mask                     Enable input masking\n", .{});
            std.debug.print("  /unmask                   Disable input masking\n", .{});
            std.debug.print("  /help                     Show this help message\n", .{});
        } else if (line_slice.len > 0 and line_slice[0] == '/') {
            std.debug.print("Unrecognized command: {s}\n", .{line_slice});
        }
    }
}
