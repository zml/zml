const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const ui = @import("../lib/ui.zig");
const Gauge = @import("gauge.zig");

test "bars and history preserve text alignment and missing values" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    vxfw.DrawContext.init(.unicode);
    const context: vxfw.DrawContext = .{
        .arena = allocator,
        .min = .{ .width = 30 },
        .max = .{ .width = 30, .height = null },
        .cell_size = .{},
    };
    const cases = [_]struct { value: ?u8, history: bool, expected: []const u8 }{
        .{ .value = 50, .history = false, .expected = "GPU [██████░░░░░░░] 50%   tail" },
        .{ .value = null, .history = false, .expected = "GPU [░░░░░░░░░░░░░] N/A   tail" },
        .{ .value = 50, .history = true, .expected = "GPU ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀  50%   tail" },
        .{ .value = null, .history = true, .expected = "GPU ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀  N/A   tail" },
    };
    for (cases) |case| {
        const gauge: Gauge = .{
            .value = case.value,
            .label = "GPU",
            .suffix = "tail",
            .label_reserve = 4,
            .suffix_reserve = 7,
            .history = if (case.history) &.{ 100, 100, null, null } else null,
        };
        const surface = try gauge.draw(context);
        var screen = try vaxis.Screen.init(allocator, .{
            .cols = 30,
            .rows = 3,
            .x_pixel = 0,
            .y_pixel = 0,
        });
        const window: vaxis.Window = .{
            .x_off = 0,
            .y_off = 0,
            .parent_x_off = 0,
            .parent_y_off = 0,
            .width = 30,
            .height = 3,
            .screen = &screen,
        };
        surface.render(window, ui.widget(&gauge));
        var rendered: std.Io.Writer.Allocating = .init(allocator);
        for (0..30) |column| {
            const cell = screen.readCell(@intCast(column), if (case.history) 1 else 0).?;
            try rendered.writer.writeAll(cell.char.grapheme);
        }
        try std.testing.expectEqualStrings(case.expected, rendered.written());
    }
}
