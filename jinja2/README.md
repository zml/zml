# Jinja2 in Zig

```zig
const std = @import("std");
const mj = @import("src/lib.zig");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    var env = mj.Environment.initMaxPerf();
    defer env.deinit();

    try env.addTemplate("greet", "Hello {{ user.name }}! score={{ user.score }}");

    const Ctx = struct {
        user: struct {
            name: []const u8,
            score: i32,
        },
    };

    const ctx = Ctx{
        .user = .{
            .name = "Ada",
            .score = 42,
        },
    };

    // Zero-alloc struct context adapter (nested structs supported)
    const out = try env.renderTemplateStruct(allocator, "greet", ctx);
    defer allocator.free(out);

    std.debug.print("{s}\n", .{out});
}
```
