pub const data = @import("data.zig");
pub const print = @import("print.zig");
pub const top = @import("top.zig");

test {
    _ = data;
    _ = @import("lib/image.zig");
    _ = @import("widgets/logo.zig");
    _ = @import("widgets/process_table.zig");
}
