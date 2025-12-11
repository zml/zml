const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/vfs");

pub const File = @import("file.zig").File;
pub const HTTP = @import("http.zig").HTTP;
