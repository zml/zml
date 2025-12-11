const builtin = @import("builtin");
const std = @import("std");

pub const VFS = @import("io").VFS;

const log = std.log.scoped(.@"zml/io");
