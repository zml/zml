const std = @import("std");

pub const BufferedAnyWriter = std.io.BufferedWriter(4096, std.io.AnyWriter);
pub const BufferedAnyReader = std.io.BufferedReader(4096, std.io.AnyReader);
