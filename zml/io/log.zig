//! The loader's log scopes. A failure is logged once, where it is
//! detected: at `err` when it fails initialization or the load, at `debug`
//! when the loader stays usable. Frames that only propagate an error do
//! not log it again.
const std = @import("std");
const builtin = @import("builtin");

pub const load = Scoped(.@"zml/io/load");
pub const io = Scoped(.@"zml/io");
pub const mem = Scoped(.@"zml/mem");

fn Scoped(comptime scope: @EnumLiteral()) type {
    const base = std.log.scoped(scope);
    return struct {
        pub const debug = base.debug;
        pub const info = base.info;
        pub const warn = base.warn;

        /// A terminal failure. The test runner counts logged errors as
        /// test failures, so test builds log it at `debug` instead.
        pub fn err(comptime fmt: []const u8, args: anytype) void {
            if (builtin.is_test) base.debug(fmt, args) else base.err(fmt, args);
        }

        /// The first failure of a pipeline, with the error's name appended.
        /// Cancellation and shutdown are expected and log at `debug`.
        pub fn failure(e: anyerror, comptime fmt: []const u8, args: anytype) void {
            const named_fmt = fmt ++ ": {s}";
            const named_args = args ++ .{@errorName(e)};
            if (e == error.Canceled or e == error.Cancelled or e == error.Closed) debug(named_fmt, named_args) else err(named_fmt, named_args);
        }
    };
}
