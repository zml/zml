//! The IOKit functions Metal calls. The hermetic toolchain has no IOKit to link
//! against and no headers for it, so they are declared here and resolved by install name.
const std = @import("std");
const native = @import("zml-smi/native");

/// An io_object_t: a service, an iterator or a connection. 0 is none.
pub const Object = std.c.mach_port_t;
const Status = std.c.kern_return_t;

/// kIOMainPortDefault.
pub const main_port: std.c.mach_port_t = 0;

pub const Api = struct {
    /// Returns a matching dictionary, which the lookup consumes.
    IOServiceMatching: *const fn ([*:0]const u8) callconv(.c) ?*anyopaque,
    IOServiceNameMatching: *const fn ([*:0]const u8) callconv(.c) ?*anyopaque,
    IOServiceGetMatchingService: *const fn (std.c.mach_port_t, ?*anyopaque) callconv(.c) Object,
    IOServiceOpen: Checked(fn (Object, std.c.mach_port_t, u32, *Object) callconv(.c) Status),
    IOServiceClose: *const fn (Object) callconv(.c) Status,
    IOConnectCallStructMethod: Checked(fn (
        Object,
        u32,
        *const anyopaque,
        usize,
        *anyopaque,
        *usize,
    ) callconv(.c) Status),
    IORegistryEntryGetChildIterator: Checked(fn (Object, [*:0]const u8, *Object) callconv(.c) Status),
    /// Returns a CoreFoundation object the caller releases.
    IORegistryEntryCreateCFProperty: *const fn (
        Object,
        *const anyopaque,
        ?*const anyopaque,
        u32,
    ) callconv(.c) ?*const anyopaque,
    IOIteratorNext: *const fn (Object) callconv(.c) Object,
    IOObjectConformsTo: *const fn (Object, [*:0]const u8) callconv(.c) c_int,
    IOObjectRelease: *const fn (Object) callconv(.c) Status,
};

pub fn open() !Api {
    return native.dynlib.open(Api, &.{"/System/Library/Frameworks/IOKit.framework/IOKit"});
}

const Checked = native.dynlib.CheckedBy(check);

fn check(status: Status) error{IoKitError}!void {
    if (status != 0) return error.IoKitError;
}
