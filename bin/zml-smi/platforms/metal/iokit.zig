//! The IOKit functions Metal calls. The hermetic toolchain has no IOKit to link
//! against and no headers for it, so they are declared here and resolved by install name.
const std = @import("std");
const smi = @import("zml-smi");

pub const Object = std.c.mach_port_t;

/// kIOMainPortDefault.
pub const main_port: std.c.mach_port_t = 0;

pub const Api = struct {
    /// Returns a matching dictionary, which the lookup consumes.
    IOServiceMatching: *const fn (name: [*:0]const u8) callconv(.c) ?*anyopaque,
    IOServiceNameMatching: *const fn (name: [*:0]const u8) callconv(.c) ?*anyopaque,
    IOServiceGetMatchingService: *const fn (port: std.c.mach_port_t, matching: ?*anyopaque) callconv(.c) Object,
    IOServiceOpen: *const fn (service: Object, task: std.c.mach_port_t, kind: u32, connection: *Object) callconv(.c) std.c.kern_return_t,
    IOServiceClose: *const fn (connection: Object) callconv(.c) std.c.kern_return_t,
    IOConnectCallStructMethod: *const fn (connection: Object, selector: u32, input: *const anyopaque, input_size: usize, output: *anyopaque, output_size: *usize) callconv(.c) std.c.kern_return_t,
    IORegistryEntryGetChildIterator: *const fn (entry: Object, plane: [*:0]const u8, iterator: *Object) callconv(.c) std.c.kern_return_t,
    IORegistryEntryCreateCFProperty: *const fn (entry: Object, key: *const anyopaque, allocator: ?*const anyopaque, options: u32) callconv(.c) ?*const anyopaque,
    IOIteratorNext: *const fn (iterator: Object) callconv(.c) Object,
    IOObjectConformsTo: *const fn (object: Object, class: [*:0]const u8) callconv(.c) c_int,
    IOObjectRelease: *const fn (object: Object) callconv(.c) std.c.kern_return_t,
};

pub fn open() !Api {
    return smi.dynlib.open(Api, &.{"/System/Library/Frameworks/IOKit.framework/IOKit"});
}

pub fn check(status: std.c.kern_return_t) error{IoKitError}!void {
    if (status != 0) return error.IoKitError;
}
