const std = @import("std");

const c = @import("c");

fn pluginInitialize(_: [*c]c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error {
    @panic("cudaz: PJRT_Plugin_Initialize is not implemented");
}

fn makeApi() c.PJRT_Api {
    var result = std.mem.zeroes(c.PJRT_Api);
    result.struct_size = c.PJRT_Api_STRUCT_SIZE;
    result.pjrt_api_version = .{
        .struct_size = c.PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = null,
        .major_version = c.PJRT_API_MAJOR,
        .minor_version = c.PJRT_API_MINOR,
    };
    // Keep the first PJRT call explicit while the remaining API is scaffolded.
    result.PJRT_Plugin_Initialize = &pluginInitialize;
    return result;
}

const api = makeApi();

pub export fn GetPjrtApi() *const c.PJRT_Api {
    return &api;
}
