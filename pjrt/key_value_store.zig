const std = @import("std");

const c = @import("c");

pub const Error = error{
    DeadlineExceeded,
    Internal,
    NotFound,
    OutOfMemory,
    Unavailable,
};

/// Thread-safe key-value store used while creating distributed PJRT clients.
///
/// The store and `user_arg` must outlive the PJRT client. Values returned by
/// `get` and `try_get` must use the provided allocator. PJRT takes ownership
/// of successful results.
pub const KeyValueStore = struct {
    user_arg: *anyopaque,
    get: *const fn (
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
        timeout_ms: c_int,
    ) Error![]u8,
    try_get: *const fn (
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
    ) Error!?[]u8,
    put: *const fn (
        user_arg: *anyopaque,
        key: []const u8,
        value: []const u8,
    ) Error!void,
};

fn callbackError(
    callback_error: [*c]c.PJRT_CallbackError,
    err: Error,
) ?*c.PJRT_Error {
    const message = @errorName(err);
    return callback_error.*.?(
        switch (err) {
            error.DeadlineExceeded => c.PJRT_Error_Code_DEADLINE_EXCEEDED,
            error.NotFound => c.PJRT_Error_Code_NOT_FOUND,
            error.OutOfMemory => c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
            error.Unavailable => c.PJRT_Error_Code_UNAVAILABLE,
            error.Internal => c.PJRT_Error_Code_INTERNAL,
        },
        message.ptr,
        message.len,
    );
}

fn free(value: [*c]u8) callconv(.c) void {
    std.c.free(@ptrCast(value));
}

pub fn getCallback(
    args: [*c]c.PJRT_KeyValueGetCallback_Args,
) callconv(.c) ?*c.PJRT_Error {
    const store: *const KeyValueStore = @ptrCast(
        @alignCast(args.*.user_arg.?),
    );
    const key = args.*.key[0..args.*.key_size];
    const value = store.get(
        store.user_arg,
        std.heap.c_allocator,
        key,
        args.*.timeout_in_ms,
    ) catch |err| return callbackError(args.*.callback_error, err);

    args.*.value = @ptrCast(value.ptr);
    args.*.value_size = value.len;
    args.*.value_deleter_callback = free;
    return null;
}

pub fn tryGetCallback(
    args: [*c]c.PJRT_KeyValueTryGetCallback_Args,
) callconv(.c) ?*c.PJRT_Error {
    const store: *const KeyValueStore = @ptrCast(
        @alignCast(args.*.user_arg.?),
    );
    const key = args.*.key[0..args.*.key_size];
    const value = store.try_get(
        store.user_arg,
        std.heap.c_allocator,
        key,
    ) catch |err| return callbackError(args.*.callback_error, err);

    const result = value orelse return callbackError(
        args.*.callback_error,
        error.NotFound,
    );
    args.*.value = @ptrCast(result.ptr);
    args.*.value_size = result.len;
    args.*.value_deleter_callback = free;
    return null;
}

pub fn putCallback(
    args: [*c]c.PJRT_KeyValuePutCallback_Args,
) callconv(.c) ?*c.PJRT_Error {
    const store: *const KeyValueStore = @ptrCast(
        @alignCast(args.*.user_arg.?),
    );
    const key = args.*.key[0..args.*.key_size];
    const value = args.*.value[0..args.*.value_size];
    store.put(store.user_arg, key, value) catch |err| {
        return callbackError(args.*.callback_error, err);
    };
    return null;
}
