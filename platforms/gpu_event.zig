const std = @import("std");

const pjrt = @import("pjrt");

pub const Backend = enum {
    cuda,
    rocm,
};

pub const Operation = enum {
    create,
    marker_init,
    record,
    synchronize,
    elapsed,
    destroy,
};

pub const Failure = struct {
    operation: Operation,
    status: c_int,
};

pub fn Result(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: Failure,
    };
}

pub const Event = *anyopaque;

pub const Provider = struct {
    pub const VTable = struct {
        create: *const fn (context: *const anyopaque, stream: *pjrt.Stream) Result(Event),
        marker_init_async: *const fn (context: *const anyopaque, marker: *anyopaque, stream: *pjrt.Stream) Result(void),
        record: *const fn (context: *const anyopaque, event: Event, stream: *pjrt.Stream) Result(void),
        synchronize: *const fn (context: *const anyopaque, event: Event) Result(void),
        elapsed_ms: *const fn (context: *const anyopaque, start: Event, stop: Event) Result(f32),
        destroy: *const fn (context: *const anyopaque, event: Event) Result(void),
        error_string: *const fn (context: *const anyopaque, status: c_int) ?[*:0]const u8,
    };

    backend: Backend,
    library_handle: *anyopaque,
    context: *const anyopaque,
    vtable: *const VTable,

    pub fn create(self: *const Provider, stream: *pjrt.Stream) Result(Event) {
        return self.vtable.create(self.context, stream);
    }

    /// Initializes the one-byte device marker without synchronizing the stream.
    pub fn markerInitAsync(self: *const Provider, marker: *anyopaque, stream: *pjrt.Stream) Result(void) {
        return self.vtable.marker_init_async(self.context, marker, stream);
    }

    pub fn record(self: *const Provider, event: Event, stream: *pjrt.Stream) Result(void) {
        return self.vtable.record(self.context, event, stream);
    }

    pub fn sync(self: *const Provider, event: Event) Result(void) {
        return self.vtable.synchronize(self.context, event);
    }

    pub fn elapsedMs(self: *const Provider, start: Event, stop: Event) Result(f32) {
        return self.vtable.elapsed_ms(self.context, start, stop);
    }

    /// Event destruction is fallible so callers can report it, but callers should
    /// still invoke it best-effort for every event they successfully created.
    pub fn destroy(self: *const Provider, event: Event) Result(void) {
        return self.vtable.destroy(self.context, event);
    }

    pub fn errorString(self: *const Provider, failure: Failure) []const u8 {
        const message = self.vtable.error_string(self.context, failure.status) orelse return "unknown GPU runtime error";
        return std.mem.span(message);
    }
};
