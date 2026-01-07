const std = @import("std");

const c = @import("c");

const pjrt = @import("pjrt.zig");

const log = std.log.scoped(.pjrt);

pub const LayoutExtension = struct {
    inner: c.PJRT_Layouts_Extension,

    pub const extension_id = c.PJRT_Extension_Type_Layouts;
    const Funcs = std.meta.FieldEnum(c.PJRT_Layouts_Extension);

    fn CallFnArgType(comptime func: Funcs) type {
        const fti = @typeInfo(@FieldType(c.PJRT_Layouts_Extension, @tagName(func)));
        const fn_ptr = @typeInfo(fti.optional.child);
        const fn_type_info = @typeInfo(fn_ptr.pointer.child);
        const arg_array_type_info = @typeInfo(fn_type_info.@"fn".params[0].type.?);
        return arg_array_type_info.pointer.child;
    }

    inline fn call(self: *const LayoutExtension, api: *const pjrt.Api, comptime method: Funcs, arg: anytype) pjrt.ApiError!CallFnArgType(method) {
        var ret = pjrt.pjrtStruct2(CallFnArgType(method), arg);
        const fn_ptr = @field(&self.inner, @tagName(method)).?;
        const result = fn_ptr(&ret);
        if (@TypeOf(result) == void) {
            return ret;
        }

        if (result) |pjrt_c_error| {
            const pjrt_error: *pjrt.Error = @ptrCast(pjrt_c_error);
            log.err("[{s}] {s}", .{ @tagName(method), pjrt_error.getMessage(api) });
            return pjrt_error.getCode(api).toApiError();
        }

        return ret;
    }

    pub const ClientGetDefaultLayoutArgs = struct {
        client: *pjrt.Client,
        type: pjrt.BufferType,
        dims: []const i64,
    };

    pub fn clientGetDefaultLayout(extension: *const LayoutExtension, api: *const pjrt.Api, args: ClientGetDefaultLayoutArgs) pjrt.ApiError!*MemoryLayout {
        const ret = try extension.call(api, .PJRT_Layouts_PJRT_Client_GetDefaultLayout, .{
            .client = args.client.inner(),
            .type = @intFromEnum(args.type),
            .dims = args.dims.ptr,
            .num_dims = args.dims.len,
        });
        return @ptrCast(ret.layout.?);
    }

    pub const TopologyGetDefaultLayoutArgs = struct {
        topology_description: *pjrt.TopologyDescription,
        type: pjrt.BufferType,
        dims: []const i64,
    };

    pub fn topologyGetDefaultLayout(extension: *const LayoutExtension, api: *const pjrt.Api, args: TopologyGetDefaultLayoutArgs) pjrt.ApiError!*MemoryLayout {
        const ret = try extension.call(api, .PJRT_Layouts_PJRT_Topology_GetDefaultLayout, .{
            .topology_description = args.topology_description.inner(),
            .type = @intFromEnum(args.type),
            .dims = args.dims.ptr,
            .num_dims = args.dims.len,
        });
        return @ptrCast(ret.layout.?);
    }

    pub fn executableGetOutputLayouts(extension: *const LayoutExtension, api: *const pjrt.Api, executable: *pjrt.Executable) pjrt.ApiError![]const *MemoryLayout {
        const ret = try extension.call(api, .PJRT_Layouts_PJRT_Executable_GetOutputLayouts, .{
            .executable = executable.inner(),
        });
        return @as([*]const *MemoryLayout, @ptrCast(ret.layouts))[0..ret.num_outputs];
    }
};

pub const MemoryLayout = opaque {
    pub const inner = pjrt.InnerMixin(c.PJRT_Layouts_MemoryLayout).inner;

    pub fn deinit(self: *MemoryLayout, extension: *const LayoutExtension, api: *const pjrt.Api) void {
        _ = extension.call(api, .PJRT_Layouts_MemoryLayout_Destroy, .{
            .layout = self.inner(),
        }) catch {};
    }

    pub const SerializeAllocError = pjrt.ApiError || std.mem.Allocator.Error;

    pub fn serializeAlloc(self: *MemoryLayout, allocator: std.mem.Allocator, extension: *const LayoutExtension, api: *const pjrt.Api) SerializeAllocError![]const u8 {
        const ret = try extension.call(api, .PJRT_Layouts_MemoryLayout_Serialize, .{ .layout = self.inner() });
        defer ret.serialized_layout_deleter.?(ret.serialized_layout);

        const out = try allocator.alloc(u8, ret.serialized_bytes_size);
        errdefer allocator.free(out);

        @memcpy(out, ret.serialized_bytes[0..ret.serialized_bytes_size]);

        return out;
    }
};
