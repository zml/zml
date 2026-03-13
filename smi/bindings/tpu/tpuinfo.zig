const std = @import("std");
const c = @import("c");

const has_tpu = @hasDecl(c, "ZML_RUNTIME_TPU");

pub const max_devices: usize = c.TPUINFO_MAX_DEVICES;

pub const Error = error{
    TpuUnavailable,
    QueryFailed,
};

pub fn queryInt(
    address: [:0]const u8,
    metric_name: [:0]const u8,
    device_ids: []c_longlong,
    values: []c_longlong,
) Error!u32 {
    if (comptime !has_tpu) return error.TpuUnavailable;

    const n = c.tpu_query_int(address.ptr, metric_name.ptr, device_ids.ptr, values.ptr, @intCast(device_ids.len));
    if (n < 0) return error.QueryFailed;

    return @intCast(n);
}

pub fn queryDouble(
    address: [:0]const u8,
    metric_name: [:0]const u8,
    device_ids: []c_longlong,
    values: []f64,
) Error!u32 {
    if (comptime !has_tpu) return error.TpuUnavailable;

    const n = c.tpu_query_double(address.ptr, metric_name.ptr, device_ids.ptr, values.ptr, @intCast(device_ids.len));
    if (n < 0) return error.QueryFailed;

    return @intCast(n);
}
