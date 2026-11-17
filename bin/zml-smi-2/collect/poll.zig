const std = @import("std");
const contract = @import("zml-smi-2/contract");
const source = @import("zml-smi-2/source");

const Metric = contract.Metric;
const Metrics = contract.Metrics;

/// A backend supplies a probe for each device. The hooks run before any probes
/// and after all device measurements, respectively.
pub fn PerDevice(comptime Backend: type) type {
    return struct {
        pub fn poll(self: *Backend, scratch: std.mem.Allocator, reading: *source.Reading) !void {
            if (@hasDecl(Backend, "begin")) self.begin();
            for (reading.samples, 0..) |*sample, index| {
                const probe: Backend.Probe = self.probe(index);
                project(sample, &probe);
            }
            if (@hasDecl(Backend, "readProcesses")) try self.readProcesses(scratch, reading);
        }
    };
}

/// Specialize a sequence of direct calls from a probe's public methods.
/// A scalar uses the method name. A grouped reply supplies its own field names.
pub fn project(sample: *Metrics, probe: anytype) void {
    const Probe = @TypeOf(probe.*);
    comptime var seen: std.EnumSet(Metric) = .empty;
    inline for (comptime std.meta.declarations(Probe)) |name| {
        const method = @field(Probe, name);
        const info = @typeInfo(@TypeOf(method));
        if (info != .@"fn") continue;
        const signature = info.@"fn";
        if (signature.is_generic or signature.attrs.varargs or signature.param_types.len != 1 or
            signature.param_types[0] != *const Probe)
            @compileError("probe method must take only *const Probe: " ++ name);
        const Result = signature.return_type.?;
        const reply = method(probe);
        if (Result == ?u64) {
            const key = comptime claim(&seen, name);
            sample.put(key, reply);
        } else {
            const result = @typeInfo(Result);
            if (result != .@"struct" or result.@"struct".is_tuple)
                @compileError("query must return ?u64 or a named record of ?u64 metrics: " ++ name);
            inline for (result.@"struct".field_names, result.@"struct".field_types) |field, Field| {
                if (Field != ?u64) @compileError("query metric must be ?u64: " ++ field);
                const key = comptime claim(&seen, field);
                sample.put(key, @field(reply, field));
            }
        }
    }
}

fn claim(seen: *std.EnumSet(Metric), comptime name: []const u8) Metric {
    if (!@hasField(Metric, name)) @compileError("unknown metric: " ++ name);
    const field = @field(Metric, name);
    if (seen.contains(field)) @compileError("more than one query writes " ++ name);
    seen.insert(field);
    return field;
}
