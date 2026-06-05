const std = @import("std");

pub const Stats = struct {
    auth_requests: u64 = 0,
    auth_waits: u64 = 0,
    reconstruction_requests: u64 = 0,
    reconstruction_prefetches: u64 = 0,
    reconstruction_prefetch_errors: u64 = 0,
    reconstruction_prefetch_unavailable: u64 = 0,
    reconstruction_cache_hits: u64 = 0,
    reconstruction_waits: u64 = 0,
    tensor_scope_hits: u64 = 0,
    tensor_scope_misses: u64 = 0,
    reads: u64 = 0,
    fetches_max: u64 = 0,
    terms_max: u64 = 0,
    output_bytes_max: u64 = 0,
    xorb_requests: u64 = 0,
    network_waits: u64 = 0,
    xorb_in_flight_current: u64 = 0,
    xorb_in_flight_max: u64 = 0,
    network_active_max: u64 = 0,
    output_direct_frames: u64 = 0,
    output_direct_bytes: u64 = 0,
    output_copy_frames: u64 = 0,
    output_copy_bytes: u64 = 0,
    xorb_range_cache_hits: u64 = 0,
    xorb_range_cache_misses: u64 = 0,
    requested_bytes: u64 = 0,
    reconstruction_bytes: u64 = 0,
    xorb_bytes_requested: u64 = 0,
    xorb_bytes_read: u64 = 0,
    xorb_bytes_discarded: u64 = 0,
    decoded_bytes: u64 = 0,
    xorb_chunks: u64 = 0,
    xorb_chunks_discarded: u64 = 0,
    xorb_chunks_none: u64 = 0,
    xorb_chunks_lz4: u64 = 0,
    xorb_chunks_bg4: u64 = 0,
    xorb_payload_bytes_none: u64 = 0,
    xorb_payload_bytes_lz4: u64 = 0,
    xorb_payload_bytes_bg4: u64 = 0,
    decoded_bytes_none: u64 = 0,
    decoded_bytes_lz4: u64 = 0,
    decoded_bytes_bg4: u64 = 0,
    reconstruction_ns: u64 = 0,
    reconstruction_send_ns: u64 = 0,
    reconstruction_head_ns: u64 = 0,
    reconstruction_body_ns: u64 = 0,
    reconstruction_parse_ns: u64 = 0,
    auth_wait_ns: u64 = 0,
    reconstruction_wait_ns: u64 = 0,
    xorb_send_ns: u64 = 0,
    xorb_head_ns: u64 = 0,
    xorb_head_ns_max: u64 = 0,
    network_wait_ns: u64 = 0,
    xorb_http_ns: u64 = 0,
    xorb_fetch_ns_max: u64 = 0,
    xorb_request_bytes_max: u64 = 0,
    xorb_decode_ns: u64 = 0,
    xorb_decode_ns_none: u64 = 0,
    xorb_decode_ns_lz4: u64 = 0,
    xorb_decode_ns_bg4: u64 = 0,
    xorb_bg4_lz4_ns: u64 = 0,
    xorb_bg4_degroup_ns: u64 = 0,

    pub fn totalRequests(self: Stats) u64 {
        return self.auth_requests + self.reconstruction_requests + self.xorb_requests;
    }
};

pub const AtomicStats = struct {
    auth_requests: std.atomic.Value(u64) = .init(0),
    auth_waits: std.atomic.Value(u64) = .init(0),
    reconstruction_requests: std.atomic.Value(u64) = .init(0),
    reconstruction_prefetches: std.atomic.Value(u64) = .init(0),
    reconstruction_prefetch_errors: std.atomic.Value(u64) = .init(0),
    reconstruction_prefetch_unavailable: std.atomic.Value(u64) = .init(0),
    reconstruction_cache_hits: std.atomic.Value(u64) = .init(0),
    reconstruction_waits: std.atomic.Value(u64) = .init(0),
    tensor_scope_hits: std.atomic.Value(u64) = .init(0),
    tensor_scope_misses: std.atomic.Value(u64) = .init(0),
    reads: std.atomic.Value(u64) = .init(0),
    fetches_max: std.atomic.Value(u64) = .init(0),
    terms_max: std.atomic.Value(u64) = .init(0),
    output_bytes_max: std.atomic.Value(u64) = .init(0),
    xorb_requests: std.atomic.Value(u64) = .init(0),
    network_waits: std.atomic.Value(u64) = .init(0),
    xorb_in_flight_current: std.atomic.Value(u64) = .init(0),
    xorb_in_flight_max: std.atomic.Value(u64) = .init(0),
    network_active_max: std.atomic.Value(u64) = .init(0),
    output_direct_frames: std.atomic.Value(u64) = .init(0),
    output_direct_bytes: std.atomic.Value(u64) = .init(0),
    output_copy_frames: std.atomic.Value(u64) = .init(0),
    output_copy_bytes: std.atomic.Value(u64) = .init(0),
    xorb_range_cache_hits: std.atomic.Value(u64) = .init(0),
    xorb_range_cache_misses: std.atomic.Value(u64) = .init(0),
    requested_bytes: std.atomic.Value(u64) = .init(0),
    reconstruction_bytes: std.atomic.Value(u64) = .init(0),
    xorb_bytes_requested: std.atomic.Value(u64) = .init(0),
    xorb_bytes_read: std.atomic.Value(u64) = .init(0),
    xorb_bytes_discarded: std.atomic.Value(u64) = .init(0),
    decoded_bytes: std.atomic.Value(u64) = .init(0),
    xorb_chunks: std.atomic.Value(u64) = .init(0),
    xorb_chunks_discarded: std.atomic.Value(u64) = .init(0),
    xorb_chunks_none: std.atomic.Value(u64) = .init(0),
    xorb_chunks_lz4: std.atomic.Value(u64) = .init(0),
    xorb_chunks_bg4: std.atomic.Value(u64) = .init(0),
    xorb_payload_bytes_none: std.atomic.Value(u64) = .init(0),
    xorb_payload_bytes_lz4: std.atomic.Value(u64) = .init(0),
    xorb_payload_bytes_bg4: std.atomic.Value(u64) = .init(0),
    decoded_bytes_none: std.atomic.Value(u64) = .init(0),
    decoded_bytes_lz4: std.atomic.Value(u64) = .init(0),
    decoded_bytes_bg4: std.atomic.Value(u64) = .init(0),
    reconstruction_ns: std.atomic.Value(u64) = .init(0),
    reconstruction_send_ns: std.atomic.Value(u64) = .init(0),
    reconstruction_head_ns: std.atomic.Value(u64) = .init(0),
    reconstruction_body_ns: std.atomic.Value(u64) = .init(0),
    reconstruction_parse_ns: std.atomic.Value(u64) = .init(0),
    auth_wait_ns: std.atomic.Value(u64) = .init(0),
    reconstruction_wait_ns: std.atomic.Value(u64) = .init(0),
    xorb_send_ns: std.atomic.Value(u64) = .init(0),
    xorb_head_ns: std.atomic.Value(u64) = .init(0),
    xorb_head_ns_max: std.atomic.Value(u64) = .init(0),
    network_wait_ns: std.atomic.Value(u64) = .init(0),
    xorb_http_ns: std.atomic.Value(u64) = .init(0),
    xorb_fetch_ns_max: std.atomic.Value(u64) = .init(0),
    xorb_request_bytes_max: std.atomic.Value(u64) = .init(0),
    xorb_decode_ns: std.atomic.Value(u64) = .init(0),
    xorb_decode_ns_none: std.atomic.Value(u64) = .init(0),
    xorb_decode_ns_lz4: std.atomic.Value(u64) = .init(0),
    xorb_decode_ns_bg4: std.atomic.Value(u64) = .init(0),
    xorb_bg4_lz4_ns: std.atomic.Value(u64) = .init(0),
    xorb_bg4_degroup_ns: std.atomic.Value(u64) = .init(0),

    pub fn reset(self: *AtomicStats) void {
        inline for (@typeInfo(Stats).@"struct".fields) |field| {
            @field(self, field.name).store(0, .monotonic);
        }
    }

    pub fn snapshot(self: *const AtomicStats) Stats {
        var result: Stats = .{};
        inline for (@typeInfo(Stats).@"struct".fields) |field| {
            @field(result, field.name) = @field(self, field.name).load(.monotonic);
        }
        return result;
    }

    pub fn add(self: *AtomicStats, comptime name: []const u8, value: u64) void {
        if (value == 0) return;
        _ = @field(self, name).fetchAdd(value, .monotonic);
    }

    pub fn max(self: *AtomicStats, comptime name: []const u8, value: u64) void {
        const field = &@field(self, name);
        var current = field.load(.monotonic);
        while (value > current) {
            if (field.cmpxchgWeak(current, value, .monotonic, .monotonic)) |actual| {
                current = actual;
            } else return;
        }
    }
};
