const builtin = @import("builtin");

test {
    _ = @import("test_collector.zig");
    _ = @import("test_counter.zig");
    _ = @import("test_double_buffer.zig");
    _ = @import("test_dynlib.zig");
    _ = @import("test_fdinfo.zig");
    _ = @import("test_hwmon.zig");
    _ = @import("test_json.zig");
    _ = @import("test_model.zig");
    _ = @import("test_pci.zig");
    _ = @import("test_proc.zig");
    _ = @import("test_prometheus.zig");
    _ = @import("test_remote.zig");
    _ = @import("test_sandbox.zig");
    _ = @import("test_sysfs.zig");
    _ = @import("test_units.zig");
    if (builtin.os.tag == .macos) _ = @import("host/test_macos.zig");
}
