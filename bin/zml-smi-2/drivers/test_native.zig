const builtin = @import("builtin");

test {
    _ = @import("shared/test_counter.zig");
    _ = @import("shared/test_dynlib.zig");
    _ = @import("shared/test_fdinfo.zig");
    _ = @import("shared/test_hwmon.zig");
    _ = @import("shared/test_pci.zig");
    _ = @import("shared/test_proc.zig");
    _ = @import("shared/test_sandbox.zig");
    _ = @import("shared/test_sysfs.zig");
    _ = @import("shared/test_units.zig");
    if (builtin.os.tag == .macos) _ = @import("host/test_macos.zig");
}
