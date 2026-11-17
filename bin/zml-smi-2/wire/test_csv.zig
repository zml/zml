const std = @import("std");
const contract = @import("zml-smi-2/contract");
const csv = @import("csv.zig");

test "CSV writes a rectangular table, preserving inventory order and quoting details" {
    const large_detail: [600]u8 = @splat('x');
    const metric_delimiters: [37]u8 = @splat(',');
    const devices = [_]contract.Device{
        .{
            .target = .metal,
            .remote = true,
            .name = "M,\"Pro\"\r\nX",
            .driver_version = "\"Driver\"",
            .details = &.{
                .{ .key = "chip,rev", .value = "A\"\n" },
                .{ .key = "large", .value = &large_detail },
            },
        },
        .{ .target = .tpu, .name = "TPU" },
    };
    var samples = [_]contract.Metrics{ .{}, .{} };
    samples[0].put(.ane_power_mw, 7);
    samples[1].put(.util_percent, 0);
    samples[1].put(.mem_total_bytes, 4096);

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try csv.write(&output.writer, .{ .devices = &devices, .samples = &samples });
    try std.testing.expectEqualStrings(
        "index,target,remote,name,driver_version,details,util_percent,encoder_util_percent," ++
            "decoder_util_percent,power_mw,power_limit_mw,temperature,fan_speed_percent," ++
            "clock_graphics_mhz,clock_sm_mhz,clock_soc_mhz,clock_mem_mhz,clock_graphics_max_mhz," ++
            "clock_mem_max_mhz,mem_used_bytes,mem_total_bytes,mem_bus_width,pcie_tx_kbps," ++
            "pcie_rx_kbps,pcie_bandwidth_mbps,pcie_link_gen,pcie_link_width,nc_tensors," ++
            "nc_constants,nc_model_code,nc_shared_scratchpad,nc_nonshared_scratchpad,nc_runtime," ++
            "nc_driver,nc_dma_rings,nc_collectives,nc_notifications,nc_uncategorized," ++
            "ambient_temperature,governor,liveness,throttle_reasons,ane_power_mw\n" ++
            "0,metal,true,\"M,\"\"Pro\"\"\r\nX\",\"\"\"Driver\"\"\",\"" ++
            "[{\"\"key\"\":\"\"chip,rev\"\",\"\"value\"\":\"\"A\\\"\"\\n\"\"}," ++
            "{\"\"key\"\":\"\"large\"\",\"\"value\"\":\"\"" ++ large_detail ++ "\"\"}]\"" ++
            metric_delimiters ++ "7\n" ++
            "1,tpu,false,TPU,,\"[]\",0" ++
            metric_delimiters[0..14] ++ "4096" ++ metric_delimiters[0..22] ++ "\n",
        output.written(),
    );
}
