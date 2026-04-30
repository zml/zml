const std = @import("std");

const python_launcher = @import("platforms/neuron/python_launcher");

// Thin executable wrapper for the Neuron SDK's neuronx-cc Python entrypoint.
// The sandbox packages this binary as `bin/neuronx-cc` so Zig callers can spawn
// it without depending on a host Python installation.
pub fn main(init: std.process.Init) !void {
    try python_launcher.runModuleEntrypoint(init, .{
        .module_name = "neuronxcc.driver.CommandDriver",
    });
}
