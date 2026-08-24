const std = @import("std");

const python_launcher = @import("platforms/neuron/python_launcher");

pub fn main(init: std.process.Init) !void {
    try python_launcher.runModuleEntrypoint(init, .{
        .module_name = "neuronxcc.driver.CommandDriver",
    });
}
