const std = @import("std");

const python_launcher = @import("platforms/neuron/python_launcher");

// Thin executable wrapper for ZML's embedded NKI compiler script. The outer
// Neuron hook spawns this as `bin/nki-cc` for each inline kernel custom-call.
pub fn main(init: std.process.Init) !void {
    _ = std;
    try python_launcher.runScriptMain(init, &.{
        "..",
        "site-packages",
        "nki_kernel_compiler.py",
    });
}
