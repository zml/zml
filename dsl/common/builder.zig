const std = @import("std");

pub const control_flow = @import("control_flow.zig");
pub const args = @import("args.zig");

pub const arity = control_flow.arity;
pub const openFor = control_flow.openFor;
pub const openIf = control_flow.openIf;
pub const openIfElse = control_flow.openIfElse;
pub const openWhile = control_flow.openWhile;
pub const innerValues = control_flow.innerValues;
pub const resultTypes = control_flow.resultTypes;
pub const ForScope = control_flow.ForScope;
pub const IfOnlyScope = control_flow.IfOnlyScope;
pub const IfScope = control_flow.IfScope;
pub const WhileScope = control_flow.WhileScope;

pub const NamedArgs = args.NamedArgs;

test {
    std.testing.refAllDecls(@This());
}
