const std = @import("std");

pub fn lookupStruct(dynlib: *std.DynLib, comptime VTable: type) !VTable {
    var result: VTable = undefined;
    inline for (std.meta.fields(VTable)) |field| {
        @field(result, field.name) = dynlib.lookup(field.type, field.name) orelse return error.SymbolResolutionFailed;
    }

    return result;
}
