const std = @import("std");

pub fn lookupStruct(dynlib: *std.DynLib, comptime VTable: type) !VTable {
    var result: VTable = undefined;
    inline for (comptime std.meta.fieldNames(VTable), comptime std.meta.fieldTypes(VTable)) |field_name, Field| {
        @field(result, field_name) = dynlib.lookup(Field, field_name) orelse return error.SymbolResolutionFailed;
    }

    return result;
}
