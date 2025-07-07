const common = @import("runtimes/common");
const c = @import("c");

pub var cuda: common.weakifySymbols(c, "cu") = undefined;

pub fn load() !void {
    try common.bindWeakSymbols(&cuda, c, "cu");
}
