const std = @import("std");
const builtin = @import("builtin");
const elf = std.elf;
const c = @import("c");

pub fn Fn(comptime name: [:0]const u8) type {
    return *const @TypeOf(@field(c, name));
}

pub fn open(comptime F: type, path: [:0]const u8) ?F {
    const handle = openDl(path) orelse return null;
    return resolveAll(F, handle, null, 0);
}

pub fn openElf(comptime F: type, io: std.Io, path: [:0]const u8, comptime known_sym: [:0]const u8) !F {
    const handle = openDl(path) orelse return error.DlOpenFailed;

    const sym_addr = std.c.dlsym(handle, known_sym) orelse
        return error.KnownSymbolNotFound;

    var info: DlInfo = undefined;
    if (dladdr(sym_addr, &info) == 0) return error.DladdrFailed;

    const elf_base = @intFromPtr(info.dli_fbase);
    const elf_path = std.mem.sliceTo(info.dli_fname, 0);
    const file = try std.Io.Dir.openFileAbsolute(io, elf_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const mmap = try std.posix.mmap(
        null,
        @intCast(stat.size),
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap);

    return resolveAll(F, handle, mmap, elf_base) orelse error.SymbolResolutionFailed;
}

fn resolveAll(comptime F: type, handle: *anyopaque, elf_mmap: ?[]align(std.heap.page_size_min) u8, elf_base: usize) ?F {
    var result: F = undefined;
    var unresolved: std.hash_map.StringHashMapUnmanaged(*usize) = .{};
    defer unresolved.deinit(std.heap.page_allocator);

    // Phase 1: try dlsym for each field; for unresolved ones, store a pointer to the result field
    inline for (std.meta.fields(F)) |field| {
        if (std.c.dlsym(handle, field.name)) |p| {
            @field(result, field.name) = @as(field.type, @ptrFromInt(@intFromPtr(p)));
        } else {
            unresolved.put(std.heap.page_allocator, field.name, @ptrCast(&@field(result, field.name))) catch return null;
        }
    }

    if (unresolved.count() == 0) { // no need to parse ELF
        return result;
    }

    // Phase 2: scan ELF symbols, write resolved addresses directly into result fields
    const buf = elf_mmap orelse return null;
    const ehdr: *const elf.ElfN.Ehdr = @ptrCast(@alignCast(buf.ptr));
    const shdrs: [*]const elf.ElfN.Shdr = @ptrCast(@alignCast(buf.ptr + @as(usize, ehdr.shoff)));

    var remaining = unresolved.count();

    outer: for (0..ehdr.shnum) |i| {
        const sh = shdrs[i];
        if (sh.type != .SYMTAB) {
            continue;
        }

        const strtab_sh = shdrs[sh.link];
        const strtab: [*]const u8 = @ptrCast(buf.ptr + @as(usize, strtab_sh.offset));
        const syms: []const elf.ElfN.Sym = @as(
            [*]const elf.ElfN.Sym,
            @ptrCast(@alignCast(buf.ptr + @as(usize, sh.offset))),
        )[0 .. @as(usize, sh.size) / @sizeOf(elf.ElfN.Sym)];

        for (syms) |s| {
            if (s.value == 0) {
                continue;
            }

            const sym_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(strtab + @as(usize, s.name))), 0);
            if (unresolved.get(sym_name)) |field_ptr| {
                field_ptr.* = elf_base + @as(usize, s.value);
                remaining -= 1;

                if (remaining == 0) {
                    break :outer;
                }
            }
        }
    }

    if (remaining > 0) {
        return null;
    }

    return result;
}

fn openDl(path: [:0]const u8) ?*anyopaque {
    return switch (builtin.os.tag) {
        .linux, .macos => std.c.dlopen(path, switch (builtin.os.tag) {
            .linux => .{ .LAZY = true, .GLOBAL = true, .NODELETE = true },
            .macos => .{ .LAZY = true, .LOCAL = true },
            else => unreachable,
        }),
        else => blk: {
            const dl = std.DynLib.open(path) catch break :blk null;
            break :blk dl.inner.handle;
        },
    };
}

const DlInfo = extern struct {
    dli_fname: [*:0]const u8,
    dli_fbase: *anyopaque,
    dli_sname: [*c]const u8,
    dli_saddr: ?*anyopaque,
};

extern fn dladdr(addr: *anyopaque, info: *DlInfo) c_int;
