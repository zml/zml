const std = @import("std");
const template_mod = @import("template.zig");
const value_mod = @import("value.zig");
const Value = value_mod.Value;

pub const Environment = struct {
    allocator: std.mem.Allocator,
    templates: std.StringHashMap(template_mod.Template),

    pub fn init(allocator: std.mem.Allocator) Environment {
        return .{
            .allocator = allocator,
            .templates = std.StringHashMap(template_mod.Template).init(allocator),
        };
    }

    pub fn initMaxPerf() Environment {
        return init(std.heap.smp_allocator);
    }

    pub fn deinit(self: *Environment) void {
        self.templates.deinit();
    }

    pub fn addTemplate(self: *Environment, name: []const u8, source: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const src_copy = try self.allocator.dupe(u8, source);
        const tmpl = try template_mod.Template.parse(self.allocator, name_copy, src_copy);
        try self.templates.put(name_copy, tmpl);
    }

    pub fn getTemplate(self: *Environment, name: []const u8) ?*const template_mod.Template {
        return self.templates.getPtr(name);
    }

    pub fn renderTemplate(self: *Environment, render_allocator: std.mem.Allocator, name: []const u8, ctx: []const Value.Entry) ![]u8 {
        const tmpl = self.getTemplate(name) orelse return error.TemplateNotFound;
        return tmpl.render(render_allocator, ctx);
    }

    pub fn renderTemplateStruct(self: *Environment, render_allocator: std.mem.Allocator, name: []const u8, ctx: anytype) ![]u8 {
        const CtxT = @TypeOf(ctx);
        const T = switch (@typeInfo(CtxT)) {
            .pointer => |p| p.child,
            else => CtxT,
        };
        var adapted: value_mod.StructContext(T) = .{};
        const entries = switch (@typeInfo(CtxT)) {
            .pointer => adapted.from(ctx),
            else => adapted.from(&ctx),
        };
        return self.renderTemplate(render_allocator, name, entries);
    }

    pub fn renderNamedStr(self: *Environment, render_allocator: std.mem.Allocator, name: []const u8, source: []const u8, ctx: []const Value.Entry) ![]u8 {
        const tmpl = try template_mod.Template.parse(self.allocator, name, source);
        return tmpl.render(render_allocator, ctx);
    }
};
