#!/usr/bin/env python3
"""Generate ROCDL bindings from llvm-tblgen --dump-json output."""
import argparse
import json
import re
from pathlib import Path


def generate(records, output):
    def get(ref):
        return records[ref['def']]

    def constraint(ref):
        record = get(ref)
        while 'OpVariable' in record['!superclasses']:
            record = get(record['constraint'])
        return record

    def belongs(record):
        return (record.get('opDialect') or record.get('dialect') or {}).get('def') == 'ROCDL_Dialect'

    operations = sorted((records[n] for n in records['!instanceof']['Op'] if belongs(records[n])), key=lambda o: o['opName'])
    attributes = sorted((records[n] for n in records['!instanceof']['AttrDef'] if belongs(records[n])), key=lambda a: a['cppClassName'])
    assert not any(belongs(records[n]) for n in records['!instanceof'].get('TypeDef', [])), 'ROCDL gained custom types; extend the bindings'
    assert [name for _, name in records['ROCDL_TargetAttr']['parameters']['args']] == ['O', 'triple', 'chip', 'features', 'abi', 'flags', 'link'], 'ROCDL target fields changed'
    assert [name for _, name in records['ROCDL_Dialect']['discardableAttrs']['args']] == ['kernel', 'reqd_work_group_size', 'flat_work_group_size', 'max_flat_work_group_size', 'waves_per_eu', 'unsafe_fp_atomics', 'xnack', 'sramecc', 'last_use', 'no_remote_memory', 'no_fine_grained_memory', 'ignore_denormal_mode'], 'ROCDL discardable attributes changed; update annotations.zig'
    keywords = set('addrspace align allowzero and anyframe anytype asm async await break callconv catch comptime const continue defer else enum errdefer error export extern fn for if inline linksection noalias noinline nosuspend opaque or orelse packed pub resume return section struct suspend switch test threadlocal try union unreachable usingnamespace var volatile while'.split())

    def ident(name):
        return '@"' + name + '"' if name in keywords or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) else name

    def parameter(name):
        return ident(name + '_' if name in {'ctx', 'opts', 'location', 'attributes', 'result_type', 'operands', 'results'} else name)

    header = '// Generated from LLVM ROCDL TableGen records; see REGENERATING.md.\n// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n\n'
    ops = [header, 'const mlir = @import("mlir");\nconst stdx = @import("stdx");\n\npub const names = [_][]const u8{\n']
    ops += [f'    "rocdl.{o["opName"]}",\n' for o in operations]
    ops.append('};\n\n')
    for op in operations:
        for section in ['regions', 'successors']:
            assert not op[section]['args'], (op['opName'], section)
        args = [(constraint(ref), name) for ref, name in op['arguments']['args']]
        results = [(constraint(ref), name) for ref, name in op['results']['args']]
        for rec, _ in args + results:
            assert not any(s in rec['!superclasses'] for s in ['Variadic', 'Optional', 'VariadicOfVariadic', 'Property']), (op['opName'], rec['!name'])
        operands = [(rec, n) for rec, n in args if 'Type' in rec['!superclasses']]
        attrs = [(rec, n) for rec, n in args if 'Attr' in rec['!superclasses']]
        assert len(operands) + len(attrs) == len(args), op['opName']
        summary = ' '.join((op.get('summary') or '').split())
        ops.append(f'/// `rocdl.{op["opName"]}`.' + (' ' + summary if summary else '') + '\n')
        ops.append(f'pub fn {ident(op["opName"].replace(".", "_"))}(\n    ctx: *mlir.Context,\n')
        for _, n in operands:
            ops.append(f'    {parameter(n)}: *const mlir.Value,\n')
        for _, n in results:
            ops.append(f'    {parameter(n + "_type")}: *const mlir.Type,\n')
        optional_attrs = [
            n for rec, n in attrs
            if rec.get('isOptional') or 'OptionalAttr' in rec['!superclasses']
            or 'DefaultValuedAttr' in rec['!superclasses'] or rec['!name'] == 'UnitAttr'
        ]
        for _, n in attrs:
            if n not in optional_attrs:
                ops.append(f'    {parameter(n)}: *const mlir.Attribute,\n')
        if optional_attrs:
            ops.append('    opts: struct {\n')
            for n in optional_attrs:
                ops.append(f'        {parameter(n)}: ?*const mlir.Attribute = null,\n')
            ops.append('    },\n')
        ops.append('    location: *const mlir.Location,\n) *mlir.Operation {\n')
        if attrs:
            ops.append(f'    var attributes: stdx.BoundedArray(mlir.NamedAttribute, {len(attrs)}) = .empty;\n')
            for rec, n in attrs:
                optional = rec.get('isOptional') or 'OptionalAttr' in rec['!superclasses'] or 'DefaultValuedAttr' in rec['!superclasses'] or rec['!name'] == 'UnitAttr'
                if optional:
                    ops.append(f'    if (opts.{parameter(n)}) |attribute| {{\n        attributes.appendAssumeCapacity(.named(ctx, "{n}", attribute));\n    }}\n')
                else:
                    ops.append(f'    attributes.appendAssumeCapacity(.named(ctx, "{n}", {parameter(n)}));\n')
        ops.append(f'    return mlir.Operation.make(ctx, "rocdl.{op["opName"]}", .{{\n')
        if operands:
            ops.append('        .operands = .{ .flat = &.{ ' + ', '.join(parameter(n) for _, n in operands) + ' } },\n')
        if results:
            ops.append('        .results = .{ .flat = &.{ ' + ', '.join(parameter(n + '_type') for _, n in results) + ' } },\n')
        if attrs:
            ops.append('        .attributes = attributes.constSlice(),\n')
        ops.append('        .location = location,\n    });\n}\n\n')
    (output / 'ops.zig').write_text(''.join(ops))

    types = [header, 'const mlir = @import("mlir");\nconst llvm = @import("mlir/dialects/llvm");\n\n']
    types.append('''/// ROCDL address spaces use LLVM pointer types.
pub const AddressSpace = enum(u32) {
    generic = 0,
    global = 1,
    shared = 3,
    constant = 4,
    private = 5,
    barrier = 15,
};

pub fn pointerType(ctx: *mlir.Context, space: AddressSpace) *const mlir.Type {
    return llvm.pointerType(ctx, space);
}

''')
    for record in sorted((r for r in records.values() if isinstance(r, dict)), key=lambda r: r.get('!name', '')):
        if not isinstance(record, dict) or record.get('!anonymous', True) or 'ROCDL_NamedType' not in record.get('!superclasses', []):
            continue
        match = re.fullmatch(r'vector<(\d+)x(i\d+|f\d+|bf16)>', record['typeName'])
        assert match, record['!name']
        length, element = match.groups()
        factory = 'int' if element.startswith('i') else 'float'
        types.append(f'/// `{record["typeName"]}`, as used by ROCDL operations.\npub fn v{length}{element}(ctx: *mlir.Context) *const mlir.Type {{\n    return .vector(&.{{{length}}}, .{factory}(ctx, .{element}));\n}}\n\n')
    (output / 'types.zig').write_text(''.join(types))

    enum_records = {}
    attribute_enums = []
    for attr in attributes:
        if attr['!name'] == 'ROCDL_TargetAttr':
            continue
        params = attr['parameters']['args']
        assert len(params) == 1 and params[0][1] == 'value', attr['!name']
        enum = get(get(params[0][0])['enum'])
        enum_records[enum['className']] = enum
        attribute_enums.append((attr, enum))
    declared_enums = {records[n]['className'] for n in records['!instanceof']['EnumInfo'] if records[n].get('cppNamespace') == '::mlir::ROCDL'}
    assert set(enum_records) == declared_enums, 'ROCDL gained an enum without an attribute; extend the bindings'
    enums = [header]
    for name, enum in sorted(enum_records.items()):
        bits = enum['bitwidth']
        enums.append('/// ' + ' '.join(enum['summary'].split()) + f'\npub const {name} = enum(u{bits}) {{\n')
        for ref in enum['enumerants']:
            case = get(ref)
            enums.append(f'    {ident(case["symbol"])} = {case["value"]},\n')
        if 'BitEnum' in enum['!superclasses']:
            enums.append(f'''    _,

    pub fn combine(values: []const {name}) {name} {{
        var bits: u{bits} = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }}

    pub fn contains(self: {name}, value: {name}) bool {{
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }}
''')
        enums.append('};\n\n')
    (output / 'enums.zig').write_text(''.join(enums))

    zig = [header, 'const std = @import("std");\nconst c = @import("c");\nconst mlir = @import("mlir");\nconst enums = @import("enums.zig");\n\npub const TargetAttr = @import("target.zig").TargetAttr;\n\n']
    h = ['#ifndef ZML_MLIR_DIALECTS_ROCDL_CAPI_H\n#define ZML_MLIR_DIALECTS_ROCDL_CAPI_H\n\n#include "mlir-c/Dialect/ROCDL.h"\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\n// Registration\n\nMLIR_CAPI_EXPORTED intptr_t mlirROCDLGetOperationCount(MlirContext context);\n\n// Attributes\n\n']
    cc = ['#include "capi.h"\n\n#include "mlir/CAPI/IR.h"\n#include "mlir/CAPI/Support.h"\n#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"\n#include "mlir/IR/Diagnostics.h"\n\n// Registration\n\nintptr_t mlirROCDLGetOperationCount(MlirContext context) {\n  auto *ctx = unwrap(context);\n  if (!ctx) return 0;\n  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();\n  intptr_t count = 0;\n  for (auto op : ctx->getRegisteredOperations())\n    if (op.getDialectNamespace() == "rocdl") ++count;\n  return count;\n}\n\n// Attributes\n\n']
    for attr, enum in attribute_enums:
        name = attr['cppClassName']
        enum_name = enum['className']
        h.append(f'''MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDL{name}(MlirAttribute attribute);
MLIR_CAPI_EXPORTED MlirAttribute mlirROCDL{name}Get(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirROCDL{name}GetValue(MlirAttribute attribute);

''')
        cc.append(f'''bool mlirAttributeIsAROCDL{name}(MlirAttribute attribute) {{
  return llvm::isa_and_nonnull<mlir::ROCDL::{name}>(unwrap(attribute));
}}

MlirAttribute mlirROCDL{name}Get(MlirContext context, uint32_t value) {{
  auto *ctx = unwrap(context);
  if (!ctx) return {{nullptr}};
''')
        if enum['bitwidth'] < 32:
            cc.append(f'  if (value > {(1 << enum["bitwidth"]) - 1}) return {{nullptr}};\n')
        cc.append(f'''  auto symbol = mlir::ROCDL::{enum['underlyingToSymbolFnName']}(static_cast<{enum['underlyingType']}>(value));
  if (!symbol) return {{nullptr}};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::{name}::get(ctx, *symbol));
}}

uint32_t mlirROCDL{name}GetValue(MlirAttribute attribute) {{
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::{name}>(unwrap(attribute)).getValue());
}}

''')
        zig.append(f'''/// `#rocdl.{attr['mnemonic']}`.
pub const {name} = opaque {{
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.{enum_name};
    pub const isAFn = c.mlirAttributeIsAROCDL{name};
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {{
        return self.attribute().format(writer);
    }}
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value_: Value) mlir.Error!*const Self {{
        return @ptrCast(c.mlirROCDL{name}Get(ctx.ptr(), @intFromEnum(value_)).ptr orelse return error.InvalidMlir);
    }}

    pub fn value(self: *const Self) Value {{
        return @enumFromInt(c.mlirROCDL{name}GetValue(self.ptr()));
    }}

    pub fn attribute(self: *const Self) *const mlir.Attribute {{
        return @ptrCast(self);
    }}
}};

''')
    zig.append('pub const enum_attributes = .{\n' + ''.join(f'    {a["cppClassName"]},\n' for a, _ in attribute_enums) + '};\n')
    (output / 'attributes.zig').write_text(''.join(zig))
    target_h = (output / 'capi.h').read_text().split('// Target\n', 1)[1] if (output / 'capi.h').exists() and '// Target\n' in (output / 'capi.h').read_text() else ''
    target_cc = (output / 'capi.cc').read_text().split('// Target\n', 1)[1] if (output / 'capi.cc').exists() and '// Target\n' in (output / 'capi.cc').read_text() else ''
    assert target_h and target_cc, 'Maintain the Target section of capi.h/cc before generation'
    (output / 'capi.h').write_text(''.join(h) + '// Target\n' + target_h)
    (output / 'capi.cc').write_text(''.join(cc) + '// Target\n' + target_cc)
    print(f'Generated {len(operations)} operations, {len(attributes)} attributes, {len(enum_records)} enums.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path, help='ROCDLOps.td llvm-tblgen --dump-json output')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    generate(json.loads(args.records.read_text()), args.output)
