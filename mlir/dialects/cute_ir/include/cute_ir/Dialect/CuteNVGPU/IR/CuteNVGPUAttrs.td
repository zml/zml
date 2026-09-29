//===- CuteNVGPUAttrs.td - CuTe NVIDIA GPU attributes ------*- tablegen -*-===//
//
// Generated for nvidia-cutlass-dsl 4.8.0 from the DSL compiler
// (`_cutlass_ir.cu13` parser messages and probing its attribute parser and
// printer) and its enum bindings; do not edit by hand.
//
// The enum attributes are CuteNVGPUEnums.td; `#cute_nvgpu.major<k>`, over the
// cute dialect's MajorMode, is here so that the enum generator run over
// CuteNVGPUEnums.td emits the nvgpu enums only. The fragment attributes are
// `key = value` lists in a fixed order; a defaulted field is printed only
// when set.
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_ATTRS_TD
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_ATTRS_TD

include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.td"
include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUEnums.td"
include "cute_ir/Dialect/Cute/IR/CuteEnums.td"

def CuteNVGPU_MajorModeAttr
    : EnumAttr<CuteNVGPU_Dialect, Cute_MajorModeEnum, "major"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

class CuteNVGPU_Attr<string name, string attrMnemonic>
    : AttrDef<CuteNVGPU_Dialect, name> {
  let mnemonic = attrMnemonic;
}

// `true` / `false`.
class CuteNVGPU_BoolParam : AttrOrTypeParameter<"bool", ""> {
  let parser = "::mlir::cutlass_compiler::cute_nvgpu::parseBool($_parser)";
  let printer = "$_printer << ($_self ? \"true\" : \"false\")";
}
def CuteNVGPU_Bool : CuteNVGPU_BoolParam;
def CuteNVGPU_FalseByDefault : CuteNVGPU_BoolParam {
  let defaultValue = "false";
}

def CuteNVGPU_RmemFrgAttr : CuteNVGPU_Attr<"RmemFrg", "rmem_frg"> {
  let summary = "Register-memory fragment of an MMA operand";
  let description = [{
    `#cute_nvgpu.rmem_frg<value_type = f32, operand = A, derive_elem_type = true>`
  }];
  let parameters = (ins
    "::mlir::Type":$value_type,
    "::mlir::cutlass_compiler::cute::MmaOperand":$operand,
    CuteNVGPU_FalseByDefault:$derive_elem_type
  );
  let assemblyFormat = [{
    `<` `value_type` `=` $value_type `,` `operand` `=` $operand
    (`,` `derive_elem_type` `=` $derive_elem_type^)? `>`
  }];
}

class CuteNVGPU_SmemFrg<string name, string attrMnemonic, string arch>
    : CuteNVGPU_Attr<name, attrMnemonic> {
  let summary = arch # " shared-memory fragment";
  let description = "`#cute_nvgpu." # attrMnemonic # "<major = k>`";
  let parameters = (ins "::mlir::cutlass_compiler::cute::MajorMode":$major_mode);
  let assemblyFormat = "`<` `major` `=` $major_mode `>`";
}

def CuteNVGPU_SM90SmemFrgAttr
    : CuteNVGPU_SmemFrg<"SM90SmemFrg", "arch.sm90.smem_frg", "SM90">;
def CuteNVGPU_SM100SmemFrgAttr
    : CuteNVGPU_SmemFrg<"SM100SmemFrg", "arch.sm100.smem_frg", "SM100">;
def CuteNVGPU_SM100CircularSmemFrgAttr
    : CuteNVGPU_SmemFrg<"SM100CircularSmemFrg", "arch.sm100.circular_smem_frg",
                        "SM100 circular">;
def CuteNVGPU_SM107SmemFrgAttr
    : CuteNVGPU_SmemFrg<"SM107SmemFrg", "arch.sm107.smem_frg", "SM107">;

def CuteNVGPU_SM100TmemFrgAttr
    : CuteNVGPU_Attr<"SM100TmemFrg", "arch.sm100.tmem_frg"> {
  let summary = "SM100 tensor-memory fragment";
  let description = [{
    `#cute_nvgpu.arch.sm100.tmem_frg<data_type = f32, storage_type = f32, cta_group = 1, tmem_alloc_mode = Interleaved>`
  }];
  let parameters = (ins
    "::mlir::Type":$data_type,
    "::mlir::Type":$storage_type,
    "int":$cta_group,
    "TmemAllocMode":$tmem_alloc_mode
  );
  let assemblyFormat = [{
    `<` `data_type` `=` $data_type `,` `storage_type` `=` $storage_type `,`
    `cta_group` `=` $cta_group `,` `tmem_alloc_mode` `=` $tmem_alloc_mode `>`
  }];
}

def CuteNVGPU_SM100TmemSfFrgAttr
    : CuteNVGPU_Attr<"SM100TmemSfFrg", "arch.sm100.tmem_sf_frg"> {
  let summary = "SM100 tensor-memory scale-factor fragment";
  let description = [{
    `#cute_nvgpu.arch.sm100.tmem_sf_frg<sf_type = f8E8M0FNU, sf_vec_size = 32, cta_group = 1, is_sfa = true, tmem_alloc_mode = Interleaved>`
  }];
  let parameters = (ins
    "::mlir::Type":$sf_type,
    "int":$sf_vec_size,
    "int":$cta_group,
    CuteNVGPU_Bool:$is_sfa,
    "TmemAllocMode":$tmem_alloc_mode
  );
  let assemblyFormat = [{
    `<` `sf_type` `=` $sf_type `,` `sf_vec_size` `=` $sf_vec_size `,`
    `cta_group` `=` $cta_group `,` `is_sfa` `=` $is_sfa `,`
    `tmem_alloc_mode` `=` $tmem_alloc_mode `>`
  }];
}

def CuteNVGPU_SM100TmemEFrgAttr
    : CuteNVGPU_Attr<"SM100TmemEFrg", "arch.sm100.tmem_e_frg"> {
  let summary = "SM100 tensor-memory sparse-metadata fragment";
  let description = [{
    `#cute_nvgpu.arch.sm100.tmem_e_frg<a_type = f16, e_type = i8>`
  }];
  let parameters = (ins "::mlir::Type":$a_type, "::mlir::Type":$e_type);
  let assemblyFormat = "`<` `a_type` `=` $a_type `,` `e_type` `=` $e_type `>`";
}

def CuteNVGPU_NotImplementedFrgAttr
    : CuteNVGPU_Attr<"NotImplementedFrg", "not_implemented_frg"> {
  let summary = "Fragment of an atom that has none";
  let description = "`#cute_nvgpu.not_implemented_frg<>`";
  let assemblyFormat = "`<` `>`";
}

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_ATTRS_TD
