// clang-format off
/***************************************************************************************************
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// clang-format on

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>

using namespace mlir;
using namespace mlir::cutlass_compiler::cute;

namespace cg = cutegen;

// The compiler's parser of the algebra types reads `<"text">` leniently: a
// missing or unreadable payload, including a bare mnemonic, gives the default
// value (`()`, `():()`, `[]`, `S<0,4,3>`) instead of an error.
template <typename T>
static T parseLenientPayload(mlir::AsmParser &parser) {
  std::string str;
  if (succeeded(parser.parseOptionalLess()) &&
      succeeded(parser.parseOptionalString(&str)) &&
      succeeded(parser.parseOptionalGreater()))
    return cutegen::from_string<T>(str).value_or(T{});
  return T{};
}

// The compiler's integer type is `!cute.i<width>`, a mnemonic per width that
// the generated dispatch cannot match. The generated CuteDialect::parseType
// and printType pass the dialect parser and printer, so these overloads of
// the renamed dispatch functions catch them first (defined below).
static OptionalParseResult generatedCuteTypeParser(DialectAsmParser &parser,
                                                   StringRef *mnemonic,
                                                   Type &value);
static LogicalResult generatedCuteTypePrinter(Type type,
                                              DialectAsmPrinter &printer);
#define generatedTypeParser generatedCuteTypeParser
#define generatedTypePrinter generatedCuteTypePrinter

// Emit the tablegen-generated type storage / parsing machinery (also drives
// CuteDialect::parseType / printType via useDefaultTypePrinterParser).
#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/Cute/IR/CuteTypes.cpp.inc"
#undef generatedTypeParser
#undef generatedTypePrinter

//===----------------------------------------------------------------------===//
// CuteDialect — type registration
//===----------------------------------------------------------------------===//

namespace {
namespace cute = ::mlir::cutlass_compiler::cute;
namespace nv = ::mlir::cutlass_compiler::cute_nvgpu;

std::string printed(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

// The compiler's name for a tiled copy's atom; empty when it has none.
std::string copyAliasName(Type atom) {
  return llvm::TypeSwitch<Type, std::string>(atom)
      .Case<nv::CopyAtomSIMTSyncCopyType>([](auto) { return "copy_simt"; })
      .Case<nv::CopyAtomSIMTAsyncCopyType>([](auto) { return "copy_ldgsts"; })
      .Case<nv::CopyAtomDsmemStoreType>([](auto) { return "copy_dsmem_store"; })
      .Case<nv::CopyAtomBulkCopyG2SType>([](auto) { return "copy_blkcp_g2s"; })
      .Case<nv::CopyAtomBulkCopyS2GType>([](auto) { return "copy_blkcp_s2g"; })
      .Case<nv::CopyAtomBulkCopyS2SType>([](auto) { return "copy_blkcp_s2s"; })
      .Case<nv::CopyAtomG2RType>([](auto) { return "copy_g2r"; })
      .Case<nv::CopyAtomR2GType>([](auto) { return "copy_r2g"; })
      .Case<nv::CopyAtomR2SType>([](auto) { return "copy_r2s"; })
      .Case<nv::CopyAtomS2RType>([](auto) { return "copy_s2r"; })
      .Case<nv::CopyAtomSM100CopyS2TType>([](auto) { return "copy_utccp"; })
      .Case<nv::CopyAtomSM100TmemLoadType>(
          [](auto t) { return "copy_ldtm_" + std::to_string(t.getNumBit()); })
      .Case<nv::CopyAtomSM100TmemStoreType>(
          [](auto t) { return "copy_sttm_" + std::to_string(t.getNumBit()); })
      .Case<nv::CopyAtomSM10xTmemLoadRedType>([](auto t) {
        return "copy_ldtm_red_" + std::to_string(t.getNumBit());
      })
      .Case<nv::CopyAtomSM107TmemLoadSPCompressType>([](auto t) {
        return "copy_ldtm_spcompress_" + std::to_string(t.getNumBit());
      })
      .Case<nv::CopyAtomLdsmType>([](auto t) {
        return "copy_ldsm_" + std::to_string(t.getNumMatrices());
      })
      .Case<nv::CopyAtomStsmType>([](auto t) {
        return "copy_stsm_" + std::to_string(t.getNumMatrices());
      })
      .Case<nv::CopyAtomTmaLoadType>([](auto t) {
        return "copy_tmaldg_" + std::to_string(t.getCopyBits() / 8) + "b";
      })
      .Case<nv::CopyAtomTmaStoreType>([](auto t) {
        return "copy_tmastg_" + std::to_string(t.getCopyBits() / 8) + "b";
      })
      .Case<nv::CopyAtomTmaReduceType>([](auto t) {
        return "copy_tmaredg_" + std::to_string(t.getCopyBits() / 8) + "b";
      })
      .Default([](Type) { return "copy_unknown_type"; });
}

// `mma_<A>_<B>_<C>_<M>x<N>x<K>`; empty for a type that is not an MMA atom.
std::string mmaAliasName(Type atom) {
  auto name = [](auto t) {
    std::string shape;
    auto mnk = cutegen::flatten(t.getShapeMnk().getRef());
    for (size_t i = 0; i < cutegen::rank(mnk); ++i)
      shape += (i ? "x" : "") + cutegen::to_string(cutegen::get(mnk, i));
    return "mma_" + printed(t.getAType()) + "_" + printed(t.getBType()) + "_" +
           printed(t.getCType()) + "_" + shape;
  };
  return llvm::TypeSwitch<Type, std::string>(atom)
      .Case<nv::UniversalFmaAtomType, nv::MmaAtomSM80Type,
            nv::MmaAtomSM80SparseType, nv::MmaAtomSM89Type, nv::MmaAtomSM90Type,
            nv::MmaAtomSM120BlockScaledType, nv::MmaAtomSM100UMMAType,
            nv::MmaAtomSM100UMMASparseType, nv::MmaAtomSM100UMMABlockScaledType,
            nv::MmaAtomSM100UMMABlockScaledSparseType, nv::MmaAtomSM107UMMAType,
            nv::MmaAtomSM107UMMASparseType, nv::MmaAtomSM107UMMABlockScaledType,
            nv::MmaAtomSM107UMMABlockScaledSparseType>(name)
      .Default([](Type) { return std::string(); });
}

// The compiler's type aliases: `!memref_<space>_<element>` for a memref,
// `!copy_<atom>` for a tiled copy and `!mma_<A>_<B>_<C>_<M>x<N>x<K>` for a
// tiled MMA (MLIR escapes and numbers them). Other types have none.
struct CuteOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Type type, raw_ostream &os) const override {
    std::string name;
    if (auto memref = llvm::dyn_cast<cute::MemRefType>(type))
      name = "memref_" +
             stringifyAddressSpace(memref.getPtr().getAddressSpace()).str() +
             "_" + printed(memref.getPtr().getValueType());
    else if (auto copy = llvm::dyn_cast<TiledCopyType>(type))
      name = copyAliasName(copy.getCopyAtom());
    else if (auto mma = llvm::dyn_cast<TiledMmaType>(type))
      name = mmaAliasName(mma.getMmaAtom());
    if (name.empty())
      return AliasResult::NoAlias;
    os << name;
    return AliasResult::OverridableAlias;
  }
};
} // namespace

void CuteDialect::registerCuteTypes(CuteDialect *dialect) {
  dialect->addTypes<
#define GET_TYPEDEF_LIST
#include "cute_ir/Dialect/Cute/IR/CuteTypes.cpp.inc"
      >();
  dialect->addInterfaces<CuteOpAsmInterface>();
}

//===----------------------------------------------------------------------===//
// IntTupleType — custom assembly format
//

// Grammar: < " cutegen-int-tuple-string " > (read leniently, see above)
//===----------------------------------------------------------------------===//

mlir::Type IntTupleType::parse(mlir::AsmParser &parser) {
  return get(parser.getContext(),
             IntTupleAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::int_tuple>(parser)));
}

void IntTupleType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool IntTupleType::isStatic() const { return cutegen::is_static(getRef()); }

mlir::Attribute IntTupleType::getValueAttr() const { return getAttr(); }

llvm::TypeSize
IntTupleType::getTypeSize(const mlir::DataLayout &dataLayout,
                          mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize
IntTupleType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                                mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t
IntTupleType::getABIAlignment(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t
IntTupleType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t>
IntTupleType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                               mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool IntTupleType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  // The lowering can be a bare integer (scalar/depth-0 dynamic leaves), which
  // does not implement DataLayoutTypeInterface; only delegate when it does and
  // otherwise fall back to the interface default ("compatible").
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
IntTupleType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                            mlir::Location loc) const {
  OpBuilder builder(getContext());
  // See areCompatible: bare-integer lowerings cannot be queried, so only
  // delegate when the lowered type implements the interface.
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoordType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type CoordType::parse(mlir::AsmParser &parser) {
  return get(parser.getContext(),
             CoordAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::coord>(parser)));
}

void CoordType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool CoordType::isStatic() const { return cutegen::is_static(getRef()); }

mlir::Attribute CoordType::getValueAttr() const { return getAttr(); }

llvm::TypeSize
CoordType::getTypeSize(const mlir::DataLayout &dataLayout,
                       mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize
CoordType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t CoordType::getABIAlignment(const mlir::DataLayout &dataLayout,
                                    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t
CoordType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                 mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t>
CoordType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool CoordType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
CoordType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                         mlir::Location loc) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShapeType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type ShapeType::parse(mlir::AsmParser &parser) {
  return get(parser.getContext(),
             ShapeAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::shape>(parser)));
}

void ShapeType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool ShapeType::isStatic() const { return cutegen::is_static(getRef()); }

mlir::Attribute ShapeType::getValueAttr() const { return getAttr(); }

llvm::TypeSize
ShapeType::getTypeSize(const mlir::DataLayout &dataLayout,
                       mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize
ShapeType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t ShapeType::getABIAlignment(const mlir::DataLayout &dataLayout,
                                    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t
ShapeType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                 mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t>
ShapeType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool ShapeType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
ShapeType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                         mlir::Location loc) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StrideType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type StrideType::parse(mlir::AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  auto attr = StrideAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::stride>(parser));
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    attr);
}

void StrideType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool StrideType::isStatic() const { return cutegen::is_static(getRef()); }

mlir::Attribute StrideType::getValueAttr() const { return getAttr(); }

// The compiler accepts every stride cutegen reads, mixed integer and
// scaled-basis leaves and mixed-depth basis paths included.
mlir::LogicalResult
StrideType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   StrideAttr attr) {
  return mlir::success();
}

llvm::TypeSize
StrideType::getTypeSize(const mlir::DataLayout &dataLayout,
                        mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize
StrideType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t
StrideType::getABIAlignment(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t
StrideType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                  mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t>
StrideType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool StrideType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
StrideType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                          mlir::Location loc) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LayoutType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type LayoutType::parse(mlir::AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  auto attr = LayoutAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::layout>(parser));
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    attr);
}

void LayoutType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool LayoutType::isStatic() const { return cutegen::is_static(getRef()); }

mlir::Attribute LayoutType::getValueAttr() const { return getAttr(); }

/// Verify LayoutType storage invariants. cutegen leaves these
/// unchecked at construction time for performance; enforce them at the
/// IR boundary.
mlir::LogicalResult
LayoutType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   LayoutAttr attr) {
  auto const &layout = attr.getRef();
  if (cutegen::has_error(layout)) {
    return emitError() << "expects no error(`x`) in layout, but got \""
                       << cutegen::to_string(layout) << "\"";
  }
  if (cutegen::any_leaf_is(layout.shape(), [](auto const &e) {
        return cutegen::holds_int(e) && cutegen::get_int(e) <= 0;
      })) {
    return emitError() << "expects positive shape mode, but got \""
                       << cutegen::to_string(layout) << "\"";
  }
  if (!cutegen::is_congruent(layout.shape(), layout.stride())) {
    return emitError()
           << "expects shape and stride profile to match, but got \""
           << cutegen::to_string(layout) << "\"";
  }
  return mlir::success();
}

llvm::TypeSize
LayoutType::getTypeSize(const mlir::DataLayout &dataLayout,
                        mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize
LayoutType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t
LayoutType::getABIAlignment(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t
LayoutType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                  mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t>
LayoutType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool LayoutType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
LayoutType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                          mlir::Location loc) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TileType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type TileType::parse(mlir::AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  auto attr = TileAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::tile>(parser));
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    attr);
}

void TileType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool TileType::isStatic() const { return cutegen::is_static(getRef()); }

mlir::Attribute TileType::getValueAttr() const { return getAttr(); }

// Like strides, the layouts of a tile are not restricted further.
mlir::LogicalResult
TileType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 TileAttr attr) {
  return mlir::success();
}

llvm::TypeSize
TileType::getTypeSize(const mlir::DataLayout &dataLayout,
                      mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize
TileType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t TileType::getABIAlignment(const mlir::DataLayout &dataLayout,
                                   mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t
TileType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t>
TileType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                           mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool TileType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
TileType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                        mlir::Location loc) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ComposedLayoutType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type ComposedLayoutType::parse(mlir::AsmParser &parser) {
  // Capture the start location so verify() failures (positivity,
  // congruence) are attributed to the type literal.
  auto loc = parser.getCurrentLocation();
  if (parser.parseLess()) {
    return {};
  }
  std::string str;
  if (parser.parseString(&str)) {
    return {};
  }
  auto opt = cutegen::from_string<cutegen::composed_layout>(str);
  if (!opt) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse composed_layout from \"" + str + "\"");
    return {};
  }
  if (parser.parseGreater()) {
    return {};
  }
  auto attr = ComposedLayoutAttr::get(parser.getContext(), std::move(*opt));
  return ComposedLayoutType::getChecked([&] { return parser.emitError(loc); },
                                        parser.getContext(), attr);
}

void ComposedLayoutType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool ComposedLayoutType::isStatic() const {
  // No free cutegen::is_static(composed_layout_t) exists — check each part.
  const auto &cl = getRef();
  bool b_ok = cutegen::is_static(cl.layout_b());
  bool off_ok = cutegen::is_static(cl.offset());
  bool a_ok = cl.is_a_swizzle() || cutegen::is_static(cl.layout_a());
  return b_ok && off_ok && a_ok;
}

mlir::Attribute ComposedLayoutType::getValueAttr() const { return getAttr(); }

/// Verify ComposedLayoutType storage invariants: no error sentinel,
/// positive static shape leaves, addable offset, and a sum that's
/// composable into the inner layout (scalar for swizzle, scalar or
/// weakly congruent for affine).
mlir::LogicalResult ComposedLayoutType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ComposedLayoutAttr attr) {
  auto const &composed = attr.getRef();
  if (cutegen::has_error(composed)) {
    return emitError() << "expects no error(`x`) in layout, but got \""
                       << cutegen::to_string(composed) << "\"";
  }
  // A trivial wrap (identity swizzle, no offset) is accepted and prints as
  // its plain layout, as the compiler does.
  auto hasNonPositiveShape = [](cutegen::layout const &lyt) {
    return cutegen::any_leaf_is(lyt.shape(), [](auto const &e) {
      return cutegen::holds_int(e) && cutegen::get_int(e) <= 0;
    });
  };
  if (hasNonPositiveShape(composed.layout_b()) ||
      (composed.is_a_affine() && hasNonPositiveShape(composed.layout_a()))) {
    return emitError() << "expects positive shape mode, but got \""
                       << cutegen::to_string(composed) << "\"";
  }

  // Addable: the offset and layout_eval(0, outer) must be compatible
  // under arith-tuple addition (no scaled-basis-vs-integer mismatch
  // or rank disagreement).
  auto offset_b = cutegen::layout_eval(0, composed.layout_b());
  auto sum =
      cutegen::arith_tuple_sum<decltype(offset_b)>(composed.offset(), offset_b);
  if (cutegen::has_error(sum)) {
    return emitError() << "expects offset and `layout_eval(0, outer)` to be "
                          "addable, but got "
                       << cutegen::to_string(composed.offset()) << " and "
                       << cutegen::to_string(offset_b);
  }

  // Swizzle inner: the arith-tuple sum must be a scalar (int or
  // dynamic int) — swizzles can only consume scalar indices, so any
  // scaled-basis structure surviving from the offset or outer is a
  // composability error.
  if (composed.is_a_swizzle() && !cutegen::holds_int_or_dynamic_int(sum)) {
    return emitError() << "swizzle layout expects scalar `offset` and "
                          "`outer` without scaled basis, but got "
                       << cutegen::to_string(composed.offset()) << " and "
                       << cutegen::to_string(composed.layout_b());
  }

  // Affine inner: the sum either is a scalar (and layout_a indexes
  // it directly) or matches layout_a's shape profile so each leaf
  // dispatches to the corresponding mode of A.
  if (composed.is_a_affine() && !cutegen::holds_int_or_dynamic_int(sum) &&
      !cutegen::weakly_congruent(sum, composed.layout_a().shape())) {
    return emitError()
           << "expects arith sum of `layout_eval(0, outer)` and `offset` to be "
              "scalar or weakly congruent to inner shape, but got sum "
           << cutegen::to_string(sum) << " and inner shape "
           << cutegen::to_string(composed.layout_a().shape());
  }

  return mlir::success();
}

llvm::TypeSize
ComposedLayoutType::getTypeSize(const mlir::DataLayout &dataLayout,
                                mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSize(cg::get_llvm_type(builder, getRef()));
}

llvm::TypeSize ComposedLayoutType::getTypeSizeInBits(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeSizeInBits(cg::get_llvm_type(builder, getRef()));
}

uint64_t
ComposedLayoutType::getABIAlignment(const mlir::DataLayout &dataLayout,
                                    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeABIAlignment(cg::get_llvm_type(builder, getRef()));
}

uint64_t ComposedLayoutType::getPreferredAlignment(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypePreferredAlignment(
      cg::get_llvm_type(builder, getRef()));
}

std::optional<uint64_t> ComposedLayoutType::getIndexBitwidth(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  OpBuilder builder(getContext());
  return dataLayout.getTypeIndexBitwidth(cg::get_llvm_type(builder, getRef()));
}

bool ComposedLayoutType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
ComposedLayoutType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                                  mlir::Location loc) const {
  OpBuilder builder(getContext());
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          cg::get_llvm_type(builder, getRef()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SwizzleType — custom assembly format
//===----------------------------------------------------------------------===//

mlir::Type SwizzleType::parse(mlir::AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  auto attr = SwizzleAttr::get(parser.getContext(),
                      parseLenientPayload<cutegen::swizzle>(parser));
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    attr);
}

void SwizzleType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getRef()));
  printer << '>';
}

bool SwizzleType::isStatic() const { return true; }

mlir::Attribute SwizzleType::getValueAttr() const { return getAttr(); }

/// Verify SwizzleType storage invariants. Defense-in-depth — the
/// string parser already rejects invalid swizzle parameters via
/// cutegen::from_string<cutegen::swizzle>, but programmatic
/// construction goes around the parser, so re-check here.
///
/// cutegen::swizzle::is_valid (swizzle.hpp) enforces two
/// invariants on S<num_bits, num_base, num_shift>:
///   1. |num_shift| + num_bits + num_base <= 32 — parameters
///      must fit within the swizzle's 32-bit working width.
///   2. |num_shift| >= num_bits — the shift amount must cover
///      the mask width so the Y and Z masks do not overlap.
mlir::LogicalResult
SwizzleType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    SwizzleAttr attr) {
  auto const &sw = attr.getRef();
  uint32_t numBits = sw.num_bits();
  uint32_t numBase = sw.num_base();
  int32_t numShift = sw.num_shift();
  uint32_t absShift = static_cast<uint32_t>(std::abs(numShift));
  constexpr uint32_t kBitWidth = 32;

  if (absShift + numBits + numBase > kBitWidth) {
    return emitError() << "expects |num_shift| + num_bits + num_base <= "
                       << kBitWidth << ", but got " << absShift << " + "
                       << numBits << " + " << numBase << " = "
                       << (absShift + numBits + numBase) << " for swizzle "
                       << cutegen::to_string(sw);
  }
  if (absShift < numBits) {
    return emitError() << "expects |num_shift| >= num_bits, but got "
                       << absShift << " < " << numBits << " for swizzle "
                       << cutegen::to_string(sw);
  }
  return mlir::success();
}

// A swizzle is a compile-time-only object whose runtime representation is an
// empty LLVM struct (mirroring convertSwizzleType in CuteTypeConverter.cpp), so
// the data-layout queries delegate to that struct rather than to
// cg::get_llvm_type.
static mlir::Type getSwizzleLLVMType(mlir::MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(ctx, SmallVector<mlir::Type>{});
}

llvm::TypeSize
SwizzleType::getTypeSize(const mlir::DataLayout &dataLayout,
                         mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeSize(getSwizzleLLVMType(getContext()));
}

llvm::TypeSize
SwizzleType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                               mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeSizeInBits(getSwizzleLLVMType(getContext()));
}

uint64_t
SwizzleType::getABIAlignment(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(getSwizzleLLVMType(getContext()));
}

uint64_t
SwizzleType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                   mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(getSwizzleLLVMType(getContext()));
}

std::optional<uint64_t>
SwizzleType::getIndexBitwidth(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeIndexBitwidth(getSwizzleLLVMType(getContext()));
}

bool SwizzleType::areCompatible(
    mlir::DataLayoutEntryListRef oldLayout,
    mlir::DataLayoutEntryListRef newLayout,
    mlir::DataLayoutSpecInterface newSpec,
    const mlir::DataLayoutIdentifiedEntryMap &identified) const {
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          getSwizzleLLVMType(getContext()))) {
    return dlTy.areCompatible(oldLayout, newLayout, newSpec, identified);
  }
  return true;
}

llvm::LogicalResult
SwizzleType::verifyEntries(mlir::DataLayoutEntryListRef entries,
                           mlir::Location loc) const {
  if (auto dlTy = llvm::dyn_cast<mlir::DataLayoutTypeInterface>(
          getSwizzleLLVMType(getContext()))) {
    return dlTy.verifyEntries(entries, loc);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// The DSL compiler's types and attributes (CuteTypesCompiler.td,
// CuteAttrsCompiler.td)
//===----------------------------------------------------------------------===//

#include "cute_ir/Dialect/Cute/IR/CuteEnums.cpp.inc"

static OptionalParseResult generatedCuteTypeParser(DialectAsmParser &parser,
                                                   StringRef *mnemonic,
                                                   Type &value) {
  OptionalParseResult result = generatedCuteTypeParser(
      static_cast<AsmParser &>(parser), mnemonic, value);
  if (result.has_value())
    return result;
  StringRef digits = *mnemonic;
  unsigned width;
  if (!digits.consume_front("i") || digits.empty() ||
      !llvm::all_of(digits, llvm::isDigit) || digits.getAsInteger(10, width))
    return std::nullopt;
  value = ConstrainedIntType::parse(parser, width);
  return success(!!value);
}

static LogicalResult generatedCuteTypePrinter(Type type,
                                              DialectAsmPrinter &printer) {
  if (auto integer = llvm::dyn_cast<ConstrainedIntType>(type)) {
    integer.print(printer);
    return success();
  }
  return generatedCuteTypePrinter(type, static_cast<AsmPrinter &>(printer));
}

// Bits of an integer or float (tf32 is stored in 32), or of a vector of them.
static unsigned storageBits(Type type) {
  if (auto integer = llvm::dyn_cast_or_null<IntegerType>(type))
    return integer.getWidth();
  if (auto fp = llvm::dyn_cast_or_null<FloatType>(type))
    return fp.isTF32() ? 32 : fp.getWidth();
  if (auto vector = llvm::dyn_cast_or_null<VectorType>(type))
    return vector.getNumElements() * storageBits(vector.getElementType());
  return 0;
}

static bool isIdentitySwizzle(SwizzleAttr swizzle) {
  return !swizzle || cutegen::to_string(swizzle.getRef()) == "S<0,4,3>";
}

uint64_t PtrType::getNaturalAlignment(Type valueType) {
  unsigned bits = 0;
  if (auto sparse = llvm::dyn_cast_or_null<SparseElemType>(valueType))
    bits = storageBits(sparse.getPhysicalType());
  else if (llvm::isa_and_nonnull<IntegerType, FloatType>(valueType))
    bits = storageBits(valueType);
  if (bits <= 1)
    return 1;
  return bits % 8 ? 0 : bits / 8;
}

BitLayoutAttr PtrType::getDefaultBitLayout(Type valueType,
                                           AddressSpace addressSpace) {
  unsigned chunk = addressSpace == AddressSpace::tmem ? 32 : 8;
  std::string text;
  if (auto sparse = llvm::dyn_cast_or_null<SparseElemType>(valueType)) {
    unsigned bits = storageBits(sparse.getPhysicalType());
    if (!bits)
      return {};
    text = llvm::formatv("({0},({1},{2})):(1,(0,{0}))", bits,
                         sparse.getNumLogical(), chunk / std::gcd(bits, chunk))
               .str();
  } else if (llvm::isa_and_nonnull<IntegerType, FloatType>(valueType)) {
    unsigned bits = storageBits(valueType);
    unsigned stride = bits == 1 ? 8 : bits;
    text = llvm::formatv("({0},{1}):(1,{2})", bits,
                         chunk / std::gcd(stride, chunk), stride)
               .str();
  } else {
    return {};
  }
  auto layout = cutegen::from_string<cutegen::layout>(text);
  if (!layout)
    return {};
  MLIRContext *ctx = valueType.getContext();
  return BitLayoutAttr::get(ctx, LayoutAttr::get(ctx, std::move(*layout)));
}

// The bits of a chunk: the largest extent times stride of the element mode.
static std::optional<int64_t> chunkBits(const cutegen::layout &layout) {
  auto elements = cutegen::get(layout, 1);
  auto shape = cutegen::flatten(elements.shape());
  auto stride = cutegen::flatten(elements.stride());
  if (cutegen::holds_int(shape) && cutegen::holds_int(stride))
    return shape.as_int64() * stride.as_int64();
  int64_t bits = 0;
  for (size_t i = 0; i < cutegen::rank(shape); ++i) {
    auto extent = cutegen::get(shape, i);
    auto step = cutegen::get(stride, i);
    if (!cutegen::holds_int(extent) || !cutegen::holds_int(step))
      return std::nullopt;
    bits = std::max(bits, extent.as_int64() * step.as_int64());
  }
  return bits;
}

LogicalResult BitLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                    LayoutAttr layout) {
  const cutegen::layout &l = layout.getRef();
  if (!cutegen::is_static(l))
    return emitError() << "bitlayout must be fully static but got " << layout;
  if (cutegen::rank(l) != 2)
    return emitError() << "bitlayout must have 2 modes corresponding to "
                          "(element bit, element index), but got "
                       << cutegen::rank(l);
  auto elements = cutegen::size(l, 1);
  if (cutegen::holds_int(elements) && elements.as_int64() == 0)
    return emitError()
           << "bitlayout must contain at least one logical element, but got 0";
  return success();
}

//===- ptr and memref -----------------------------------------------------===//

namespace {
struct PtrFields {
  Type valueType;
  AddressSpace space = AddressSpace::generic;
  std::optional<uint64_t> alignment;
  SwizzleAttr swizzle;
  BitLayoutAttr bitlayout;
};
} // namespace

// `bitlayout<"layout">`, the keyword already read.
static BitLayoutAttr parseBitLayout(AsmParser &parser) {
  std::string text;
  if (failed(parser.parseLess()) || failed(parser.parseString(&text)) ||
      failed(parser.parseGreater()))
    return {};
  auto layout = cutegen::from_string<cutegen::layout>(text);
  if (!layout) {
    parser.emitError(parser.getCurrentLocation(), "expects cg::layout");
    return {};
  }
  MLIRContext *ctx = parser.getContext();
  return BitLayoutAttr::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); }, ctx,
      LayoutAttr::get(ctx, std::move(*layout)));
}

// `S<b,m,s>`, the `S` already read.
static SwizzleAttr parseInlineSwizzle(AsmParser &parser) {
  int64_t bits, base, shift;
  if (failed(parser.parseLess()) || failed(parser.parseInteger(bits)) ||
      failed(parser.parseComma()) || failed(parser.parseInteger(base)) ||
      failed(parser.parseComma()) || failed(parser.parseInteger(shift)) ||
      failed(parser.parseGreater()))
    return {};
  auto value = cutegen::from_string<cutegen::swizzle>(
      llvm::formatv("S<{0},{1},{2}>", bits, base, shift).str());
  if (!value) {
    parser.emitError(parser.getCurrentLocation(), "invalid swizzle parameters");
    return {};
  }
  return SwizzleAttr::get(parser.getContext(), std::move(*value));
}

// A layout or composed layout spelled as a string; a composed layout that is
// a plain one (identity swizzle, no offset) is read as that layout.
static Type parseLayoutPayload(AsmParser &parser) {
  std::string text;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseString(&text)))
    return {};
  MLIRContext *ctx = parser.getContext();
  auto emitError = [&] { return parser.emitError(loc); };
  if (auto value = cutegen::from_string<cutegen::layout>(text))
    return LayoutType::getChecked(emitError, ctx,
                                  LayoutAttr::get(ctx, std::move(*value)));
  if (auto value = cutegen::from_string<cutegen::composed_layout>(text)) {
    if (value->is_normal_layout())
      return LayoutType::getChecked(emitError, ctx,
                                    LayoutAttr::get(ctx, value->layout_b()));
    return ComposedLayoutType::getChecked(
        emitError, ctx, ComposedLayoutAttr::get(ctx, std::move(*value)));
  }
  parser.emitError(parser.getCurrentLocation(), "layout parse failure");
  return {};
}

static void printLayoutPayload(AsmPrinter &printer, Type layout) {
  if (auto plain = llvm::dyn_cast<LayoutType>(layout))
    printer.printString(cutegen::to_string(plain.getRef()));
  else
    printer.printString(
        cutegen::to_string(llvm::cast<ComposedLayoutType>(layout).getRef()));
}

static LogicalResult memrefLayoutExpected(AsmParser &parser) {
  return parser.emitError(parser.getCurrentLocation(),
                          "failed to parse CuteMemRefType parameter `,` "
                          "expected, layout cannot be omitted.");
}

// `[type, ][space][, align<N>][, S<b,m,s>][, bitlayout<"L">]` in this order;
// a memref requires the type and ends with its layout string instead of the
// bit layout, which follows the layout.
static FailureOr<PtrFields> parsePtrFields(AsmParser &parser, Type *layout) {
  PtrFields fields;
  OptionalParseResult typed = parser.parseOptionalType(fields.valueType);
  if (typed.has_value() && failed(*typed))
    return failure();
  if (layout && !typed.has_value())
    return memrefLayoutExpected(parser);
  enum { kSpace, kAlign, kSwizzle, kBitLayout, kDone } next = kSpace;
  bool first = !typed.has_value();
  while (true) {
    if (!first && failed(parser.parseOptionalComma()))
      break;
    StringRef keyword;
    if (failed(parser.parseOptionalKeyword(&keyword))) {
      if (layout) {
        if (!(*layout = parseLayoutPayload(parser)))
          return failure();
        return fields;
      }
      if (first)
        break;
      parser.emitError(parser.getCurrentLocation(),
                       "expected a pointer field after `,`");
      return failure();
    }
    first = false;
    if (auto space = symbolizeAddressSpace(keyword);
        space && next <= kSpace) {
      fields.space = *space;
      next = kAlign;
    } else if (keyword == "align" && next <= kAlign) {
      uint64_t alignment;
      if (failed(parser.parseLess()) || failed(parser.parseInteger(alignment)) ||
          failed(parser.parseGreater()))
        return failure();
      fields.alignment = alignment;
      next = kSwizzle;
    } else if (keyword == "S" && next <= kSwizzle) {
      if (!(fields.swizzle = parseInlineSwizzle(parser)))
        return failure();
      next = kBitLayout;
    } else if (keyword == "bitlayout" && !layout && next <= kBitLayout) {
      if (!(fields.bitlayout = parseBitLayout(parser)))
        return failure();
      next = kDone;
    } else {
      parser.emitError(parser.getCurrentLocation(), "unexpected pointer field `")
          << keyword << "`";
      return failure();
    }
  }
  if (layout)
    return memrefLayoutExpected(parser);
  return fields;
}

// Defaults are left out: they are what the parser fills in.
static PtrType buildPtr(AsmParser &parser, const PtrFields &fields) {
  SwizzleAttr swizzle = isIdentitySwizzle(fields.swizzle) ? SwizzleAttr()
                                                          : fields.swizzle;
  BitLayoutAttr bitlayout = fields.bitlayout;
  if (bitlayout &&
      bitlayout == PtrType::getDefaultBitLayout(fields.valueType, fields.space))
    bitlayout = {};
  return PtrType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), fields.valueType, fields.space,
      fields.alignment.value_or(PtrType::getNaturalAlignment(fields.valueType)),
      swizzle, bitlayout);
}

static void printPtrFields(AsmPrinter &printer, PtrType ptr) {
  if (ptr.getValueType()) {
    printer.printType(ptr.getValueType());
    printer << ", ";
  }
  printer << stringifyAddressSpace(ptr.getAddressSpace());
  if (ptr.getAlignment() != PtrType::getNaturalAlignment(ptr.getValueType()))
    printer << ", align<" << ptr.getAlignment() << '>';
  if (!isIdentitySwizzle(ptr.getSwizzle()))
    printer << ", " << cutegen::to_string(ptr.getSwizzle().getRef());
}

static void printBitLayout(AsmPrinter &printer, PtrType ptr) {
  BitLayoutAttr bitlayout = ptr.getBitlayout();
  if (!bitlayout || bitlayout == PtrType::getDefaultBitLayout(
                                     ptr.getValueType(), ptr.getAddressSpace()))
    return;
  printer << ", bitlayout<";
  printer.printString(cutegen::to_string(bitlayout.getLayout().getRef()));
  printer << '>';
}

Type PtrType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};
  FailureOr<PtrFields> fields = parsePtrFields(parser, nullptr);
  if (failed(fields) || failed(parser.parseGreater()))
    return {};
  return buildPtr(parser, *fields);
}

void PtrType::print(AsmPrinter &printer) const {
  printer << '<';
  printPtrFields(printer, *this);
  printBitLayout(printer, *this);
  printer << '>';
}

LogicalResult PtrType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type valueType, AddressSpace addressSpace,
                              uint64_t alignment, SwizzleAttr swizzle,
                              BitLayoutAttr bitlayout) {
  if (alignment == 0 && getNaturalAlignment(valueType) != 0)
    return emitError() << "expects alignment >= 1, got 0";
  bool tmem = addressSpace == AddressSpace::tmem;
  if (tmem && swizzle && swizzle.getRef().num_bits() != 0)
    return emitError() << "tmem pointer does not support non-identity swizzle";
  BitLayoutAttr defaultBitLayout = getDefaultBitLayout(valueType, addressSpace);
  BitLayoutAttr effective = bitlayout ? bitlayout : defaultBitLayout;
  if (!effective)
    return success();
  if (auto bits = chunkBits(effective.getLayout().getRef())) {
    if (*bits % 8 != 0)
      return emitError() << "bitlayout chunk must be byte aligned, but size is "
                         << *bits << " bits";
    if (tmem && (*bits < 32 || !llvm::isPowerOf2_64(*bits)))
      return emitError()
             << "tmem pointer only supports 32 bit chunks that is a power of "
                "2, got "
             << *bits;
  }
  if (llvm::isa_and_nonnull<SparseElemType>(valueType) &&
      effective != defaultBitLayout)
    return emitError() << "expects bitlayout to be " << defaultBitLayout
                       << " for valueType '" << valueType << "'but got "
                       << effective;
  return success();
}

// Whether the offsets of `layout` are whole aligned blocks of `sparsity`
// elements: the contiguous run from offset 0 (stride-1 mode, then each mode
// whose stride is the run so far) is a multiple of it, and so is every other
// stride.
static bool isDenseForSparsity(const cutegen::layout &layout,
                               int64_t sparsity) {
  if (!cutegen::is_static(layout))
    return false;
  auto shape = cutegen::flatten(layout.shape());
  auto stride = cutegen::flatten(layout.stride());
  std::vector<std::pair<int64_t, int64_t>> modes;  // (stride, extent)
  auto add = [&](const auto &extent, const auto &step) {
    if (extent.as_int64() != 1 && step.as_int64() != 0)
      modes.emplace_back(step.as_int64(), extent.as_int64());
  };
  if (cutegen::holds_int(shape)) {
    add(shape, stride);
  } else {
    for (size_t i = 0; i < cutegen::rank(shape); ++i)
      add(cutegen::get(shape, i), cutegen::get(stride, i));
  }
  llvm::sort(modes, [](auto a, auto b) {
    return std::abs(a.first) < std::abs(b.first);
  });
  int64_t run = 1;
  bool contiguous = true;
  for (auto [step, extent] : modes) {
    if (contiguous && step == run) {
      run *= extent;
      continue;
    }
    contiguous = false;
    if (step % sparsity != 0)
      return false;
  }
  return run % sparsity == 0;
}

Type mlir::cutlass_compiler::cute::MemRefType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};
  Type layout;
  FailureOr<PtrFields> fields = parsePtrFields(parser, &layout);
  if (failed(fields))
    return {};
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("bitlayout")) ||
        !(fields->bitlayout = parseBitLayout(parser)))
      return {};
  }
  if (failed(parser.parseGreater()))
    return {};
  PtrType ptr = buildPtr(parser, *fields);
  if (!ptr)
    return {};
  return getChecked([&] { return parser.emitError(parser.getCurrentLocation()); },
                    parser.getContext(), ptr, layout);
}

void mlir::cutlass_compiler::cute::MemRefType::print(
    AsmPrinter &printer) const {
  printer << '<';
  printPtrFields(printer, getPtr());
  printer << ", ";
  printLayoutPayload(printer, getLayout());
  printBitLayout(printer, getPtr());
  printer << '>';
}

LogicalResult mlir::cutlass_compiler::cute::MemRefType::verify(
    function_ref<InFlightDiagnostic()> emitError, PtrType ptr, Type layout) {
  if (!ptr)
    return emitError() << "expects a CuTe pointer";
  if (!llvm::isa<LayoutType, ComposedLayoutType>(layout))
    return emitError() << "expects a CuTe layout or composed layout";
  Type valueType = ptr.getValueType();
  if (!valueType || !(valueType.isIntOrIndexOrFloat() ||
                      llvm::isa<PtrType, SparseElemType>(valueType)))
    return emitError()
           << "expects value type to be int, float, ptr or sparse_elem, but got '"
           << valueType << "'";
  if (auto sparse = llvm::dyn_cast<SparseElemType>(valueType)) {
    int64_t sparsity = sparse.getNumLogical();
    bool dense;
    if (auto composed = llvm::dyn_cast<ComposedLayoutType>(layout)) {
      // A swizzle may not move elements within a block.
      const auto &c = composed.getRef();
      dense = c.is_a_swizzle() &&
              (int64_t(1) << c.swizzle_a().num_base()) % sparsity == 0 &&
              cutegen::holds_int(c.offset()) &&
              c.offset().as_int64() % sparsity == 0 &&
              isDenseForSparsity(c.layout_b(), sparsity);
    } else {
      dense = isDenseForSparsity(llvm::cast<LayoutType>(layout).getRef(),
                                 sparsity);
    }
    if (!dense)
      return emitError()
             << "layout must be dense but cannot be divided by sparsity";
  }
  if (auto composed = llvm::dyn_cast<ComposedLayoutType>(layout)) {
    if (!isIdentitySwizzle(ptr.getSwizzle()) && composed.getRef().is_a_swizzle() &&
        !composed.getRef().swizzle_a().is_identity())
      return emitError() << "cannot have both a swizzle on the pointer and a "
                            "non-identity swizzle in the layout";
  } else if (cutegen::has_scaled_basis(
                 llvm::cast<LayoutType>(layout).getRef().stride())) {
    return emitError() << "expected iterator and layout codomain to have "
                          "congruent profiles, but got iterator '"
                       << ptr << "' and layout layout: "
                       << cutegen::to_string(
                              llvm::cast<LayoutType>(layout).getRef());
  }
  return success();
}

//===- arith_tuple_iter, coord_tensor, i<N>, fast_divmod_divisor, ... -----===//

Type ArithTupleIteratorType::parse(AsmParser &parser) {
  auto tuple = IntTupleType::parse(parser);
  return tuple ? get(parser.getContext(), llvm::cast<IntTupleType>(tuple))
               : Type();
}

void ArithTupleIteratorType::print(AsmPrinter &printer) const {
  getArithTuple().print(printer);
}

Type CoordTensorType::parse(AsmParser &parser) {
  std::string text;
  if (failed(parser.parseLess()) || failed(parser.parseString(&text)))
    return {};
  auto tuple = cutegen::from_string<cutegen::int_tuple>(text);
  if (!tuple) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse CoordTensorType parameter 'arith_tuple' "
                     "which is to be a `cutegen::int_tuple`");
    return {};
  }
  Type layout;
  if (failed(parser.parseComma()) || !(layout = parseLayoutPayload(parser)) ||
      failed(parser.parseGreater()))
    return {};
  return getChecked([&] { return parser.emitError(parser.getCurrentLocation()); },
                    parser.getContext(),
                    IntTupleType::get(parser.getContext(), std::move(*tuple)),
                    layout);
}

void CoordTensorType::print(AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getArithTuple().getRef()));
  printer << ", ";
  printLayoutPayload(printer, getLayout());
  printer << '>';
}

LogicalResult CoordTensorType::verify(
    function_ref<InFlightDiagnostic()> emitError, IntTupleType arithTuple,
    Type layout) {
  if (!arithTuple)
    return emitError() << "expects an arithmetic tuple";
  if (!llvm::isa<LayoutType, ComposedLayoutType>(layout))
    return emitError() << "expects a CuTe layout or composed layout";
  return success();
}

// `<[pow2, ]divby D>`, or nothing for a plain integer.
Type ConstrainedIntType::parse(AsmParser &parser, unsigned width) {
  int64_t divisibility = 1;
  bool isPow2 = false;
  if (succeeded(parser.parseOptionalLess())) {
    isPow2 = succeeded(parser.parseOptionalKeyword("pow2"));
    bool divided = !isPow2 || succeeded(parser.parseOptionalComma());
    if (divided && (failed(parser.parseKeyword("divby")) ||
                    failed(parser.parseInteger(divisibility))))
      return {};
    if (failed(parser.parseGreater()))
      return {};
  }
  return get(parser.getContext(), divisibility, width, isPow2);
}

void ConstrainedIntType::print(AsmPrinter &printer) const {
  printer << 'i' << getWidth();
  if (!getIsPow2() && getDivisibility() == 1)
    return;
  printer << '<';
  if (getIsPow2())
    printer << "pow2";
  if (getDivisibility() != 1)
    printer << (getIsPow2() ? ", " : "") << "divby " << getDivisibility();
  printer << '>';
}

Type FastDivmodDivisorType::parse(AsmParser &parser) {
  unsigned width;
  if (failed(parser.parseLess()) || failed(parser.parseInteger(width)))
    return {};
  bool isPow2 = false;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("is_pow_2")))
      return {};
    isPow2 = true;
  }
  if (failed(parser.parseGreater()))
    return {};
  return get(parser.getContext(), width, isPow2);
}

void FastDivmodDivisorType::print(AsmPrinter &printer) const {
  printer << '<' << getWidth();
  if (getIsPow_2())
    printer << ", is_pow_2";
  printer << '>';
}

LogicalResult SparseElemType::verify(
    function_ref<InFlightDiagnostic()> emitError, int numLogical,
    Type physicalType) {
  if (numLogical <= 0)
    return emitError() << "sparse element type logical count must be > 0";
  Type element = physicalType;
  if (auto vector = llvm::dyn_cast_or_null<VectorType>(physicalType))
    element = vector.getElementType();
  if (!llvm::isa_and_nonnull<IntegerType, FloatType>(element))
    return emitError() << "unsupported physical element type '" << physicalType
                       << "'";
  return success();
}

LogicalResult TiledMmaType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   Type mmaAtom, LayoutAttr atomLayoutMNK,
                                   TileAttr permutationMNK) {
  if (cutegen::rank(atomLayoutMNK.getRef()) != 3)
    return emitError() << "TiledMMA requires rank-3 atomLayoutMNK";
  if (permutationMNK && cutegen::rank(permutationMNK.getRef()) != 3)
    return emitError() << "TiledMMA requires rank-3 permutationMNK";
  return success();
}
