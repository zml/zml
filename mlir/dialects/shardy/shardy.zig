// /* Copyright 2024 The Shardy Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

// #ifndef SHARDY_INTEGRATIONS_C_ATTRIBUTES_H_
// #define SHARDY_INTEGRATIONS_C_ATTRIBUTES_H_

// #include <stdint.h>
// #include <sys/types.h>

// #include "mlir-c/IR.h"
// #include "mlir-c/Support.h"

// #ifdef __cplusplus
// extern "C" {
// #endif

// //===----------------------------------------------------------------------===//
// // MeshAxisAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsAMeshAxisAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyMeshAxisAttrGet(MlirContext ctx,
//                                                     MlirStringRef name,
//                                                     int64_t size);

// MLIR_CAPI_EXPORTED MlirStringRef sdyMeshAxisAttrGetName(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t sdyMeshAxisAttrGetSize(MlirAttribute attr);

// //===----------------------------------------------------------------------===//
// // MeshAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsAMeshAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyMeshAttrGet(MlirContext ctx, intptr_t nAxes,
//                                                 const MlirAttribute* axes,
//                                                 intptr_t nDeviceIds,
//                                                 const int64_t* deviceIds);

// MLIR_CAPI_EXPORTED int64_t sdyMeshAttrGetDeviceIdsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t sdyMeshAttrGetDeviceIdsElem(MlirAttribute attr,
//                                                         int64_t pos);

// MLIR_CAPI_EXPORTED intptr_t sdyMeshAttrGetAxesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyMeshAttrGetAxesElem(MlirAttribute attr,
//                                                         intptr_t pos);

// //===----------------------------------------------------------------------===//
// // SubAxisInfoAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsASubAxisInfoAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdySubAxisInfoAttrGet(MlirContext ctx,
//                                                        int64_t preSize,
//                                                        int64_t size);

// MLIR_CAPI_EXPORTED int64_t sdySubAxisInfoAttrGetPreSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t sdySubAxisInfoAttrGetSize(MlirAttribute attr);

// //===----------------------------------------------------------------------===//
// // AxisRefAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsAnAxisRefAttr(MlirAttribute attr);

// // NOTE: Pass a null subAxisInfo if the attr has none.
// MLIR_CAPI_EXPORTED MlirAttribute sdyAxisRefAttrGet(MlirContext ctx,
//                                                    MlirStringRef name,
//                                                    MlirAttribute subAxisInfo);

// MLIR_CAPI_EXPORTED MlirStringRef sdyAxisRefAttrGetName(MlirAttribute attr);

// // NOTE: Attr is null if there is no sub axis info.
// MLIR_CAPI_EXPORTED MlirAttribute
// sdyAxisRefAttrGetSubAxisInfo(MlirAttribute attr);

// //===----------------------------------------------------------------------===//
// // DimensionShardingAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsADimensionShardingAttr(
//     MlirAttribute attr);

// // NOTE: Specify -1 if the attr has no priority.
// MLIR_CAPI_EXPORTED MlirAttribute sdyDimensionShardingAttrGet(
//     MlirContext ctx, intptr_t nAxes, const MlirAttribute* axes, bool isClosed,
//     int64_t priority);

// MLIR_CAPI_EXPORTED intptr_t
// sdyDimensionShardingAttrGetAxesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyDimensionShardingAttrGetAxesElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED bool sdyDimensionShardingAttrGetIsClosed(MlirAttribute attr);

// // NOTE: returns -1 if the attr has no priority.
// MLIR_CAPI_EXPORTED int64_t
// sdyDimensionShardingAttrGetPriority(MlirAttribute attr);

// //===----------------------------------------------------------------------===//
// // TensorShardingAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsATensorShardingAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyTensorShardingAttrGet(
//     MlirContext ctx, MlirAttribute meshOrRef, intptr_t nDimShardings,
//     const MlirAttribute* dimShardings, intptr_t nReplicatedAxes,
//     const MlirAttribute* replicatedAxes, intptr_t nUnreducedAxes,
//     const MlirAttribute* unreducedAxes);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyTensorShardingAttrGetMeshOrRef(MlirAttribute attr);

// MLIR_CAPI_EXPORTED intptr_t
// sdyTensorShardingAttrGetDimShardingsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyTensorShardingAttrGetDimShardingsElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyTensorShardingAttrGetReplicatedAxesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyTensorShardingAttrGetReplicatedAxesElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyTensorShardingAttrGetUnreducedAxesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyTensorShardingAttrGetUnreducedAxesElem(MlirAttribute attr, intptr_t pos);

// //===----------------------------------------------------------------------===//
// // TensorShardingPerValueAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsATensorShardingPerValueAttr(
//     MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyTensorShardingPerValueAttrGet(
//     MlirContext ctx, intptr_t nShardings, const MlirAttribute* shardings);

// MLIR_CAPI_EXPORTED intptr_t
// sdyTensorShardingPerValueAttrGetShardingsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyTensorShardingPerValueAttrGetShardingsElem(MlirAttribute attr, intptr_t pos);

// //===----------------------------------------------------------------------===//
// // DimMappingAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsADimMappingAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyDimMappingAttrGet(
//     MlirContext ctx, intptr_t nFactorIndices, const int64_t* factorIndices);

// MLIR_CAPI_EXPORTED intptr_t
// sdyDimMappingAttrGetFactorIndicesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t
// sdyDimMappingAttrGetFactorIndicesElem(MlirAttribute attr, intptr_t pos);

// //===----------------------------------------------------------------------===//
// // TensorMappingAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsATensorMappingAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyTensorMappingAttrGet(
//     MlirContext ctx, intptr_t nMappings, const MlirAttribute* mappings);

// MLIR_CAPI_EXPORTED intptr_t sdyTensorMappingAttrGetRank(MlirAttribute attr);

// MLIR_CAPI_EXPORTED intptr_t
// sdyTensorMappingAttrGetDimMappingsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyTensorMappingAttrGetDimMappingsElem(MlirAttribute attr, intptr_t pos);

// //===----------------------------------------------------------------------===//
// // OpShardingRuleAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsAOpShardingRuleAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyOpShardingRuleAttrGet(
//     MlirContext ctx, intptr_t nFactorSizes, const int64_t* factorSizes,
//     intptr_t nOperandMappings, const MlirAttribute* operandMappings,
//     intptr_t nResultMappings, const MlirAttribute* resultMappings,
//     intptr_t nReductionFactors, const int64_t* reductionFactors,
//     intptr_t nNeedReplicationFactors, const int64_t* needReplicationFactors,
//     intptr_t nPermutationFactors, const int64_t* permutationFactors,
//     intptr_t nBlockedPropagationFactors,
//     const int64_t* blockedPropagationFactors, bool isCustomRule);

// MLIR_CAPI_EXPORTED bool sdyOpShardingRuleAttrGetIsCustom(MlirAttribute attr);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetFactorSizesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t
// sdyOpShardingRuleAttrGetFactorSizesElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetOperandMappingsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyOpShardingRuleAttrGetOperandMappingsElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetResultMappingsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute
// sdyOpShardingRuleAttrGetResultMappingsElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetReductionFactorsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t
// sdyOpShardingRuleAttrGetReductionFactorsElem(MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(
//     MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetPermutationFactorsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t sdyOpShardingRuleAttrGetPermutationFactorsElem(
//     MlirAttribute attr, intptr_t pos);

// MLIR_CAPI_EXPORTED intptr_t
// sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED int64_t
// sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(MlirAttribute attr,
//                                                       intptr_t pos);

// //===----------------------------------------------------------------------===//
// // ManualAxesAttr
// //===----------------------------------------------------------------------===//

// MLIR_CAPI_EXPORTED bool sdyAttributeIsAManualAxesAttr(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirAttribute sdyManualAxesAttrGet(
//     MlirContext ctx, intptr_t nAxes, const MlirAttribute* axes);

// MLIR_CAPI_EXPORTED intptr_t sdyManualAxesAttrGetAxesSize(MlirAttribute attr);

// MLIR_CAPI_EXPORTED MlirStringRef sdyManualAxesAttrGetAxesElem(
//   MlirAttribute attr, intptr_t pos);

// #ifdef __cplusplus
// }
// #endif

// #endif  // SHARDY_INTEGRATIONS_C_ATTRIBUTES_H_



// hugo@9960x-5090x2:~/zml$ cat  bazel-out/linux_amd64-opt/bin/examples/sharding/sharding_c.zig | grep sdy
//     pub const sdyMeshAxisAttrGet = __root.sdyMeshAxisAttrGet;
//     pub const sdyMeshAttrGet = __root.sdyMeshAttrGet;
//     pub const sdySubAxisInfoAttrGet = __root.sdySubAxisInfoAttrGet;
//     pub const sdyAxisRefAttrGet = __root.sdyAxisRefAttrGet;
//     pub const sdyDimensionShardingAttrGet = __root.sdyDimensionShardingAttrGet;
//     pub const sdyTensorShardingAttrGet = __root.sdyTensorShardingAttrGet;
//     pub const sdyTensorShardingPerValueAttrGet = __root.sdyTensorShardingPerValueAttrGet;
//     pub const sdyDimMappingAttrGet = __root.sdyDimMappingAttrGet;
//     pub const sdyTensorMappingAttrGet = __root.sdyTensorMappingAttrGet;
//     pub const sdyOpShardingRuleAttrGet = __root.sdyOpShardingRuleAttrGet;
//     pub const sdyManualAxesAttrGet = __root.sdyManualAxesAttrGet;
//     pub const sdyAttributeIsAMeshAxisAttr = __root.sdyAttributeIsAMeshAxisAttr;
//     pub const sdyMeshAxisAttrGetName = __root.sdyMeshAxisAttrGetName;
//     pub const sdyMeshAxisAttrGetSize = __root.sdyMeshAxisAttrGetSize;
//     pub const sdyAttributeIsAMeshAttr = __root.sdyAttributeIsAMeshAttr;
//     pub const sdyMeshAttrGetDeviceIdsSize = __root.sdyMeshAttrGetDeviceIdsSize;
//     pub const sdyMeshAttrGetDeviceIdsElem = __root.sdyMeshAttrGetDeviceIdsElem;
//     pub const sdyMeshAttrGetAxesSize = __root.sdyMeshAttrGetAxesSize;
//     pub const sdyMeshAttrGetAxesElem = __root.sdyMeshAttrGetAxesElem;
//     pub const sdyAttributeIsASubAxisInfoAttr = __root.sdyAttributeIsASubAxisInfoAttr;
//     pub const sdySubAxisInfoAttrGetPreSize = __root.sdySubAxisInfoAttrGetPreSize;
//     pub const sdySubAxisInfoAttrGetSize = __root.sdySubAxisInfoAttrGetSize;
//     pub const sdyAttributeIsAnAxisRefAttr = __root.sdyAttributeIsAnAxisRefAttr;
//     pub const sdyAxisRefAttrGetName = __root.sdyAxisRefAttrGetName;
//     pub const sdyAxisRefAttrGetSubAxisInfo = __root.sdyAxisRefAttrGetSubAxisInfo;
//     pub const sdyAttributeIsADimensionShardingAttr = __root.sdyAttributeIsADimensionShardingAttr;
//     pub const sdyDimensionShardingAttrGetAxesSize = __root.sdyDimensionShardingAttrGetAxesSize;
//     pub const sdyDimensionShardingAttrGetAxesElem = __root.sdyDimensionShardingAttrGetAxesElem;
//     pub const sdyDimensionShardingAttrGetIsClosed = __root.sdyDimensionShardingAttrGetIsClosed;
//     pub const sdyDimensionShardingAttrGetPriority = __root.sdyDimensionShardingAttrGetPriority;
//     pub const sdyAttributeIsATensorShardingAttr = __root.sdyAttributeIsATensorShardingAttr;
//     pub const sdyTensorShardingAttrGetMeshOrRef = __root.sdyTensorShardingAttrGetMeshOrRef;
//     pub const sdyTensorShardingAttrGetDimShardingsSize = __root.sdyTensorShardingAttrGetDimShardingsSize;
//     pub const sdyTensorShardingAttrGetDimShardingsElem = __root.sdyTensorShardingAttrGetDimShardingsElem;
//     pub const sdyTensorShardingAttrGetReplicatedAxesSize = __root.sdyTensorShardingAttrGetReplicatedAxesSize;
//     pub const sdyTensorShardingAttrGetReplicatedAxesElem = __root.sdyTensorShardingAttrGetReplicatedAxesElem;
//     pub const sdyTensorShardingAttrGetUnreducedAxesSize = __root.sdyTensorShardingAttrGetUnreducedAxesSize;
//     pub const sdyTensorShardingAttrGetUnreducedAxesElem = __root.sdyTensorShardingAttrGetUnreducedAxesElem;
//     pub const sdyAttributeIsATensorShardingPerValueAttr = __root.sdyAttributeIsATensorShardingPerValueAttr;
//     pub const sdyTensorShardingPerValueAttrGetShardingsSize = __root.sdyTensorShardingPerValueAttrGetShardingsSize;
//     pub const sdyTensorShardingPerValueAttrGetShardingsElem = __root.sdyTensorShardingPerValueAttrGetShardingsElem;
//     pub const sdyAttributeIsADimMappingAttr = __root.sdyAttributeIsADimMappingAttr;
//     pub const sdyDimMappingAttrGetFactorIndicesSize = __root.sdyDimMappingAttrGetFactorIndicesSize;
//     pub const sdyDimMappingAttrGetFactorIndicesElem = __root.sdyDimMappingAttrGetFactorIndicesElem;
//     pub const sdyAttributeIsATensorMappingAttr = __root.sdyAttributeIsATensorMappingAttr;
//     pub const sdyTensorMappingAttrGetRank = __root.sdyTensorMappingAttrGetRank;
//     pub const sdyTensorMappingAttrGetDimMappingsSize = __root.sdyTensorMappingAttrGetDimMappingsSize;
//     pub const sdyTensorMappingAttrGetDimMappingsElem = __root.sdyTensorMappingAttrGetDimMappingsElem;
//     pub const sdyAttributeIsAOpShardingRuleAttr = __root.sdyAttributeIsAOpShardingRuleAttr;
//     pub const sdyOpShardingRuleAttrGetIsCustom = __root.sdyOpShardingRuleAttrGetIsCustom;
//     pub const sdyOpShardingRuleAttrGetFactorSizesSize = __root.sdyOpShardingRuleAttrGetFactorSizesSize;
//     pub const sdyOpShardingRuleAttrGetFactorSizesElem = __root.sdyOpShardingRuleAttrGetFactorSizesElem;
//     pub const sdyOpShardingRuleAttrGetOperandMappingsSize = __root.sdyOpShardingRuleAttrGetOperandMappingsSize;
//     pub const sdyOpShardingRuleAttrGetOperandMappingsElem = __root.sdyOpShardingRuleAttrGetOperandMappingsElem;
//     pub const sdyOpShardingRuleAttrGetResultMappingsSize = __root.sdyOpShardingRuleAttrGetResultMappingsSize;
//     pub const sdyOpShardingRuleAttrGetResultMappingsElem = __root.sdyOpShardingRuleAttrGetResultMappingsElem;
//     pub const sdyOpShardingRuleAttrGetReductionFactorsSize = __root.sdyOpShardingRuleAttrGetReductionFactorsSize;
//     pub const sdyOpShardingRuleAttrGetReductionFactorsElem = __root.sdyOpShardingRuleAttrGetReductionFactorsElem;
//     pub const sdyOpShardingRuleAttrGetNeedReplicationFactorsSize = __root.sdyOpShardingRuleAttrGetNeedReplicationFactorsSize;
//     pub const sdyOpShardingRuleAttrGetNeedReplicationFactorsElem = __root.sdyOpShardingRuleAttrGetNeedReplicationFactorsElem;
//     pub const sdyOpShardingRuleAttrGetPermutationFactorsSize = __root.sdyOpShardingRuleAttrGetPermutationFactorsSize;
//     pub const sdyOpShardingRuleAttrGetPermutationFactorsElem = __root.sdyOpShardingRuleAttrGetPermutationFactorsElem;
//     pub const sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize = __root.sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize;
//     pub const sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem = __root.sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem;
//     pub const sdyAttributeIsAManualAxesAttr = __root.sdyAttributeIsAManualAxesAttr;
//     pub const sdyManualAxesAttrGetAxesSize = __root.sdyManualAxesAttrGetAxesSize;
//     pub const sdyManualAxesAttrGetAxesElem = __root.sdyManualAxesAttrGetAxesElem;
// pub extern fn sdyAttributeIsAMeshAxisAttr(attr: MlirAttribute) bool;
// pub extern fn sdyMeshAxisAttrGet(ctx: MlirContext, name: MlirStringRef, size: i64) MlirAttribute;
// pub extern fn sdyMeshAxisAttrGetName(attr: MlirAttribute) MlirStringRef;
// pub extern fn sdyMeshAxisAttrGetSize(attr: MlirAttribute) i64;
// pub extern fn sdyAttributeIsAMeshAttr(attr: MlirAttribute) bool;
// pub extern fn sdyMeshAttrGet(ctx: MlirContext, nAxes: isize, axes: [*c]const MlirAttribute, nDeviceIds: isize, deviceIds: [*c]const i64) MlirAttribute;
// pub extern fn sdyMeshAttrGetDeviceIdsSize(attr: MlirAttribute) i64;
// pub extern fn sdyMeshAttrGetDeviceIdsElem(attr: MlirAttribute, pos: i64) i64;
// pub extern fn sdyMeshAttrGetAxesSize(attr: MlirAttribute) isize;
// pub extern fn sdyMeshAttrGetAxesElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyAttributeIsASubAxisInfoAttr(attr: MlirAttribute) bool;
// pub extern fn sdySubAxisInfoAttrGet(ctx: MlirContext, preSize: i64, size: i64) MlirAttribute;
// pub extern fn sdySubAxisInfoAttrGetPreSize(attr: MlirAttribute) i64;
// pub extern fn sdySubAxisInfoAttrGetSize(attr: MlirAttribute) i64;
// pub extern fn sdyAttributeIsAnAxisRefAttr(attr: MlirAttribute) bool;
// pub extern fn sdyAxisRefAttrGet(ctx: MlirContext, name: MlirStringRef, subAxisInfo: MlirAttribute) MlirAttribute;
// pub extern fn sdyAxisRefAttrGetName(attr: MlirAttribute) MlirStringRef;
// pub extern fn sdyAxisRefAttrGetSubAxisInfo(attr: MlirAttribute) MlirAttribute;
// pub extern fn sdyAttributeIsADimensionShardingAttr(attr: MlirAttribute) bool;
// pub extern fn sdyDimensionShardingAttrGet(ctx: MlirContext, nAxes: isize, axes: [*c]const MlirAttribute, isClosed: bool, priority: i64) MlirAttribute;
// pub extern fn sdyDimensionShardingAttrGetAxesSize(attr: MlirAttribute) isize;
// pub extern fn sdyDimensionShardingAttrGetAxesElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyDimensionShardingAttrGetIsClosed(attr: MlirAttribute) bool;
// pub extern fn sdyDimensionShardingAttrGetPriority(attr: MlirAttribute) i64;
// pub extern fn sdyAttributeIsATensorShardingAttr(attr: MlirAttribute) bool;
// pub extern fn sdyTensorShardingAttrGet(ctx: MlirContext, meshOrRef: MlirAttribute, nDimShardings: isize, dimShardings: [*c]const MlirAttribute, nReplicatedAxes: isize, replicatedAxes: [*c]const MlirAttribute, nUnreducedAxes: isize, unreducedAxes: [*c]const MlirAttribute) MlirAttribute;
// pub extern fn sdyTensorShardingAttrGetMeshOrRef(attr: MlirAttribute) MlirAttribute;
// pub extern fn sdyTensorShardingAttrGetDimShardingsSize(attr: MlirAttribute) isize;
// pub extern fn sdyTensorShardingAttrGetDimShardingsElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyTensorShardingAttrGetReplicatedAxesSize(attr: MlirAttribute) isize;
// pub extern fn sdyTensorShardingAttrGetReplicatedAxesElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyTensorShardingAttrGetUnreducedAxesSize(attr: MlirAttribute) isize;
// pub extern fn sdyTensorShardingAttrGetUnreducedAxesElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyAttributeIsATensorShardingPerValueAttr(attr: MlirAttribute) bool;
// pub extern fn sdyTensorShardingPerValueAttrGet(ctx: MlirContext, nShardings: isize, shardings: [*c]const MlirAttribute) MlirAttribute;
// pub extern fn sdyTensorShardingPerValueAttrGetShardingsSize(attr: MlirAttribute) isize;
// pub extern fn sdyTensorShardingPerValueAttrGetShardingsElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyAttributeIsADimMappingAttr(attr: MlirAttribute) bool;
// pub extern fn sdyDimMappingAttrGet(ctx: MlirContext, nFactorIndices: isize, factorIndices: [*c]const i64) MlirAttribute;
// pub extern fn sdyDimMappingAttrGetFactorIndicesSize(attr: MlirAttribute) isize;
// pub extern fn sdyDimMappingAttrGetFactorIndicesElem(attr: MlirAttribute, pos: isize) i64;
// pub extern fn sdyAttributeIsATensorMappingAttr(attr: MlirAttribute) bool;
// pub extern fn sdyTensorMappingAttrGet(ctx: MlirContext, nMappings: isize, mappings: [*c]const MlirAttribute) MlirAttribute;
// pub extern fn sdyTensorMappingAttrGetRank(attr: MlirAttribute) isize;
// pub extern fn sdyTensorMappingAttrGetDimMappingsSize(attr: MlirAttribute) isize;
// pub extern fn sdyTensorMappingAttrGetDimMappingsElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyAttributeIsAOpShardingRuleAttr(attr: MlirAttribute) bool;
// pub extern fn sdyOpShardingRuleAttrGet(ctx: MlirContext, nFactorSizes: isize, factorSizes: [*c]const i64, nOperandMappings: isize, operandMappings: [*c]const MlirAttribute, nResultMappings: isize, resultMappings: [*c]const MlirAttribute, nReductionFactors: isize, reductionFactors: [*c]const i64, nNeedReplicationFactors: isize, needReplicationFactors: [*c]const i64, nPermutationFactors: isize, permutationFactors: [*c]const i64, nBlockedPropagationFactors: isize, blockedPropagationFactors: [*c]const i64, isCustomRule: bool) MlirAttribute;
// pub extern fn sdyOpShardingRuleAttrGetIsCustom(attr: MlirAttribute) bool;
// pub extern fn sdyOpShardingRuleAttrGetFactorSizesSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetFactorSizesElem(attr: MlirAttribute, pos: isize) i64;
// pub extern fn sdyOpShardingRuleAttrGetOperandMappingsSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetOperandMappingsElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyOpShardingRuleAttrGetResultMappingsSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetResultMappingsElem(attr: MlirAttribute, pos: isize) MlirAttribute;
// pub extern fn sdyOpShardingRuleAttrGetReductionFactorsSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetReductionFactorsElem(attr: MlirAttribute, pos: isize) i64;
// pub extern fn sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(attr: MlirAttribute, pos: isize) i64;
// pub extern fn sdyOpShardingRuleAttrGetPermutationFactorsSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetPermutationFactorsElem(attr: MlirAttribute, pos: isize) i64;
// pub extern fn sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(attr: MlirAttribute) isize;
// pub extern fn sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(attr: MlirAttribute, pos: isize) i64;
// pub extern fn sdyAttributeIsAManualAxesAttr(attr: MlirAttribute) bool;
// pub extern fn sdyManualAxesAttrGet(ctx: MlirContext, nAxes: isize, axes: [*c]const MlirAttribute) MlirAttribute;
// pub extern fn sdyManualAxesAttrGetAxesSize(attr: MlirAttribute) isize;
// pub extern fn sdyManualAxesAttrGetAxesElem(attr: MlirAttribute, pos: isize) MlirStringRef;
// pub extern fn mlirGetDialectHandle__sdy__() MlirDialectHandle;
