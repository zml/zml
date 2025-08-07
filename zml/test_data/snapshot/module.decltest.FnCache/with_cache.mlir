module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @module.decltest.FnCache.Layer._fwd_b4103d12b0fc0621(%arg0: tensor<2x2xf16>, %arg1: tensor<2xf16>, %arg2: tensor<2xf16>) -> tensor<2xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %1 = stablehlo.add %0, %arg1 : tensor<2xf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<2xf16>
    %3 = stablehlo.maximum %1, %2 : tensor<2xf16>
    return %3 : tensor<2xf16>
  }
  func.func @module.decltest.FnCache.Layer._fwd_544d74fe80abb17(%arg0: tensor<3x2xf16>, %arg1: tensor<3xf16>, %arg2: tensor<2xf16>) -> tensor<3xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf16>, tensor<2xf16>) -> tensor<3xf16>
    %1 = stablehlo.add %0, %arg1 : tensor<3xf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<3xf16>
    %3 = stablehlo.maximum %1, %2 : tensor<3xf16>
    return %3 : tensor<3xf16>
  }
  func.func @main(%arg0: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg4: tensor<3x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg5: tensor<3xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg6: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<3xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = call @module.decltest.FnCache.Layer._fwd_b4103d12b0fc0621(%arg0, %arg1, %arg6) {operandSegmentSizes = dense<3> : tensor<1xi32>} : (tensor<2x2xf16>, tensor<2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %1 = call @module.decltest.FnCache.Layer._fwd_b4103d12b0fc0621(%arg2, %arg3, %0) {operandSegmentSizes = dense<3> : tensor<1xi32>} : (tensor<2x2xf16>, tensor<2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %2 = call @module.decltest.FnCache.Layer._fwd_544d74fe80abb17(%arg4, %arg5, %1) {operandSegmentSizes = dense<3> : tensor<1xi32>} : (tensor<3x2xf16>, tensor<3xf16>, tensor<2xf16>) -> tensor<3xf16>
    return %2 : tensor<3xf16>
  }
}
