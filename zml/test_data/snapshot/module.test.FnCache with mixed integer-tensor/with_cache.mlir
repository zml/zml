module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @"module.test.FnCache with mixed integer-tensor.Layer._fwd_975b269a9674d3be"(%arg0: tensor<2x2xf16>, %arg1: tensor<2xf16>) -> tensor<2xf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<2xf16>
    %2 = stablehlo.add %0, %1 : tensor<2xf16>
    return %2 : tensor<2xf16>
  }
  func.func @"module.test.FnCache with mixed integer-tensor.Layer._fwd_9dc18e8af9fce7c1"(%arg0: tensor<3x2xf16>, %arg1: tensor<2xf16>) -> tensor<3xf16> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf16>, tensor<2xf16>) -> tensor<3xf16>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<3xf16>
    %2 = stablehlo.add %0, %1 : tensor<3xf16>
    return %2 : tensor<3xf16>
  }
  func.func @main(%arg0: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<3x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<3xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = call @"module.test.FnCache with mixed integer-tensor.Layer._fwd_975b269a9674d3be"(%arg0, %arg3) {operandSegmentSizes = dense<2> : tensor<1xi32>} : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %1 = call @"module.test.FnCache with mixed integer-tensor.Layer._fwd_975b269a9674d3be"(%arg1, %0) {operandSegmentSizes = dense<2> : tensor<1xi32>} : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %2 = call @"module.test.FnCache with mixed integer-tensor.Layer._fwd_9dc18e8af9fce7c1"(%arg2, %1) {operandSegmentSizes = dense<2> : tensor<1xi32>} : (tensor<3x2xf16>, tensor<2xf16>) -> tensor<3xf16>
    return %2 : tensor<3xf16>
  }
}
