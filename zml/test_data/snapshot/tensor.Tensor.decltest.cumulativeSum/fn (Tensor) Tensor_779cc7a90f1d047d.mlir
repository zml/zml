module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [4, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 5>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }) {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<2x5xf32>, tensor<f32>) -> tensor<2x5xf32>
    return %0 : tensor<2x5xf32>
  }
}
