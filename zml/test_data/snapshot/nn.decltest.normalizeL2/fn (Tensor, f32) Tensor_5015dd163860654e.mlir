module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %1 = stablehlo.power %arg0, %0 : tensor<2x2xf32>
    %2 = stablehlo.reduce(%1 init: %cst_0) across dimensions = [1] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %9 = stablehlo.add %arg2, %arg1 : tensor<f32>
      stablehlo.return %9 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<2x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<2x1xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<2x2xf32>
    return %8 : tensor<2x2xf32>
  }
}
