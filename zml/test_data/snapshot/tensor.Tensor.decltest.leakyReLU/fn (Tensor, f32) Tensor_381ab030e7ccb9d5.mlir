module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<2xf32>
    %4 = stablehlo.minimum %3, %0 : tensor<2xf32>
    %5 = stablehlo.add %1, %4 : tensor<2xf32>
    return %5 : tensor<2xf32>
  }
}
