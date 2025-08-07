module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<6xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<6xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<-1.500000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %1 = stablehlo.remainder %arg0, %0 : tensor<6xf32>
    return %1 : tensor<6xf32>
  }
}
