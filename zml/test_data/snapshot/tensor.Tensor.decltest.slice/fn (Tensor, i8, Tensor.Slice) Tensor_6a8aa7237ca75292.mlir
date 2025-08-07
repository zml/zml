module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.slice %arg0 [0:2, 1:5:2] : (tensor<2x5xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.reshape %0 : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
