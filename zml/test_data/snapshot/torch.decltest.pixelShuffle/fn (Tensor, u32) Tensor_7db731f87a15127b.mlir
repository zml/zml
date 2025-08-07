module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x9x4x4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x12x12xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x9x4x4xi32>) -> tensor<1x1x3x3x4x4xi32>
    %1 = stablehlo.transpose %0, dims = [0, 1, 4, 2, 5, 3] : (tensor<1x1x3x3x4x4xi32>) -> tensor<1x1x4x3x4x3xi32>
    %2 = stablehlo.reshape %1 : (tensor<1x1x4x3x4x3xi32>) -> tensor<1x1x12x12xi32>
    return %2 : tensor<1x1x12x12xi32>
  }
}
