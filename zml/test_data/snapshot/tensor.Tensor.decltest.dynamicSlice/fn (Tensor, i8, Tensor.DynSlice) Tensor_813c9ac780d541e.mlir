module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c, %arg1, sizes = [2, 2] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<2x5xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
