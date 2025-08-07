module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<10xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<10xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.reshape %arg2 : (tensor<2xf32>) -> tensor<2xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg1 {operandSegmentSizes = dense<1> : tensor<3xi32>} : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }
}
