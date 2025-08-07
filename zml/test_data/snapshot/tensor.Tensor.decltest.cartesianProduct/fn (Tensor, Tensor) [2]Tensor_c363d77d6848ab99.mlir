module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<6xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<6x4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<6x4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<6xi32>) -> tensor<6x4xi32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<4xi32>) -> tensor<6x4xi32>
    return %0, %1 : tensor<6x4xi32>, tensor<6x4xi32>
  }
}
