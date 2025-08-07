module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<5x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2xf32>, tensor<5x2xf32>) -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
