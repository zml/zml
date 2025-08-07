module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<4x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<4x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<4x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<i1>
    %1 = "stablehlo.if"(%0) ({
      %2 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      stablehlo.return %2 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<4x4xf32>
    }, {
      %2 = stablehlo.dot_general %arg2, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      stablehlo.return %2 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<4x4xf32>
    }) : (tensor<i1>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
