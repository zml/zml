module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = "stablehlo.case"(%arg0) ({
      %1 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
      stablehlo.return %1 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<2x2xf32>
    }, {
      %1 = stablehlo.add %arg1, %arg2 : tensor<2x2xf32>
      stablehlo.return %1 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<2x2xf32>
    }, {
      %1 = stablehlo.subtract %arg1, %arg2 : tensor<2x2xf32>
      stablehlo.return %1 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<2x2xf32>
    }) : (tensor<i32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
