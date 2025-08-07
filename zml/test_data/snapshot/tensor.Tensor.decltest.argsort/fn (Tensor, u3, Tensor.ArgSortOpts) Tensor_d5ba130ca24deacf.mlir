module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x5xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.iota dim = 0 : tensor<5xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<5xi32>) -> tensor<2x5xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.compare  LT, %arg1, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    }) {operandSegmentSizes = dense<2> : tensor<1xi32>} : (tensor<2x5xf32>, tensor<2x5xi32>) -> (tensor<2x5xf32>, tensor<2x5xi32>)
    return %2#1 : tensor<2x5xi32>
  }
}
