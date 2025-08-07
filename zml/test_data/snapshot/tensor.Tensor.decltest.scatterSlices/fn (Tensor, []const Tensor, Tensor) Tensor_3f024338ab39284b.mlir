module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<3x3xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2x3xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<3x3xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.reshape %arg1 : (tensor<2xi32>) -> tensor<2x1xi32>
    %1 = "stablehlo.scatter"(%arg0, %0, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %2 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }) {operandSegmentSizes = dense<1> : tensor<3xi32>} : (tensor<3x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
    return %1 : tensor<3x3xi32>
  }
}
