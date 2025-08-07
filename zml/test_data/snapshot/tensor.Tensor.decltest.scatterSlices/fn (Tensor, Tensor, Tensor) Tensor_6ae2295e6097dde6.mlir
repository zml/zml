module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x3x4x2xui16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2x3x2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2x2x3x2x2xui16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x3x4x2xui16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.iota dim = 1 : tensor<2x2x3xi32>
    %1 = stablehlo.reshape %0 : (tensor<2x2x3xi32>) -> tensor<2x2x3x1xi32>
    %2 = stablehlo.slice %arg1 [0:2, 0:2, 0:3, 1:2] : (tensor<2x2x3x2xi32>) -> tensor<2x2x3x1xi32>
    %3 = stablehlo.reshape %2 : (tensor<2x2x3x1xi32>) -> tensor<2x2x3xi32>
    %4 = stablehlo.reshape %3 : (tensor<2x2x3xi32>) -> tensor<2x2x3x1xi32>
    %5 = stablehlo.slice %arg1 [0:2, 0:2, 0:3, 0:1] : (tensor<2x2x3x2xi32>) -> tensor<2x2x3x1xi32>
    %6 = stablehlo.reshape %5 : (tensor<2x2x3x1xi32>) -> tensor<2x2x3xi32>
    %7 = stablehlo.reshape %6 : (tensor<2x2x3xi32>) -> tensor<2x2x3x1xi32>
    %8 = stablehlo.concatenate %1, %4, %7, dim = 3 : (tensor<2x2x3x1xi32>, tensor<2x2x3x1xi32>, tensor<2x2x3x1xi32>) -> tensor<2x2x3x3xi32>
    %9 = "stablehlo.scatter"(%arg0, %8, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [3, 4], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<ui16>, %arg4: tensor<ui16>):
      %10 = stablehlo.add %arg3, %arg4 : tensor<ui16>
      stablehlo.return %10 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<ui16>
    }) {operandSegmentSizes = dense<1> : tensor<3xi32>} : (tensor<2x3x4x2xui16>, tensor<2x2x3x3xi32>, tensor<2x2x3x2x2xui16>) -> tensor<2x3x4x2xui16>
    return %9 : tensor<2x3x4x2xui16>
  }
}
