module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x3x4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x3x8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<8xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.add %0, %1 : tensor<8xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<8xf32>
    %4 = stablehlo.floor %3 : tensor<8xf32>
    %5 = stablehlo.convert %4 : (tensor<8xf32>) -> tensor<8xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 2, 3, 1>}> : (tensor<2x3x4xi32>, tensor<8xi32>) -> tensor<2x3x8xi32>
    return %6 : tensor<2x3x8xi32>
  }
}
