module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x2x2x2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2x4x4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<4xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.add %0, %1 : tensor<4xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<4xf32>
    %4 = stablehlo.floor %3 : tensor<4xf32>
    %5 = stablehlo.convert %4 : (tensor<4xf32>) -> tensor<4xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 2, 2, 1, 2>}> : (tensor<2x2x2x2xi32>, tensor<4xi32>) -> tensor<2x2x4x2xi32>
    %7 = "stablehlo.gather"(%6, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 2, 2, 4, 1>}> : (tensor<2x2x4x2xi32>, tensor<4xi32>) -> tensor<2x2x4x4xi32>
    return %7 : tensor<2x2x4x4xi32>
  }
}
