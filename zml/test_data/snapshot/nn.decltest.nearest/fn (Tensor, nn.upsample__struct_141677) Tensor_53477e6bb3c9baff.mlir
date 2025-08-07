module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x1x1x2x2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x2x4x4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<2xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %2 = stablehlo.add %0, %1 : tensor<2xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<2xf32>
    %4 = stablehlo.floor %3 : tensor<2xf32>
    %5 = stablehlo.convert %4 : (tensor<2xf32>) -> tensor<2xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3, 4], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 2, 2>}> : (tensor<1x1x1x2x2xi32>, tensor<2xi32>) -> tensor<1x1x2x2x2xi32>
    %7 = stablehlo.iota dim = 0 : tensor<4xf32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %9 = stablehlo.add %7, %8 : tensor<4xf32>
    %10 = stablehlo.multiply %9, %8 : tensor<4xf32>
    %11 = stablehlo.floor %10 : tensor<4xf32>
    %12 = stablehlo.convert %11 : (tensor<4xf32>) -> tensor<4xi32>
    %13 = "stablehlo.gather"(%6, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 4], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 2, 1, 2>}> : (tensor<1x1x2x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x4x2xi32>
    %14 = "stablehlo.gather"(%13, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], collapsed_slice_dims = [4], start_index_map = [4], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 2, 4, 1>}> : (tensor<1x1x2x4x2xi32>, tensor<4xi32>) -> tensor<1x1x2x4x4xi32>
    return %14 : tensor<1x1x2x4x4xi32>
  }
}
