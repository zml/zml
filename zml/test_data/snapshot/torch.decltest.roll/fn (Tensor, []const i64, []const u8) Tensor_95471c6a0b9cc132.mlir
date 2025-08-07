module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<4x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<4x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<4xf32>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.add %0, %1 : tensor<4xf32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %4 = stablehlo.remainder %2, %3 : tensor<4xf32>
    %5 = stablehlo.convert %4 : (tensor<4xf32>) -> tensor<4xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<4x2xf32>, tensor<4xi32>) -> tensor<4x2xf32>
    %7 = stablehlo.iota dim = 0 : tensor<2xf32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.add %7, %8 : tensor<2xf32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %11 = stablehlo.remainder %9, %10 : tensor<2xf32>
    %12 = stablehlo.convert %11 : (tensor<2xf32>) -> tensor<2xi32>
    %13 = "stablehlo.gather"(%6, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2xf32>, tensor<2xi32>) -> tensor<4x2xf32>
    return %13 : tensor<4x2xf32>
  }
}
