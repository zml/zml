module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x1x2x2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x6x6xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0.333333343> : tensor<f32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<6xf32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %2 = stablehlo.add %0, %1 : tensor<6xf32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<6xf32>
    %5 = stablehlo.floor %4 : tensor<6xf32>
    %6 = stablehlo.convert %5 : (tensor<6xf32>) -> tensor<6xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 2>}> : (tensor<1x1x2x2xi32>, tensor<6xi32>) -> tensor<1x1x6x2xi32>
    %8 = "stablehlo.gather"(%7, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 6, 1>}> : (tensor<1x1x6x2xi32>, tensor<6xi32>) -> tensor<1x1x6x6xi32>
    return %8 : tensor<1x1x6x6xi32>
  }
}
