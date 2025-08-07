module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x4x6xui16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2x2x3xui16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3], start_index_map = [1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 2, 3>}> : (tensor<2x4x6xui16>, tensor<2x2xi32>) -> tensor<2x2x2x3xui16>
    return %0 : tensor<2x2x2x3xui16>
  }
}
