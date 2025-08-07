module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<200x100x300xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<100x200x1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<100x200x300xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [1, 0], start_index_map = [2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 300>}> : (tensor<200x100x300xf16>, tensor<100x200x1xi32>) -> tensor<100x200x300xf16>
    return %0 : tensor<100x200x300xf16>
  }
}
