module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<10x20xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<8x1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<8x7x20xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 7, 20>}> : (tensor<10x20xf16>, tensor<8x1xi32>) -> tensor<8x7x20xf16>
    return %0 : tensor<8x7x20xf16>
  }
}
