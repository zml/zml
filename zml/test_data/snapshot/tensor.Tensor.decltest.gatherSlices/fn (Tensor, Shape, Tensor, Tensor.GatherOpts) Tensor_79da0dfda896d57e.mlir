module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<10xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<0xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<10xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 10>}> : (tensor<10xf16>, tensor<0xi32>) -> tensor<10xf16>
    return %0 : tensor<10xf16>
  }
}
