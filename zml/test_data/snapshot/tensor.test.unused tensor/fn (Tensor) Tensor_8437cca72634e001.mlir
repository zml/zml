module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<10xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i32}) -> (tensor<10xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    return %arg0 : tensor<10xf32>
  }
}
