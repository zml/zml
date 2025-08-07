module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<8xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x8x1x1xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<8xf16>) -> tensor<1x8xf16>
    %1 = stablehlo.reshape %0 : (tensor<1x8xf16>) -> tensor<1x8x1xf16>
    %2 = stablehlo.reshape %1 : (tensor<1x8x1xf16>) -> tensor<1x8x1x1xf16>
    return %2 : tensor<1x8x1x1xf16>
  }
}
