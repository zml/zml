module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
  }
}
