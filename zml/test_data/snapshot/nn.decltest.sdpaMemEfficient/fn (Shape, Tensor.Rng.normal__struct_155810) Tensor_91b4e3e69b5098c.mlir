module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<512x512xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<512> : tensor<2xi64>
    %0 = stablehlo.rng %cst, %cst_0, %c, distribution =  NORMAL : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
}
