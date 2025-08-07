module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<i1> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<ui8> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<f32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<bf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<ui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<false> : tensor<i1>
    %c_0 = stablehlo.constant dense<0> : tensor<ui8>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %c_3 = stablehlo.constant dense<0> : tensor<ui64>
    return %c, %c_0, %c_1, %cst, %cst_2, %c_3 : tensor<i1>, tensor<ui8>, tensor<i32>, tensor<f32>, tensor<bf16>, tensor<ui64>
  }
}
