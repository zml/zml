module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<10xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<10> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.convert %c_1 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.pad %2, %cst, low = [0], high = [9], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<10xf32>
    %4:2 = stablehlo.while(%iterArg = %3, %iterArg_2 = %c_0) : tensor<10xf32>, tensor<i32> attributes {operandSegmentSizes = dense<2> : tensor<1xi32>}
    cond {
      %5 = stablehlo.compare  LT, %iterArg_2, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %5 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    } do {
      %5 = stablehlo.convert %iterArg_2 : (tensor<i32>) -> tensor<f32>
      %6 = stablehlo.multiply %5, %5 : tensor<f32>
      %7 = stablehlo.reshape %6 : (tensor<f32>) -> tensor<1xf32>
      %8 = stablehlo.dynamic_update_slice %iterArg, %7, %iterArg_2 {operandSegmentSizes = dense<1> : tensor<3xi32>} : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      %9 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
      stablehlo.return %8, %9 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<10xf32>, tensor<i32>
    }
    return %4#0 : tensor<10xf32>
  }
}
