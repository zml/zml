module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<3> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.convert %c_2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.convert %c_1 : (tensor<i32>) -> tensor<f32>
    %4 = stablehlo.multiply %3, %3 : tensor<f32>
    %5 = stablehlo.reshape %4 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.convert %c_0 : (tensor<i32>) -> tensor<f32>
    %7 = stablehlo.multiply %6, %6 : tensor<f32>
    %8 = stablehlo.reshape %7 : (tensor<f32>) -> tensor<1xf32>
    %9 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %10 = stablehlo.multiply %9, %9 : tensor<f32>
    %11 = stablehlo.reshape %10 : (tensor<f32>) -> tensor<1xf32>
    %12 = stablehlo.concatenate %2, %5, %8, %11, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
    return %12 : tensor<4xf32>
  }
}
