module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x2x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<2x2x2xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.iota dim = 2 : tensor<2x2x5xi32>
    %1:2 = "stablehlo.reduce_window"(%arg0, %0, %cst, %c) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 2>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %6 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %8 = stablehlo.and %6, %7 : tensor<i1>
      %9 = stablehlo.or %4, %8 : tensor<i1>
      %10 = stablehlo.select %9, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %5, %10 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<f32>, tensor<i32>
    }) {operandSegmentSizes = dense<2> : tensor<2xi32>} : (tensor<2x2x5xf32>, tensor<2x2x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2x2xf32>, tensor<2x2x2xi32>)
    return %1#0, %1#1 : tensor<2x2x2xf32>, tensor<2x2x2xi32>
  }
}
