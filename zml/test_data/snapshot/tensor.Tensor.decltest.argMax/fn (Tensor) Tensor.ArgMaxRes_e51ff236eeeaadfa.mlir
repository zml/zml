module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1x1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<5xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<5xi32>) -> tensor<1x5xi32>
    %2:2 = stablehlo.reduce(%arg0 init: %cst), (%1 init: %c) across dimensions = [1] {operandSegmentSizes = dense<2> : tensor<2xi32>} : (tensor<1x5xf32>, tensor<1x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %5 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.or %5, %6 : tensor<i1>
      %8 = stablehlo.select %7, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %9 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %11 = stablehlo.and %9, %10 : tensor<i1>
      %12 = stablehlo.or %7, %11 : tensor<i1>
      %13 = stablehlo.select %12, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %8, %13 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<f32>, tensor<i32>
    }
    %3 = stablehlo.broadcast_in_dim %2#0, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %4 = stablehlo.broadcast_in_dim %2#1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    return %3, %4 : tensor<1x1xf32>, tensor<1x1xi32>
  }
}
