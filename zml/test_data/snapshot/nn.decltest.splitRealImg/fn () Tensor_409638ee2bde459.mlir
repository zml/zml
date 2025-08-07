module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<20xf32>
    %1 = stablehlo.reshape %0 : (tensor<20xf32>) -> tensor<5x4xf32>
    %2 = stablehlo.slice %1 [0:5, 0:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %3 = stablehlo.reshape %2 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %4 = stablehlo.iota dim = 0 : tensor<5xf32>
    %5 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<5xf32>
    %7 = stablehlo.reshape %6 : (tensor<5xf32>) -> tensor<5x1xf32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %9 = stablehlo.add %6, %8 : tensor<5xf32>
    %10 = stablehlo.reshape %9 : (tensor<5xf32>) -> tensor<5x1xf32>
    %11 = stablehlo.concatenate %7, %10, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
    %12 = stablehlo.compare  EQ, %3, %11,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %13 = stablehlo.reshape %12 : (tensor<5x2xi1>) -> tensor<10xi1>
    %14 = stablehlo.convert %13 : (tensor<10xi1>) -> tensor<10xi32>
    %15 = stablehlo.reduce(%14 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %32 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %32 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.slice %1 [0:5, 2:4] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %18 = stablehlo.reshape %17 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %19 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %20 = stablehlo.add %6, %19 : tensor<5xf32>
    %21 = stablehlo.reshape %20 : (tensor<5xf32>) -> tensor<5x1xf32>
    %22 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %23 = stablehlo.add %6, %22 : tensor<5xf32>
    %24 = stablehlo.reshape %23 : (tensor<5xf32>) -> tensor<5x1xf32>
    %25 = stablehlo.concatenate %21, %24, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
    %26 = stablehlo.compare  EQ, %18, %25,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %27 = stablehlo.reshape %26 : (tensor<5x2xi1>) -> tensor<10xi1>
    %28 = stablehlo.convert %27 : (tensor<10xi1>) -> tensor<10xi32>
    %29 = stablehlo.reduce(%28 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %32 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %32 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.add %16, %30 : tensor<1xi32>
    return %31 : tensor<1xi32>
  }
}
