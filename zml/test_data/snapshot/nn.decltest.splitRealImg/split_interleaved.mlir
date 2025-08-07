module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<20xf32>
    %1 = stablehlo.reshape %0 : (tensor<20xf32>) -> tensor<5x4xf32>
    %2 = stablehlo.slice %1 [0:5, 0:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %3 = stablehlo.reshape %2 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %4 = stablehlo.iota dim = 0 : tensor<10xf32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<10xf32>
    %7 = stablehlo.reshape %6 : (tensor<10xf32>) -> tensor<5x2xf32>
    %8 = stablehlo.compare  EQ, %3, %7,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %9 = stablehlo.reshape %8 : (tensor<5x2xi1>) -> tensor<10xi1>
    %10 = stablehlo.convert %9 : (tensor<10xi1>) -> tensor<10xi32>
    %11 = stablehlo.reduce(%10 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %24 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %24 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.slice %1 [0:5, 1:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %14 = stablehlo.reshape %13 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %16 = stablehlo.add %6, %15 : tensor<10xf32>
    %17 = stablehlo.reshape %16 : (tensor<10xf32>) -> tensor<5x2xf32>
    %18 = stablehlo.compare  EQ, %14, %17,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %19 = stablehlo.reshape %18 : (tensor<5x2xi1>) -> tensor<10xi1>
    %20 = stablehlo.convert %19 : (tensor<10xi1>) -> tensor<10xi32>
    %21 = stablehlo.reduce(%20 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %24 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %24 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.add %12, %22 : tensor<1xi32>
    return %23 : tensor<1xi32>
  }
}
