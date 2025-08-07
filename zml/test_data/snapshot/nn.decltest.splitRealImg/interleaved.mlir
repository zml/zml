module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.iota dim = 0 : tensor<20xf32>
    %1 = stablehlo.reshape %0 : (tensor<20xf32>) -> tensor<5x4xf32>
    %2 = stablehlo.slice %1 [0:5, 0:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %3 = stablehlo.reshape %2 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %4 = stablehlo.reshape %3 : (tensor<5x2xf32>) -> tensor<5x2x1xf32>
    %5 = stablehlo.slice %1 [0:5, 1:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %6 = stablehlo.reshape %5 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %7 = stablehlo.reshape %6 : (tensor<5x2xf32>) -> tensor<5x2x1xf32>
    %8 = stablehlo.concatenate %4, %7, dim = 2 : (tensor<5x2x1xf32>, tensor<5x2x1xf32>) -> tensor<5x2x2xf32>
    %9 = stablehlo.reshape %8 : (tensor<5x2x2xf32>) -> tensor<5x4xf32>
    %10 = stablehlo.slice %9 [0:5, 0:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %11 = stablehlo.reshape %10 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %12 = stablehlo.compare  EQ, %3, %11,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %13 = stablehlo.reshape %12 : (tensor<5x2xi1>) -> tensor<10xi1>
    %14 = stablehlo.convert %13 : (tensor<10xi1>) -> tensor<10xi32>
    %15 = stablehlo.reduce(%14 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %25 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %25 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.slice %9 [0:5, 1:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %18 = stablehlo.reshape %17 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %19 = stablehlo.compare  EQ, %6, %18,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %20 = stablehlo.reshape %19 : (tensor<5x2xi1>) -> tensor<10xi1>
    %21 = stablehlo.convert %20 : (tensor<10xi1>) -> tensor<10xi32>
    %22 = stablehlo.reduce(%21 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %25 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %25 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %24 = stablehlo.add %16, %23 : tensor<1xi32>
    return %24 : tensor<1xi32>
  }
}
