module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main() -> (tensor<1xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.iota dim = 0 : tensor<20xf32>
    %1 = stablehlo.reshape %0 : (tensor<20xf32>) -> tensor<5x4xf32>
    %2 = stablehlo.slice %1 [0:5, 0:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %3 = stablehlo.reshape %2 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %4 = stablehlo.slice %1 [0:5, 2:4] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %5 = stablehlo.reshape %4 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %6 = stablehlo.concatenate %3, %5, dim = 1 : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x4xf32>
    %7 = stablehlo.slice %6 [0:5, 0:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %8 = stablehlo.reshape %7 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %9 = stablehlo.compare  EQ, %3, %8,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %10 = stablehlo.reshape %9 : (tensor<5x2xi1>) -> tensor<10xi1>
    %11 = stablehlo.convert %10 : (tensor<10xi1>) -> tensor<10xi32>
    %12 = stablehlo.reduce(%11 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %22 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %22 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.slice %6 [0:5, 2:4] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %15 = stablehlo.reshape %14 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %16 = stablehlo.compare  EQ, %5, %15,  FLOAT : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    %17 = stablehlo.reshape %16 : (tensor<5x2xi1>) -> tensor<10xi1>
    %18 = stablehlo.convert %17 : (tensor<10xi1>) -> tensor<10xi32>
    %19 = stablehlo.reduce(%18 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %22 = stablehlo.add %arg1, %arg0 : tensor<i32>
      stablehlo.return %22 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i32>
    }
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %21 = stablehlo.add %13, %20 : tensor<1xi32>
    return %21 : tensor<1xi32>
  }
}
