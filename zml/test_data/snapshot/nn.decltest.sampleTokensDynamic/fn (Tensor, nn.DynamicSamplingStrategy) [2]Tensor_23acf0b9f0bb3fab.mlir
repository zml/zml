module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<4xbf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<bf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<bf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg4: tensor<bf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<4xbf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<4xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.compare  GE, %0, %1,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<4xbf16>
    %4:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg5: tensor<bf16>, %arg6: tensor<bf16>, %arg7: tensor<i32>, %arg8: tensor<i32>):
      %36 = stablehlo.compare  GT, %arg5, %arg6,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %36 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    }) {operandSegmentSizes = dense<2> : tensor<1xi32>} : (tensor<4xbf16>, tensor<4xi32>) -> (tensor<4xbf16>, tensor<4xi32>)
    %5 = stablehlo.select %2, %3, %4#0 : tensor<4xi1>, tensor<4xbf16>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<4xbf16>
    %7 = stablehlo.multiply %5, %6 : tensor<4xbf16>
    %8 = stablehlo.reduce(%7 init: %cst_0) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<4xbf16>, tensor<bf16>) -> tensor<bf16>
     reducer(%arg5: tensor<bf16>, %arg6: tensor<bf16>)  {
      %36 = stablehlo.maximum %arg6, %arg5 : tensor<bf16>
      stablehlo.return %36 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<bf16>
    }
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %11 = stablehlo.compare  GT, %9, %10,  FLOAT : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xi1>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi1>) -> tensor<4xi1>
    %13 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xbf16>) -> tensor<4xbf16>
    %14 = stablehlo.subtract %7, %13 : tensor<4xbf16>
    %15 = stablehlo.exponential %14 : tensor<4xbf16>
    %16 = stablehlo.reduce(%15 init: %cst) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<4xbf16>, tensor<bf16>) -> tensor<bf16>
     reducer(%arg5: tensor<bf16>, %arg6: tensor<bf16>)  {
      %36 = stablehlo.add %arg6, %arg5 : tensor<bf16>
      stablehlo.return %36 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<bf16>
    }
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<1xbf16>) -> tensor<4xbf16>
    %19 = stablehlo.divide %15, %18 : tensor<4xbf16>
    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<4xbf16>
    %21 = stablehlo.select %12, %19, %20 : tensor<4xi1>, tensor<4xbf16>
    %22 = "stablehlo.reduce_window"(%21, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[3, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 4>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg5: tensor<bf16>, %arg6: tensor<bf16>):
      %36 = stablehlo.add %arg5, %arg6 : tensor<bf16>
      stablehlo.return %36 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<bf16>
    }) {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<4xbf16>, tensor<bf16>) -> tensor<4xbf16>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<bf16>) -> tensor<4xbf16>
    %24 = stablehlo.compare  LE, %22, %23,  FLOAT : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xi1>
    %25 = stablehlo.slice %21 [0:1] : (tensor<4xbf16>) -> tensor<1xbf16>
    %26 = stablehlo.reshape %25 : (tensor<1xbf16>) -> tensor<1xbf16>
    %27 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %28 = stablehlo.multiply %26, %27 : tensor<1xbf16>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0] : (tensor<1xbf16>) -> tensor<4xbf16>
    %30 = stablehlo.compare  GE, %21, %29,  FLOAT : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xi1>
    %31 = stablehlo.and %24, %30 : tensor<4xi1>
    %32 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %33 = stablehlo.compare  EQ, %0, %32,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %34 = stablehlo.or %31, %33 : tensor<4xi1>
    %35 = stablehlo.select %34, %7, %3 : tensor<4xi1>, tensor<4xbf16>
    return %35, %4#1 : tensor<4xbf16>, tensor<4xi32>
  }
}
