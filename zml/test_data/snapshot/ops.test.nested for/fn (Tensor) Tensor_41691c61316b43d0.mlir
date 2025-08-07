module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<5xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<5x5x1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<5> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c_1, sizes = [1] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<5xf32>, tensor<i32>) -> tensor<1xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3 = stablehlo.pad %2, %cst, low = [0, 0], high = [4, 0], interior = [0, 0] : (tensor<1x1xf32>, tensor<f32>) -> tensor<5x1xf32>
    %4:2 = stablehlo.while(%iterArg = %3, %iterArg_2 = %c_0) : tensor<5x1xf32>, tensor<i32> attributes {operandSegmentSizes = dense<2> : tensor<1xi32>}
    cond {
      %8 = stablehlo.compare  LT, %iterArg_2, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %8 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    } do {
      %8 = stablehlo.dynamic_slice %arg0, %iterArg_2, sizes = [1] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<5xf32>, tensor<i32>) -> tensor<1xf32>
      %9 = stablehlo.multiply %0, %8 : tensor<1xf32>
      %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<1x1xf32>
      %11 = stablehlo.dynamic_update_slice %iterArg, %10, %iterArg_2, %c_1 {operandSegmentSizes = dense<[1, 1, 2]> : tensor<3xi32>} : (tensor<5x1xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<5x1xf32>
      %12 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
      stablehlo.return %11, %12 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<5x1xf32>, tensor<i32>
    }
    %5 = stablehlo.reshape %4#0 : (tensor<5x1xf32>) -> tensor<1x5x1xf32>
    %6 = stablehlo.pad %5, %cst, low = [0, 0, 0], high = [4, 0, 0], interior = [0, 0, 0] : (tensor<1x5x1xf32>, tensor<f32>) -> tensor<5x5x1xf32>
    %7:2 = stablehlo.while(%iterArg = %6, %iterArg_2 = %c_0) : tensor<5x5x1xf32>, tensor<i32> attributes {operandSegmentSizes = dense<2> : tensor<1xi32>}
    cond {
      %8 = stablehlo.compare  LT, %iterArg_2, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %8 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    } do {
      %8 = stablehlo.dynamic_slice %arg0, %iterArg_2, sizes = [1] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<5xf32>, tensor<i32>) -> tensor<1xf32>
      %9 = stablehlo.multiply %8, %0 : tensor<1xf32>
      %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<1x1xf32>
      %11 = stablehlo.pad %10, %cst, low = [0, 0], high = [4, 0], interior = [0, 0] : (tensor<1x1xf32>, tensor<f32>) -> tensor<5x1xf32>
      %12:2 = stablehlo.while(%iterArg_3 = %11, %iterArg_4 = %c_0) : tensor<5x1xf32>, tensor<i32> attributes {operandSegmentSizes = dense<2> : tensor<1xi32>}
      cond {
        %16 = stablehlo.compare  LT, %iterArg_4, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %16 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
      } do {
        %16 = stablehlo.dynamic_slice %arg0, %iterArg_4, sizes = [1] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<5xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.multiply %8, %16 : tensor<1xf32>
        %18 = stablehlo.reshape %17 : (tensor<1xf32>) -> tensor<1x1xf32>
        %19 = stablehlo.dynamic_update_slice %iterArg_3, %18, %iterArg_4, %c_1 {operandSegmentSizes = dense<[1, 1, 2]> : tensor<3xi32>} : (tensor<5x1xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<5x1xf32>
        %20 = stablehlo.add %iterArg_4, %c_0 : tensor<i32>
        stablehlo.return %19, %20 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<5x1xf32>, tensor<i32>
      }
      %13 = stablehlo.reshape %12#0 : (tensor<5x1xf32>) -> tensor<1x5x1xf32>
      %14 = stablehlo.dynamic_update_slice %iterArg, %13, %iterArg_2, %c_1, %c_1 {operandSegmentSizes = dense<[1, 1, 3]> : tensor<3xi32>} : (tensor<5x5x1xf32>, tensor<1x5x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x5x1xf32>
      %15 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
      stablehlo.return %14, %15 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<5x5x1xf32>, tensor<i32>
    }
    return %7#0 : tensor<5x5x1xf32>
  }
}
