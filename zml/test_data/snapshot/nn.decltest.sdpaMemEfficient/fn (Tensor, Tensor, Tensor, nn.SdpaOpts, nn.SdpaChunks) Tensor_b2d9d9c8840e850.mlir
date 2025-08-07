module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<512x512xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<-1.000000e+16> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %c_1 = stablehlo.constant dense<32> : tensor<i32>
    %c_2 = stablehlo.constant dense<16> : tensor<i32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_5 = stablehlo.constant dense<256> : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c_6, %c_6, %c_6, %c_6, sizes = [1, 10, 256, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x256x64xf32>
    %1 = stablehlo.dynamic_slice %arg3, %c_6, %c_6, sizes = [256, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x512xf32>
    %2 = stablehlo.dynamic_slice %arg0, %c_6, %c_6, %c_5, %c_6, sizes = [1, 10, 256, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x256x64xf32>
    %3 = stablehlo.dynamic_slice %arg3, %c_5, %c_6, sizes = [256, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x512xf32>
    %4 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x10x256x64xf32>
    %5 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x10x256x1xf32>
    %6 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x10x256x1xf32>
    %7:4 = stablehlo.while(%iterArg = %4, %iterArg_7 = %5, %iterArg_8 = %6, %iterArg_9 = %c_6) : tensor<1x10x256x64xf32>, tensor<1x10x256x1xf32>, tensor<1x10x256x1xf32>, tensor<i32> attributes {operandSegmentSizes = dense<4> : tensor<1xi32>}
    cond {
      %14 = stablehlo.compare  LT, %iterArg_9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %14 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    } do {
      %14 = stablehlo.multiply %iterArg_9, %c_1 : tensor<i32>
      %15 = stablehlo.dynamic_slice %arg1, %c_6, %c_6, %14, %c_6, sizes = [1, 10, 32, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x32x64xf32>
      %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x10x32x64xf32>
      %17 = stablehlo.multiply %15, %16 : tensor<1x10x32x64xf32>
      %18 = stablehlo.dot_general %0, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x32x64xf32>) -> tensor<1x10x256x32xf32>
      %19 = stablehlo.dynamic_slice %1, %c_6, %14, sizes = [256, 32] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x32xf32>
      %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<256x32xf32>) -> tensor<1x10x256x32xf32>
      %21 = stablehlo.add %18, %20 : tensor<1x10x256x32xf32>
      %22 = stablehlo.reduce(%21 init: %cst_3) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x32xf32>, tensor<f32>) -> tensor<1x10x256xf32>
       reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
        %47 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
        stablehlo.return %47 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
      }
      %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
      %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x10x256x1xf32>
      %25 = stablehlo.maximum %23, %24 : tensor<1x10x256x1xf32>
      %26 = stablehlo.maximum %iterArg_8, %25 : tensor<1x10x256x1xf32>
      %27 = stablehlo.subtract %iterArg_8, %26 : tensor<1x10x256x1xf32>
      %28 = stablehlo.exponential %27 : tensor<1x10x256x1xf32>
      %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
      %30 = stablehlo.multiply %iterArg, %29 : tensor<1x10x256x64xf32>
      %31 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x32xf32>
      %32 = stablehlo.subtract %21, %31 : tensor<1x10x256x32xf32>
      %33 = stablehlo.exponential %32 : tensor<1x10x256x32xf32>
      %34 = stablehlo.dynamic_slice %arg2, %c_6, %c_6, %14, %c_6, sizes = [1, 10, 32, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x32x64xf32>
      %35 = stablehlo.dot_general %33, %34, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x32xf32>, tensor<1x10x32x64xf32>) -> tensor<1x10x256x64xf32>
      %36 = stablehlo.subtract %25, %26 : tensor<1x10x256x1xf32>
      %37 = stablehlo.exponential %36 : tensor<1x10x256x1xf32>
      %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
      %39 = stablehlo.multiply %35, %38 : tensor<1x10x256x64xf32>
      %40 = stablehlo.add %30, %39 : tensor<1x10x256x64xf32>
      %41 = stablehlo.multiply %iterArg_7, %28 : tensor<1x10x256x1xf32>
      %42 = stablehlo.reduce(%33 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x32xf32>, tensor<f32>) -> tensor<1x10x256xf32>
       reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
        %47 = stablehlo.add %arg5, %arg4 : tensor<f32>
        stablehlo.return %47 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
      }
      %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
      %44 = stablehlo.multiply %43, %37 : tensor<1x10x256x1xf32>
      %45 = stablehlo.add %41, %44 : tensor<1x10x256x1xf32>
      %46 = stablehlo.add %iterArg_9, %c : tensor<i32>
      stablehlo.return %40, %45, %26, %46 {operandSegmentSizes = dense<4> : tensor<1xi32>} : tensor<1x10x256x64xf32>, tensor<1x10x256x1xf32>, tensor<1x10x256x1xf32>, tensor<i32>
    }
    %8 = stablehlo.broadcast_in_dim %7#1, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %9 = stablehlo.divide %7#0, %8 : tensor<1x10x256x64xf32>
    %10:4 = stablehlo.while(%iterArg = %4, %iterArg_7 = %5, %iterArg_8 = %6, %iterArg_9 = %c_6) : tensor<1x10x256x64xf32>, tensor<1x10x256x1xf32>, tensor<1x10x256x1xf32>, tensor<i32> attributes {operandSegmentSizes = dense<4> : tensor<1xi32>}
    cond {
      %14 = stablehlo.compare  LT, %iterArg_9, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %14 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    } do {
      %14 = stablehlo.multiply %iterArg_9, %c_1 : tensor<i32>
      %15 = stablehlo.dynamic_slice %arg1, %c_6, %c_6, %14, %c_6, sizes = [1, 10, 32, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x32x64xf32>
      %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x10x32x64xf32>
      %17 = stablehlo.multiply %15, %16 : tensor<1x10x32x64xf32>
      %18 = stablehlo.dot_general %2, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x32x64xf32>) -> tensor<1x10x256x32xf32>
      %19 = stablehlo.dynamic_slice %3, %c_6, %14, sizes = [256, 32] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x32xf32>
      %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<256x32xf32>) -> tensor<1x10x256x32xf32>
      %21 = stablehlo.add %18, %20 : tensor<1x10x256x32xf32>
      %22 = stablehlo.reduce(%21 init: %cst_3) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x32xf32>, tensor<f32>) -> tensor<1x10x256xf32>
       reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
        %47 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
        stablehlo.return %47 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
      }
      %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
      %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x10x256x1xf32>
      %25 = stablehlo.maximum %23, %24 : tensor<1x10x256x1xf32>
      %26 = stablehlo.maximum %iterArg_8, %25 : tensor<1x10x256x1xf32>
      %27 = stablehlo.subtract %iterArg_8, %26 : tensor<1x10x256x1xf32>
      %28 = stablehlo.exponential %27 : tensor<1x10x256x1xf32>
      %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
      %30 = stablehlo.multiply %iterArg, %29 : tensor<1x10x256x64xf32>
      %31 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x32xf32>
      %32 = stablehlo.subtract %21, %31 : tensor<1x10x256x32xf32>
      %33 = stablehlo.exponential %32 : tensor<1x10x256x32xf32>
      %34 = stablehlo.dynamic_slice %arg2, %c_6, %c_6, %14, %c_6, sizes = [1, 10, 32, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x32x64xf32>
      %35 = stablehlo.dot_general %33, %34, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x32xf32>, tensor<1x10x32x64xf32>) -> tensor<1x10x256x64xf32>
      %36 = stablehlo.subtract %25, %26 : tensor<1x10x256x1xf32>
      %37 = stablehlo.exponential %36 : tensor<1x10x256x1xf32>
      %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
      %39 = stablehlo.multiply %35, %38 : tensor<1x10x256x64xf32>
      %40 = stablehlo.add %30, %39 : tensor<1x10x256x64xf32>
      %41 = stablehlo.multiply %iterArg_7, %28 : tensor<1x10x256x1xf32>
      %42 = stablehlo.reduce(%33 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x32xf32>, tensor<f32>) -> tensor<1x10x256xf32>
       reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
        %47 = stablehlo.add %arg5, %arg4 : tensor<f32>
        stablehlo.return %47 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
      }
      %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
      %44 = stablehlo.multiply %43, %37 : tensor<1x10x256x1xf32>
      %45 = stablehlo.add %41, %44 : tensor<1x10x256x1xf32>
      %46 = stablehlo.add %iterArg_9, %c : tensor<i32>
      stablehlo.return %40, %45, %26, %46 {operandSegmentSizes = dense<4> : tensor<1xi32>} : tensor<1x10x256x64xf32>, tensor<1x10x256x1xf32>, tensor<1x10x256x1xf32>, tensor<i32>
    }
    %11 = stablehlo.broadcast_in_dim %10#1, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %12 = stablehlo.divide %10#0, %11 : tensor<1x10x256x64xf32>
    %13 = stablehlo.concatenate %9, %12, dim = 2 : (tensor<1x10x256x64xf32>, tensor<1x10x256x64xf32>) -> tensor<1x10x512x64xf32>
    return %13 : tensor<1x10x512x64xf32>
  }
}
