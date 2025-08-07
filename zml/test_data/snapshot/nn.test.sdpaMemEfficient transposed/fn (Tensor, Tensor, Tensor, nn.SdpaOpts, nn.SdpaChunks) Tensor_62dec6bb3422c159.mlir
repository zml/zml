module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<512x512xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<256> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %cst_3 = stablehlo.constant dense<-1.000000e+16> : tensor<f32>
    %cst_4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %c_6 = stablehlo.constant dense<128> : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c_7, %c_7, %c_7, %c_7, sizes = [1, 256, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x256x10x64xf32>
    %1 = stablehlo.multiply %c_7, %c_6 : tensor<i32>
    %2 = stablehlo.dynamic_slice %arg1, %c_7, %1, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %3 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x128x10x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x128x10x64xf32>
    %5 = stablehlo.dot_general %0, %4, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %6 = stablehlo.dynamic_slice %arg3, %c_7, %c_7, sizes = [256, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x512xf32>
    %7 = stablehlo.dynamic_slice %6, %c_7, %1, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %9 = stablehlo.add %5, %8 : tensor<1x10x256x128xf32>
    %10 = stablehlo.reduce(%9 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x10x256x1xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x10x256x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %15 = stablehlo.subtract %9, %14 : tensor<1x10x256x128xf32>
    %16 = stablehlo.exponential %15 : tensor<1x10x256x128xf32>
    %17 = stablehlo.dynamic_slice %arg2, %c_7, %1, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %19 = stablehlo.transpose %18, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %20 = stablehlo.transpose %13, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %21 = stablehlo.multiply %c_2, %c_6 : tensor<i32>
    %22 = stablehlo.dynamic_slice %arg1, %c_7, %21, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %23 = stablehlo.multiply %22, %3 : tensor<1x128x10x64xf32>
    %24 = stablehlo.dot_general %0, %23, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %25 = stablehlo.dynamic_slice %6, %c_7, %21, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %27 = stablehlo.add %24, %26 : tensor<1x10x256x128xf32>
    %28 = stablehlo.reduce(%27 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %30 = stablehlo.maximum %29, %12 : tensor<1x10x256x1xf32>
    %31 = stablehlo.transpose %30, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %32 = stablehlo.maximum %20, %31 : tensor<1x256x10x1xf32>
    %33 = stablehlo.subtract %20, %32 : tensor<1x256x10x1xf32>
    %34 = stablehlo.exponential %33 : tensor<1x256x10x1xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %36 = stablehlo.multiply %19, %35 : tensor<1x256x10x64xf32>
    %37 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %38 = stablehlo.subtract %27, %37 : tensor<1x10x256x128xf32>
    %39 = stablehlo.exponential %38 : tensor<1x10x256x128xf32>
    %40 = stablehlo.dynamic_slice %arg2, %c_7, %21, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %41 = stablehlo.dot_general %39, %40, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %42 = stablehlo.transpose %41, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %43 = stablehlo.subtract %31, %32 : tensor<1x256x10x1xf32>
    %44 = stablehlo.exponential %43 : tensor<1x256x10x1xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %46 = stablehlo.multiply %42, %45 : tensor<1x256x10x64xf32>
    %47 = stablehlo.add %36, %46 : tensor<1x256x10x64xf32>
    %48 = stablehlo.multiply %c_1, %c_6 : tensor<i32>
    %49 = stablehlo.dynamic_slice %arg1, %c_7, %48, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %50 = stablehlo.multiply %49, %3 : tensor<1x128x10x64xf32>
    %51 = stablehlo.dot_general %0, %50, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %52 = stablehlo.dynamic_slice %6, %c_7, %48, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %54 = stablehlo.add %51, %53 : tensor<1x10x256x128xf32>
    %55 = stablehlo.reduce(%54 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %57 = stablehlo.maximum %56, %12 : tensor<1x10x256x1xf32>
    %58 = stablehlo.transpose %57, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %59 = stablehlo.maximum %32, %58 : tensor<1x256x10x1xf32>
    %60 = stablehlo.subtract %32, %59 : tensor<1x256x10x1xf32>
    %61 = stablehlo.exponential %60 : tensor<1x256x10x1xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %63 = stablehlo.multiply %47, %62 : tensor<1x256x10x64xf32>
    %64 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %65 = stablehlo.subtract %54, %64 : tensor<1x10x256x128xf32>
    %66 = stablehlo.exponential %65 : tensor<1x10x256x128xf32>
    %67 = stablehlo.dynamic_slice %arg2, %c_7, %48, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %68 = stablehlo.dot_general %66, %67, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %69 = stablehlo.transpose %68, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %70 = stablehlo.subtract %58, %59 : tensor<1x256x10x1xf32>
    %71 = stablehlo.exponential %70 : tensor<1x256x10x1xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %73 = stablehlo.multiply %69, %72 : tensor<1x256x10x64xf32>
    %74 = stablehlo.add %63, %73 : tensor<1x256x10x64xf32>
    %75 = stablehlo.multiply %c_0, %c_6 : tensor<i32>
    %76 = stablehlo.dynamic_slice %arg1, %c_7, %75, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %77 = stablehlo.multiply %76, %3 : tensor<1x128x10x64xf32>
    %78 = stablehlo.dot_general %0, %77, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %79 = stablehlo.dynamic_slice %6, %c_7, %75, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %81 = stablehlo.add %78, %80 : tensor<1x10x256x128xf32>
    %82 = stablehlo.reduce(%81 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %83 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %84 = stablehlo.maximum %83, %12 : tensor<1x10x256x1xf32>
    %85 = stablehlo.transpose %84, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %86 = stablehlo.maximum %59, %85 : tensor<1x256x10x1xf32>
    %87 = stablehlo.subtract %59, %86 : tensor<1x256x10x1xf32>
    %88 = stablehlo.exponential %87 : tensor<1x256x10x1xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %90 = stablehlo.multiply %74, %89 : tensor<1x256x10x64xf32>
    %91 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %92 = stablehlo.subtract %81, %91 : tensor<1x10x256x128xf32>
    %93 = stablehlo.exponential %92 : tensor<1x10x256x128xf32>
    %94 = stablehlo.dynamic_slice %arg2, %c_7, %75, %c_7, %c_7, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %95 = stablehlo.dot_general %93, %94, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %96 = stablehlo.transpose %95, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %97 = stablehlo.subtract %85, %86 : tensor<1x256x10x1xf32>
    %98 = stablehlo.exponential %97 : tensor<1x256x10x1xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %100 = stablehlo.multiply %96, %99 : tensor<1x256x10x64xf32>
    %101 = stablehlo.add %90, %100 : tensor<1x256x10x64xf32>
    %102 = stablehlo.reduce(%16 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %104 = stablehlo.transpose %103, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %105 = stablehlo.multiply %104, %34 : tensor<1x256x10x1xf32>
    %106 = stablehlo.reduce(%39 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %108 = stablehlo.transpose %107, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %109 = stablehlo.multiply %108, %44 : tensor<1x256x10x1xf32>
    %110 = stablehlo.add %105, %109 : tensor<1x256x10x1xf32>
    %111 = stablehlo.multiply %110, %61 : tensor<1x256x10x1xf32>
    %112 = stablehlo.reduce(%66 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %114 = stablehlo.transpose %113, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %115 = stablehlo.multiply %114, %71 : tensor<1x256x10x1xf32>
    %116 = stablehlo.add %111, %115 : tensor<1x256x10x1xf32>
    %117 = stablehlo.multiply %116, %88 : tensor<1x256x10x1xf32>
    %118 = stablehlo.reduce(%93 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %120 = stablehlo.transpose %119, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %121 = stablehlo.multiply %120, %98 : tensor<1x256x10x1xf32>
    %122 = stablehlo.add %117, %121 : tensor<1x256x10x1xf32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %124 = stablehlo.divide %101, %123 : tensor<1x256x10x64xf32>
    %125 = stablehlo.dynamic_slice %arg0, %c_7, %c, %c_7, %c_7, sizes = [1, 256, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x256x10x64xf32>
    %126 = stablehlo.dot_general %125, %4, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %127 = stablehlo.dynamic_slice %arg3, %c, %c_7, sizes = [256, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x512xf32>
    %128 = stablehlo.dynamic_slice %127, %c_7, %1, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %130 = stablehlo.add %126, %129 : tensor<1x10x256x128xf32>
    %131 = stablehlo.reduce(%130 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %133 = stablehlo.maximum %132, %12 : tensor<1x10x256x1xf32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %135 = stablehlo.subtract %130, %134 : tensor<1x10x256x128xf32>
    %136 = stablehlo.exponential %135 : tensor<1x10x256x128xf32>
    %137 = stablehlo.dot_general %136, %17, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %138 = stablehlo.transpose %137, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %139 = stablehlo.transpose %133, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %140 = stablehlo.dot_general %125, %23, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %141 = stablehlo.dynamic_slice %127, %c_7, %21, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %143 = stablehlo.add %140, %142 : tensor<1x10x256x128xf32>
    %144 = stablehlo.reduce(%143 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %146 = stablehlo.maximum %145, %12 : tensor<1x10x256x1xf32>
    %147 = stablehlo.transpose %146, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %148 = stablehlo.maximum %139, %147 : tensor<1x256x10x1xf32>
    %149 = stablehlo.subtract %139, %148 : tensor<1x256x10x1xf32>
    %150 = stablehlo.exponential %149 : tensor<1x256x10x1xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %152 = stablehlo.multiply %138, %151 : tensor<1x256x10x64xf32>
    %153 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %154 = stablehlo.subtract %143, %153 : tensor<1x10x256x128xf32>
    %155 = stablehlo.exponential %154 : tensor<1x10x256x128xf32>
    %156 = stablehlo.dot_general %155, %40, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %157 = stablehlo.transpose %156, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %158 = stablehlo.subtract %147, %148 : tensor<1x256x10x1xf32>
    %159 = stablehlo.exponential %158 : tensor<1x256x10x1xf32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %161 = stablehlo.multiply %157, %160 : tensor<1x256x10x64xf32>
    %162 = stablehlo.add %152, %161 : tensor<1x256x10x64xf32>
    %163 = stablehlo.dot_general %125, %50, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %164 = stablehlo.dynamic_slice %127, %c_7, %48, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %166 = stablehlo.add %163, %165 : tensor<1x10x256x128xf32>
    %167 = stablehlo.reduce(%166 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %169 = stablehlo.maximum %168, %12 : tensor<1x10x256x1xf32>
    %170 = stablehlo.transpose %169, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %171 = stablehlo.maximum %148, %170 : tensor<1x256x10x1xf32>
    %172 = stablehlo.subtract %148, %171 : tensor<1x256x10x1xf32>
    %173 = stablehlo.exponential %172 : tensor<1x256x10x1xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %175 = stablehlo.multiply %162, %174 : tensor<1x256x10x64xf32>
    %176 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %177 = stablehlo.subtract %166, %176 : tensor<1x10x256x128xf32>
    %178 = stablehlo.exponential %177 : tensor<1x10x256x128xf32>
    %179 = stablehlo.dot_general %178, %67, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %180 = stablehlo.transpose %179, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %181 = stablehlo.subtract %170, %171 : tensor<1x256x10x1xf32>
    %182 = stablehlo.exponential %181 : tensor<1x256x10x1xf32>
    %183 = stablehlo.broadcast_in_dim %182, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %184 = stablehlo.multiply %180, %183 : tensor<1x256x10x64xf32>
    %185 = stablehlo.add %175, %184 : tensor<1x256x10x64xf32>
    %186 = stablehlo.dot_general %125, %77, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x256x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x128xf32>
    %187 = stablehlo.dynamic_slice %127, %c_7, %75, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %189 = stablehlo.add %186, %188 : tensor<1x10x256x128xf32>
    %190 = stablehlo.reduce(%189 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %192 = stablehlo.maximum %191, %12 : tensor<1x10x256x1xf32>
    %193 = stablehlo.transpose %192, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %194 = stablehlo.maximum %171, %193 : tensor<1x256x10x1xf32>
    %195 = stablehlo.subtract %171, %194 : tensor<1x256x10x1xf32>
    %196 = stablehlo.exponential %195 : tensor<1x256x10x1xf32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %198 = stablehlo.multiply %185, %197 : tensor<1x256x10x64xf32>
    %199 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %200 = stablehlo.subtract %189, %199 : tensor<1x10x256x128xf32>
    %201 = stablehlo.exponential %200 : tensor<1x10x256x128xf32>
    %202 = stablehlo.dot_general %201, %94, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x256x64xf32>
    %203 = stablehlo.transpose %202, dims = [0, 2, 1, 3] : (tensor<1x10x256x64xf32>) -> tensor<1x256x10x64xf32>
    %204 = stablehlo.subtract %193, %194 : tensor<1x256x10x1xf32>
    %205 = stablehlo.exponential %204 : tensor<1x256x10x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %207 = stablehlo.multiply %203, %206 : tensor<1x256x10x64xf32>
    %208 = stablehlo.add %198, %207 : tensor<1x256x10x64xf32>
    %209 = stablehlo.reduce(%136 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %211 = stablehlo.transpose %210, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %212 = stablehlo.multiply %211, %150 : tensor<1x256x10x1xf32>
    %213 = stablehlo.reduce(%155 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %215 = stablehlo.transpose %214, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %216 = stablehlo.multiply %215, %159 : tensor<1x256x10x1xf32>
    %217 = stablehlo.add %212, %216 : tensor<1x256x10x1xf32>
    %218 = stablehlo.multiply %217, %173 : tensor<1x256x10x1xf32>
    %219 = stablehlo.reduce(%178 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %220 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %221 = stablehlo.transpose %220, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %222 = stablehlo.multiply %221, %182 : tensor<1x256x10x1xf32>
    %223 = stablehlo.add %218, %222 : tensor<1x256x10x1xf32>
    %224 = stablehlo.multiply %223, %196 : tensor<1x256x10x1xf32>
    %225 = stablehlo.reduce(%201 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %233 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %233 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %227 = stablehlo.transpose %226, dims = [0, 2, 1, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x256x10x1xf32>
    %228 = stablehlo.multiply %227, %205 : tensor<1x256x10x1xf32>
    %229 = stablehlo.add %224, %228 : tensor<1x256x10x1xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x256x10x1xf32>) -> tensor<1x256x10x64xf32>
    %231 = stablehlo.divide %208, %230 : tensor<1x256x10x64xf32>
    %232 = stablehlo.concatenate %124, %231, dim = 1 : (tensor<1x256x10x64xf32>, tensor<1x256x10x64xf32>) -> tensor<1x512x10x64xf32>
    return %232 : tensor<1x512x10x64xf32>
  }
}
