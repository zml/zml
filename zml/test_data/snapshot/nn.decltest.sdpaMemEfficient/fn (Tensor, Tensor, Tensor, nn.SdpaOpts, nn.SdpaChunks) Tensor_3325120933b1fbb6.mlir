module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<512x512xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
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
    %0 = stablehlo.dynamic_slice %arg0, %c_7, %c_7, %c_7, %c_7, sizes = [1, 10, 256, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x256x64xf32>
    %1 = stablehlo.multiply %c_7, %c_6 : tensor<i32>
    %2 = stablehlo.dynamic_slice %arg1, %c_7, %c_7, %1, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %3 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x10x128x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x10x128x64xf32>
    %5 = stablehlo.dot_general %0, %4, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %6 = stablehlo.dynamic_slice %arg3, %c_7, %c_7, sizes = [256, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x512xf32>
    %7 = stablehlo.dynamic_slice %6, %c_7, %1, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %9 = stablehlo.add %5, %8 : tensor<1x10x256x128xf32>
    %10 = stablehlo.reduce(%9 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x10x256x1xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x10x256x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %15 = stablehlo.subtract %9, %14 : tensor<1x10x256x128xf32>
    %16 = stablehlo.exponential %15 : tensor<1x10x256x128xf32>
    %17 = stablehlo.dynamic_slice %arg2, %c_7, %c_7, %1, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %19 = stablehlo.multiply %c_2, %c_6 : tensor<i32>
    %20 = stablehlo.dynamic_slice %arg1, %c_7, %c_7, %19, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %21 = stablehlo.multiply %20, %3 : tensor<1x10x128x64xf32>
    %22 = stablehlo.dot_general %0, %21, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %23 = stablehlo.dynamic_slice %6, %c_7, %19, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %25 = stablehlo.add %22, %24 : tensor<1x10x256x128xf32>
    %26 = stablehlo.reduce(%25 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %28 = stablehlo.maximum %27, %12 : tensor<1x10x256x1xf32>
    %29 = stablehlo.maximum %13, %28 : tensor<1x10x256x1xf32>
    %30 = stablehlo.subtract %13, %29 : tensor<1x10x256x1xf32>
    %31 = stablehlo.exponential %30 : tensor<1x10x256x1xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %33 = stablehlo.multiply %18, %32 : tensor<1x10x256x64xf32>
    %34 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %35 = stablehlo.subtract %25, %34 : tensor<1x10x256x128xf32>
    %36 = stablehlo.exponential %35 : tensor<1x10x256x128xf32>
    %37 = stablehlo.dynamic_slice %arg2, %c_7, %c_7, %19, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %38 = stablehlo.dot_general %36, %37, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %39 = stablehlo.subtract %28, %29 : tensor<1x10x256x1xf32>
    %40 = stablehlo.exponential %39 : tensor<1x10x256x1xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %42 = stablehlo.multiply %38, %41 : tensor<1x10x256x64xf32>
    %43 = stablehlo.add %33, %42 : tensor<1x10x256x64xf32>
    %44 = stablehlo.multiply %c_1, %c_6 : tensor<i32>
    %45 = stablehlo.dynamic_slice %arg1, %c_7, %c_7, %44, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %46 = stablehlo.multiply %45, %3 : tensor<1x10x128x64xf32>
    %47 = stablehlo.dot_general %0, %46, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %48 = stablehlo.dynamic_slice %6, %c_7, %44, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %50 = stablehlo.add %47, %49 : tensor<1x10x256x128xf32>
    %51 = stablehlo.reduce(%50 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %53 = stablehlo.maximum %52, %12 : tensor<1x10x256x1xf32>
    %54 = stablehlo.maximum %29, %53 : tensor<1x10x256x1xf32>
    %55 = stablehlo.subtract %29, %54 : tensor<1x10x256x1xf32>
    %56 = stablehlo.exponential %55 : tensor<1x10x256x1xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %58 = stablehlo.multiply %43, %57 : tensor<1x10x256x64xf32>
    %59 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %60 = stablehlo.subtract %50, %59 : tensor<1x10x256x128xf32>
    %61 = stablehlo.exponential %60 : tensor<1x10x256x128xf32>
    %62 = stablehlo.dynamic_slice %arg2, %c_7, %c_7, %44, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %63 = stablehlo.dot_general %61, %62, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %64 = stablehlo.subtract %53, %54 : tensor<1x10x256x1xf32>
    %65 = stablehlo.exponential %64 : tensor<1x10x256x1xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %67 = stablehlo.multiply %63, %66 : tensor<1x10x256x64xf32>
    %68 = stablehlo.add %58, %67 : tensor<1x10x256x64xf32>
    %69 = stablehlo.multiply %c_0, %c_6 : tensor<i32>
    %70 = stablehlo.dynamic_slice %arg1, %c_7, %c_7, %69, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %71 = stablehlo.multiply %70, %3 : tensor<1x10x128x64xf32>
    %72 = stablehlo.dot_general %0, %71, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %73 = stablehlo.dynamic_slice %6, %c_7, %69, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %75 = stablehlo.add %72, %74 : tensor<1x10x256x128xf32>
    %76 = stablehlo.reduce(%75 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %78 = stablehlo.maximum %77, %12 : tensor<1x10x256x1xf32>
    %79 = stablehlo.maximum %54, %78 : tensor<1x10x256x1xf32>
    %80 = stablehlo.subtract %54, %79 : tensor<1x10x256x1xf32>
    %81 = stablehlo.exponential %80 : tensor<1x10x256x1xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %83 = stablehlo.multiply %68, %82 : tensor<1x10x256x64xf32>
    %84 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %85 = stablehlo.subtract %75, %84 : tensor<1x10x256x128xf32>
    %86 = stablehlo.exponential %85 : tensor<1x10x256x128xf32>
    %87 = stablehlo.dynamic_slice %arg2, %c_7, %c_7, %69, %c_7, sizes = [1, 10, 128, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x128x64xf32>
    %88 = stablehlo.dot_general %86, %87, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %89 = stablehlo.subtract %78, %79 : tensor<1x10x256x1xf32>
    %90 = stablehlo.exponential %89 : tensor<1x10x256x1xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %92 = stablehlo.multiply %88, %91 : tensor<1x10x256x64xf32>
    %93 = stablehlo.add %83, %92 : tensor<1x10x256x64xf32>
    %94 = stablehlo.reduce(%16 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %96 = stablehlo.multiply %95, %31 : tensor<1x10x256x1xf32>
    %97 = stablehlo.reduce(%36 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %99 = stablehlo.multiply %98, %40 : tensor<1x10x256x1xf32>
    %100 = stablehlo.add %96, %99 : tensor<1x10x256x1xf32>
    %101 = stablehlo.multiply %100, %56 : tensor<1x10x256x1xf32>
    %102 = stablehlo.reduce(%61 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %104 = stablehlo.multiply %103, %65 : tensor<1x10x256x1xf32>
    %105 = stablehlo.add %101, %104 : tensor<1x10x256x1xf32>
    %106 = stablehlo.multiply %105, %81 : tensor<1x10x256x1xf32>
    %107 = stablehlo.reduce(%86 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %109 = stablehlo.multiply %108, %90 : tensor<1x10x256x1xf32>
    %110 = stablehlo.add %106, %109 : tensor<1x10x256x1xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %112 = stablehlo.divide %93, %111 : tensor<1x10x256x64xf32>
    %113 = stablehlo.dynamic_slice %arg0, %c_7, %c_7, %c, %c_7, sizes = [1, 10, 256, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x10x512x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x256x64xf32>
    %114 = stablehlo.dot_general %113, %4, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %115 = stablehlo.dynamic_slice %arg3, %c, %c_7, sizes = [256, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x512xf32>
    %116 = stablehlo.dynamic_slice %115, %c_7, %1, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %118 = stablehlo.add %114, %117 : tensor<1x10x256x128xf32>
    %119 = stablehlo.reduce(%118 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %120 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %121 = stablehlo.maximum %120, %12 : tensor<1x10x256x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %123 = stablehlo.subtract %118, %122 : tensor<1x10x256x128xf32>
    %124 = stablehlo.exponential %123 : tensor<1x10x256x128xf32>
    %125 = stablehlo.dot_general %124, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %126 = stablehlo.dot_general %113, %21, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %127 = stablehlo.dynamic_slice %115, %c_7, %19, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %129 = stablehlo.add %126, %128 : tensor<1x10x256x128xf32>
    %130 = stablehlo.reduce(%129 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %132 = stablehlo.maximum %131, %12 : tensor<1x10x256x1xf32>
    %133 = stablehlo.maximum %121, %132 : tensor<1x10x256x1xf32>
    %134 = stablehlo.subtract %121, %133 : tensor<1x10x256x1xf32>
    %135 = stablehlo.exponential %134 : tensor<1x10x256x1xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %137 = stablehlo.multiply %125, %136 : tensor<1x10x256x64xf32>
    %138 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %139 = stablehlo.subtract %129, %138 : tensor<1x10x256x128xf32>
    %140 = stablehlo.exponential %139 : tensor<1x10x256x128xf32>
    %141 = stablehlo.dot_general %140, %37, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %142 = stablehlo.subtract %132, %133 : tensor<1x10x256x1xf32>
    %143 = stablehlo.exponential %142 : tensor<1x10x256x1xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %145 = stablehlo.multiply %141, %144 : tensor<1x10x256x64xf32>
    %146 = stablehlo.add %137, %145 : tensor<1x10x256x64xf32>
    %147 = stablehlo.dot_general %113, %46, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %148 = stablehlo.dynamic_slice %115, %c_7, %44, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %150 = stablehlo.add %147, %149 : tensor<1x10x256x128xf32>
    %151 = stablehlo.reduce(%150 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %153 = stablehlo.maximum %152, %12 : tensor<1x10x256x1xf32>
    %154 = stablehlo.maximum %133, %153 : tensor<1x10x256x1xf32>
    %155 = stablehlo.subtract %133, %154 : tensor<1x10x256x1xf32>
    %156 = stablehlo.exponential %155 : tensor<1x10x256x1xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %158 = stablehlo.multiply %146, %157 : tensor<1x10x256x64xf32>
    %159 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %160 = stablehlo.subtract %150, %159 : tensor<1x10x256x128xf32>
    %161 = stablehlo.exponential %160 : tensor<1x10x256x128xf32>
    %162 = stablehlo.dot_general %161, %62, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %163 = stablehlo.subtract %153, %154 : tensor<1x10x256x1xf32>
    %164 = stablehlo.exponential %163 : tensor<1x10x256x1xf32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %166 = stablehlo.multiply %162, %165 : tensor<1x10x256x64xf32>
    %167 = stablehlo.add %158, %166 : tensor<1x10x256x64xf32>
    %168 = stablehlo.dot_general %113, %71, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x64xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x128xf32>
    %169 = stablehlo.dynamic_slice %115, %c_7, %69, sizes = [256, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<256x512xf32>, tensor<i32>, tensor<i32>) -> tensor<256x128xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [2, 3] : (tensor<256x128xf32>) -> tensor<1x10x256x128xf32>
    %171 = stablehlo.add %168, %170 : tensor<1x10x256x128xf32>
    %172 = stablehlo.reduce(%171 init: %cst_4) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %174 = stablehlo.maximum %173, %12 : tensor<1x10x256x1xf32>
    %175 = stablehlo.maximum %154, %174 : tensor<1x10x256x1xf32>
    %176 = stablehlo.subtract %154, %175 : tensor<1x10x256x1xf32>
    %177 = stablehlo.exponential %176 : tensor<1x10x256x1xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %179 = stablehlo.multiply %167, %178 : tensor<1x10x256x64xf32>
    %180 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x128xf32>
    %181 = stablehlo.subtract %171, %180 : tensor<1x10x256x128xf32>
    %182 = stablehlo.exponential %181 : tensor<1x10x256x128xf32>
    %183 = stablehlo.dot_general %182, %87, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x256x128xf32>, tensor<1x10x128x64xf32>) -> tensor<1x10x256x64xf32>
    %184 = stablehlo.subtract %174, %175 : tensor<1x10x256x1xf32>
    %185 = stablehlo.exponential %184 : tensor<1x10x256x1xf32>
    %186 = stablehlo.broadcast_in_dim %185, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %187 = stablehlo.multiply %183, %186 : tensor<1x10x256x64xf32>
    %188 = stablehlo.add %179, %187 : tensor<1x10x256x64xf32>
    %189 = stablehlo.reduce(%124 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %190 = stablehlo.broadcast_in_dim %189, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %191 = stablehlo.multiply %190, %135 : tensor<1x10x256x1xf32>
    %192 = stablehlo.reduce(%140 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %194 = stablehlo.multiply %193, %143 : tensor<1x10x256x1xf32>
    %195 = stablehlo.add %191, %194 : tensor<1x10x256x1xf32>
    %196 = stablehlo.multiply %195, %156 : tensor<1x10x256x1xf32>
    %197 = stablehlo.reduce(%161 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %199 = stablehlo.multiply %198, %164 : tensor<1x10x256x1xf32>
    %200 = stablehlo.add %196, %199 : tensor<1x10x256x1xf32>
    %201 = stablehlo.multiply %200, %177 : tensor<1x10x256x1xf32>
    %202 = stablehlo.reduce(%182 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x256x128xf32>, tensor<f32>) -> tensor<1x10x256xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %209 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %209 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2] : (tensor<1x10x256xf32>) -> tensor<1x10x256x1xf32>
    %204 = stablehlo.multiply %203, %185 : tensor<1x10x256x1xf32>
    %205 = stablehlo.add %201, %204 : tensor<1x10x256x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<1x10x256x1xf32>) -> tensor<1x10x256x64xf32>
    %207 = stablehlo.divide %188, %206 : tensor<1x10x256x64xf32>
    %208 = stablehlo.concatenate %112, %207, dim = 2 : (tensor<1x10x256x64xf32>, tensor<1x10x256x64xf32>) -> tensor<1x10x512x64xf32>
    return %208 : tensor<1x10x512x64xf32>
  }
}
