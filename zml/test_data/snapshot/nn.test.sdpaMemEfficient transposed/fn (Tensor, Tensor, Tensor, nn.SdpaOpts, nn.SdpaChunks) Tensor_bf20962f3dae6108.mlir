module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<512x512xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x512x10x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<3> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %cst_2 = stablehlo.constant dense<-1.000000e+16> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %c_5 = stablehlo.constant dense<128> : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c_6, %c_6, %c_6, %c_6, sizes = [1, 512, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x512x10x64xf32>
    %1 = stablehlo.multiply %c_6, %c_5 : tensor<i32>
    %2 = stablehlo.dynamic_slice %arg1, %c_6, %1, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %3 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x128x10x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x128x10x64xf32>
    %5 = stablehlo.dot_general %0, %4, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x512x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x128xf32>
    %6 = stablehlo.dynamic_slice %arg3, %c_6, %c_6, sizes = [512, 512] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<512x512xf32>
    %7 = stablehlo.dynamic_slice %6, %c_6, %1, sizes = [512, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<512x128xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2, 3] : (tensor<512x128xf32>) -> tensor<1x10x512x128xf32>
    %9 = stablehlo.add %5, %8 : tensor<1x10x512x128xf32>
    %10 = stablehlo.reduce(%9 init: %cst_3) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x10x512x1xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x10x512x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x10x512x128xf32>
    %15 = stablehlo.subtract %9, %14 : tensor<1x10x512x128xf32>
    %16 = stablehlo.exponential %15 : tensor<1x10x512x128xf32>
    %17 = stablehlo.dynamic_slice %arg2, %c_6, %1, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x512x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x64xf32>
    %19 = stablehlo.transpose %18, dims = [0, 2, 1, 3] : (tensor<1x10x512x64xf32>) -> tensor<1x512x10x64xf32>
    %20 = stablehlo.transpose %13, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %21 = stablehlo.multiply %c_1, %c_5 : tensor<i32>
    %22 = stablehlo.dynamic_slice %arg1, %c_6, %21, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %23 = stablehlo.multiply %22, %3 : tensor<1x128x10x64xf32>
    %24 = stablehlo.dot_general %0, %23, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x512x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x128xf32>
    %25 = stablehlo.dynamic_slice %6, %c_6, %21, sizes = [512, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<512x128xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [2, 3] : (tensor<512x128xf32>) -> tensor<1x10x512x128xf32>
    %27 = stablehlo.add %24, %26 : tensor<1x10x512x128xf32>
    %28 = stablehlo.reduce(%27 init: %cst_3) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %30 = stablehlo.maximum %29, %12 : tensor<1x10x512x1xf32>
    %31 = stablehlo.transpose %30, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %32 = stablehlo.maximum %20, %31 : tensor<1x512x10x1xf32>
    %33 = stablehlo.subtract %20, %32 : tensor<1x512x10x1xf32>
    %34 = stablehlo.exponential %33 : tensor<1x512x10x1xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %36 = stablehlo.multiply %19, %35 : tensor<1x512x10x64xf32>
    %37 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x10x512x128xf32>
    %38 = stablehlo.subtract %27, %37 : tensor<1x10x512x128xf32>
    %39 = stablehlo.exponential %38 : tensor<1x10x512x128xf32>
    %40 = stablehlo.dynamic_slice %arg2, %c_6, %21, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %41 = stablehlo.dot_general %39, %40, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x512x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x64xf32>
    %42 = stablehlo.transpose %41, dims = [0, 2, 1, 3] : (tensor<1x10x512x64xf32>) -> tensor<1x512x10x64xf32>
    %43 = stablehlo.subtract %31, %32 : tensor<1x512x10x1xf32>
    %44 = stablehlo.exponential %43 : tensor<1x512x10x1xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %46 = stablehlo.multiply %42, %45 : tensor<1x512x10x64xf32>
    %47 = stablehlo.add %36, %46 : tensor<1x512x10x64xf32>
    %48 = stablehlo.multiply %c_0, %c_5 : tensor<i32>
    %49 = stablehlo.dynamic_slice %arg1, %c_6, %48, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %50 = stablehlo.multiply %49, %3 : tensor<1x128x10x64xf32>
    %51 = stablehlo.dot_general %0, %50, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x512x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x128xf32>
    %52 = stablehlo.dynamic_slice %6, %c_6, %48, sizes = [512, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<512x128xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [2, 3] : (tensor<512x128xf32>) -> tensor<1x10x512x128xf32>
    %54 = stablehlo.add %51, %53 : tensor<1x10x512x128xf32>
    %55 = stablehlo.reduce(%54 init: %cst_3) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %57 = stablehlo.maximum %56, %12 : tensor<1x10x512x1xf32>
    %58 = stablehlo.transpose %57, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %59 = stablehlo.maximum %32, %58 : tensor<1x512x10x1xf32>
    %60 = stablehlo.subtract %32, %59 : tensor<1x512x10x1xf32>
    %61 = stablehlo.exponential %60 : tensor<1x512x10x1xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %63 = stablehlo.multiply %47, %62 : tensor<1x512x10x64xf32>
    %64 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x10x512x128xf32>
    %65 = stablehlo.subtract %54, %64 : tensor<1x10x512x128xf32>
    %66 = stablehlo.exponential %65 : tensor<1x10x512x128xf32>
    %67 = stablehlo.dynamic_slice %arg2, %c_6, %48, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %68 = stablehlo.dot_general %66, %67, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x512x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x64xf32>
    %69 = stablehlo.transpose %68, dims = [0, 2, 1, 3] : (tensor<1x10x512x64xf32>) -> tensor<1x512x10x64xf32>
    %70 = stablehlo.subtract %58, %59 : tensor<1x512x10x1xf32>
    %71 = stablehlo.exponential %70 : tensor<1x512x10x1xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %73 = stablehlo.multiply %69, %72 : tensor<1x512x10x64xf32>
    %74 = stablehlo.add %63, %73 : tensor<1x512x10x64xf32>
    %75 = stablehlo.multiply %c, %c_5 : tensor<i32>
    %76 = stablehlo.dynamic_slice %arg1, %c_6, %75, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %77 = stablehlo.multiply %76, %3 : tensor<1x128x10x64xf32>
    %78 = stablehlo.dot_general %0, %77, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x512x10x64xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x128xf32>
    %79 = stablehlo.dynamic_slice %6, %c_6, %75, sizes = [512, 128] {operandSegmentSizes = dense<[1, 2]> : tensor<2xi32>} : (tensor<512x512xf32>, tensor<i32>, tensor<i32>) -> tensor<512x128xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [2, 3] : (tensor<512x128xf32>) -> tensor<1x10x512x128xf32>
    %81 = stablehlo.add %78, %80 : tensor<1x10x512x128xf32>
    %82 = stablehlo.reduce(%81 init: %cst_3) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %83 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %84 = stablehlo.maximum %83, %12 : tensor<1x10x512x1xf32>
    %85 = stablehlo.transpose %84, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %86 = stablehlo.maximum %59, %85 : tensor<1x512x10x1xf32>
    %87 = stablehlo.subtract %59, %86 : tensor<1x512x10x1xf32>
    %88 = stablehlo.exponential %87 : tensor<1x512x10x1xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %90 = stablehlo.multiply %74, %89 : tensor<1x512x10x64xf32>
    %91 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x10x512x128xf32>
    %92 = stablehlo.subtract %81, %91 : tensor<1x10x512x128xf32>
    %93 = stablehlo.exponential %92 : tensor<1x10x512x128xf32>
    %94 = stablehlo.dynamic_slice %arg2, %c_6, %75, %c_6, %c_6, sizes = [1, 128, 10, 64] {operandSegmentSizes = dense<[1, 4]> : tensor<2xi32>} : (tensor<1x512x10x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x128x10x64xf32>
    %95 = stablehlo.dot_general %93, %94, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x10x512x128xf32>, tensor<1x128x10x64xf32>) -> tensor<1x10x512x64xf32>
    %96 = stablehlo.transpose %95, dims = [0, 2, 1, 3] : (tensor<1x10x512x64xf32>) -> tensor<1x512x10x64xf32>
    %97 = stablehlo.subtract %85, %86 : tensor<1x512x10x1xf32>
    %98 = stablehlo.exponential %97 : tensor<1x512x10x1xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %100 = stablehlo.multiply %96, %99 : tensor<1x512x10x64xf32>
    %101 = stablehlo.add %90, %100 : tensor<1x512x10x64xf32>
    %102 = stablehlo.reduce(%16 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %104 = stablehlo.transpose %103, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %105 = stablehlo.multiply %104, %34 : tensor<1x512x10x1xf32>
    %106 = stablehlo.reduce(%39 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %108 = stablehlo.transpose %107, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %109 = stablehlo.multiply %108, %44 : tensor<1x512x10x1xf32>
    %110 = stablehlo.add %105, %109 : tensor<1x512x10x1xf32>
    %111 = stablehlo.multiply %110, %61 : tensor<1x512x10x1xf32>
    %112 = stablehlo.reduce(%66 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %114 = stablehlo.transpose %113, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %115 = stablehlo.multiply %114, %71 : tensor<1x512x10x1xf32>
    %116 = stablehlo.add %111, %115 : tensor<1x512x10x1xf32>
    %117 = stablehlo.multiply %116, %88 : tensor<1x512x10x1xf32>
    %118 = stablehlo.reduce(%93 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x128xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %125 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %125 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %120 = stablehlo.transpose %119, dims = [0, 2, 1, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x512x10x1xf32>
    %121 = stablehlo.multiply %120, %98 : tensor<1x512x10x1xf32>
    %122 = stablehlo.add %117, %121 : tensor<1x512x10x1xf32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2, 3] : (tensor<1x512x10x1xf32>) -> tensor<1x512x10x64xf32>
    %124 = stablehlo.divide %101, %123 : tensor<1x512x10x64xf32>
    return %124 : tensor<1x512x10x64xf32>
  }
}
