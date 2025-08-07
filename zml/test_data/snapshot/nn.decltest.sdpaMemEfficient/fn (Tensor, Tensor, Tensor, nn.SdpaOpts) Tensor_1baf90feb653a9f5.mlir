module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<512x512xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x10x512x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x10x512x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x10x512x64xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x10x512x64xf32>, tensor<1x10x512x64xf32>) -> tensor<1x10x512x512xf32>
    %3 = stablehlo.broadcast_in_dim %arg3, dims = [2, 3] : (tensor<512x512xf32>) -> tensor<1x10x512x512xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x10x512x512xf32>
    %5 = stablehlo.reduce(%4 init: %cst_0) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x512xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %20 = stablehlo.maximum %arg5, %arg4 : tensor<f32>
      stablehlo.return %20 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x10x512x1xf32>
    %8 = stablehlo.compare  GT, %6, %7,  FLOAT : (tensor<1x10x512x1xf32>, tensor<1x10x512x1xf32>) -> tensor<1x10x512x1xi1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xi1>) -> tensor<1x10x512x512xi1>
    %10 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x10x512x512xf32>
    %11 = stablehlo.subtract %4, %10 : tensor<1x10x512x512xf32>
    %12 = stablehlo.exponential %11 : tensor<1x10x512x512xf32>
    %13 = stablehlo.reduce(%12 init: %cst) across dimensions = [3] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1x10x512x512xf32>, tensor<f32>) -> tensor<1x10x512xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %20 = stablehlo.add %arg5, %arg4 : tensor<f32>
      stablehlo.return %20 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1x10x512xf32>) -> tensor<1x10x512x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x10x512x1xf32>) -> tensor<1x10x512x512xf32>
    %16 = stablehlo.divide %12, %15 : tensor<1x10x512x512xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x10x512x512xf32>
    %18 = stablehlo.select %9, %16, %17 : tensor<1x10x512x512xi1>, tensor<1x10x512x512xf32>
    %19 = stablehlo.dot_general %18, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x10x512x512xf32>, tensor<1x10x512x64xf32>) -> tensor<1x10x512x64xf32>
    return %19 : tensor<1x10x512x64xf32>
  }
}
