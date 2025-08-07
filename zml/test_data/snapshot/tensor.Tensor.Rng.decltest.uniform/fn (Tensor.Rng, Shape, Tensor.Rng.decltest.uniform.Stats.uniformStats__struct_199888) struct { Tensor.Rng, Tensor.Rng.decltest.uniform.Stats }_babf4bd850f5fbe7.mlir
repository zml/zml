module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2xui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i32}) -> (tensor<2xui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_4 = stablehlo.constant dense<-2.000000e+00> : tensor<f32>
    %cst_5 = stablehlo.constant dense<1.200000e+01> : tensor<f32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<9> : tensor<ui32>
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1024xui32>)
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024xui32>
    %1 = stablehlo.shift_right_logical %output, %0 : tensor<1024xui32>
    %2 = stablehlo.bitcast_convert %cst_6 : (tensor<f32>) -> tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1024xui32>
    %4 = stablehlo.or %1, %3 : tensor<1024xui32>
    %5 = stablehlo.bitcast_convert %4 : (tensor<1024xui32>) -> tensor<1024xf32>
    %6 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %7 = stablehlo.subtract %5, %6 : tensor<1024xf32>
    %8 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1024xf32>
    %10 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %11 = stablehlo.add %9, %10 : tensor<1024xf32>
    %12 = stablehlo.reduce(%11 init: %cst_3) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1024xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %27 = stablehlo.add %arg2, %arg1 : tensor<f32>
      stablehlo.return %27 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %15 = stablehlo.divide %13, %14 : tensor<1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<1xf32>) -> tensor<1024xf32>
    %17 = stablehlo.subtract %11, %16 : tensor<1024xf32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %19 = stablehlo.power %17, %18 : tensor<1024xf32>
    %20 = stablehlo.reduce(%19 init: %cst_3) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1024xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %27 = stablehlo.add %arg2, %arg1 : tensor<f32>
      stablehlo.return %27 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %22 = stablehlo.divide %21, %14 : tensor<1xf32>
    %23 = stablehlo.reduce(%11 init: %cst_0) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1024xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %27 = stablehlo.minimum %arg2, %arg1 : tensor<f32>
      stablehlo.return %27 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %25 = stablehlo.reduce(%11 init: %cst) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1024xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %27 = stablehlo.maximum %arg2, %arg1 : tensor<f32>
      stablehlo.return %27 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<1xf32>
    return %output_state, %15, %22, %24, %26 : tensor<2xui64>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
  }
}
